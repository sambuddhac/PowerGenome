"""Command-line interface for PowerLASCOPF data export

This script allows users to export PowerGenome data in PowerLASCOPF.jl format
directly from the command line. It follows a similar pattern to the existing
GenX export CLI.

Example usage:
    python run_lascopf_export.py --settings_file settings --results_folder lascopf_output
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime as dt
from pathlib import Path

import pandas as pd

import powergenome
from powergenome.external_data import make_generator_variability
from powergenome.fuels import fuel_cost_table
from powergenome.generators import GeneratorClusters
from powergenome.load_profiles import make_final_load_curves
from powergenome.PowerLASCOPF import (
    process_lascopf_data,
    write_lascopf_data,
)
from powergenome.transmission import (
    agg_transmission_constraints,
    transmission_line_distance,
)
from powergenome.util import (
    build_scenario_settings,
    init_pudl_connection,
    load_ipm_shapefile,
    load_settings,
    remove_fuel_scenario_name,
)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def parse_command_line(argv):
    """Parse command line arguments for PowerLASCOPF export.
    
    Parameters
    ----------
    argv : list
        Command line arguments including caller file name
        
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Export PowerGenome data to PowerLASCOPF.jl format"
    )
    parser.add_argument(
        "-sf",
        "--settings_file",
        dest="settings_file",
        type=str,
        default="example_settings.yml",
        help="Specify a YAML settings file.",
    )
    parser.add_argument(
        "-rf",
        "--results_folder",
        dest="results_folder",
        type=str,
        default=dt.now().strftime("%Y-%m-%d %H.%M.%S"),
        help="Specify the results subfolder to write output",
    )
    parser.add_argument(
        "--no-current-gens",
        dest="current_gens",
        action="store_false",
        help="Don't load and cluster current generators.",
    )
    parser.add_argument(
        "--no-gens",
        dest="gens",
        action="store_false",
        help="Use flag to not calculate generator clusters.",
    )
    parser.add_argument(
        "--no-load",
        dest="load",
        action="store_false",
        help="Use flag to not calculate load profiles.",
    )
    parser.add_argument(
        "--no-transmission",
        dest="transmission",
        action="store_false",
        help="Use flag to not calculate transmission constraints.",
    )
    parser.add_argument(
        "--sort-gens",
        dest="sort_gens",
        action="store_true",
        help="Sort generators alphabetically within each region.",
    )
    
    return parser.parse_args(argv[1:])


def main(**kwargs):
    """Main function to export PowerGenome data to PowerLASCOPF format.
    
    Parameters
    ----------
    **kwargs : dict
        Keyword arguments to override command line arguments
    """
    args = parse_command_line(sys.argv)
    args.__dict__.update(kwargs)
    cwd = Path.cwd()
    
    out_folder = cwd / args.results_folder
    out_folder.mkdir(exist_ok=True)
    
    # Set up logging
    logger = logging.getLogger(powergenome.__name__)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    handler = logging.StreamHandler()
    stream_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
        "%H:%M:%S",
    )
    handler.setFormatter(stream_formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # File handler
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)s %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    filehandler = logging.FileHandler(out_folder / "log.txt")
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(file_formatter)
    logger.addHandler(filehandler)
    
    logger.info("Starting PowerLASCOPF data export")
    logger.info("Reading settings file")
    settings = load_settings(path=args.settings_file)
    
    # Copy the settings file to results folder
    if Path(args.settings_file).is_file():
        shutil.copy(args.settings_file, out_folder)
    else:
        shutil.copytree(
            args.settings_file, out_folder / "pg_settings", dirs_exist_ok=True
        )
    
    logger.debug("Initiating PUDL connections")
    pudl_engine, pudl_out, pg_engine = init_pudl_connection(
        freq="AS",
        start_year=min(settings.get("eia_data_years")),
        end_year=max(settings.get("eia_data_years")),
    )
    
    # Build scenario settings
    scenario_definitions = pd.read_csv(
        settings["input_folder"] / settings["scenario_definitions_fn"]
    )
    scenario_settings = build_scenario_settings(settings, scenario_definitions)
    
    # Create zone number mapping
    zones = settings.get("model_regions")
    if zones:
        zone_num_map = {
            zone: f"{index + 1}" for index, zone in enumerate(zones)
        }
    else:
        zone_num_map = {}
    
    logger.info(f"Found {len(scenario_settings)} scenarios to process")
    
    # Process each scenario
    for scenario, _settings in scenario_settings.items():
        logger.info(f"\n\nStarting scenario {scenario}\n\n")
        
        # Create scenario folder
        case_folder = out_folder / scenario
        case_folder.mkdir(parents=True, exist_ok=True)
        
        # Dictionary to hold all data for this scenario
        case_data = {}
        
        # Generate generator data
        if args.gens:
            logger.info("Generating generator clusters")
            gc = GeneratorClusters(
                pudl_engine=pudl_engine,
                pudl_out=pudl_out,
                pg_engine=pg_engine,
                settings=_settings,
                current_gens=args.current_gens,
                sort_gens=args.sort_gens,
            )
            gen_data = gc.create_all_generators()
            gen_data["Zone"] = gen_data["region"].map(zone_num_map)
            case_data["gen_data"] = gen_data
            
            # Generate variability profiles
            logger.info("Generating generation variability profiles")
            gen_variability = make_generator_variability(gen_data)
            gen_variability.index.name = "Time_Index"
            gen_variability.columns = gen_data["Resource"]
            case_data["gen_variability"] = gen_variability
        
        # Generate load data
        if args.load:
            logger.info("Generating load profiles")
            load = make_final_load_curves(pudl_engine=pudl_engine, settings=_settings)
            load.columns = "Load_MW_z" + load.columns.map(zone_num_map)
            case_data["demand_data"] = load
        
        # Generate transmission data
        if args.transmission:
            logger.info("Generating transmission constraints")
            if args.gens:
                model_regions_gdf = gc.model_regions_gdf
            else:
                model_regions_gdf = load_ipm_shapefile(_settings)
            
            transmission = agg_transmission_constraints(
                pudl_engine=pudl_engine, settings=_settings
            )
            transmission = transmission.pipe(
                transmission_line_distance,
                ipm_shapefile=model_regions_gdf,
                settings=_settings,
                units="mile",
            )
            case_data["network"] = transmission
        
        # Process and write PowerLASCOPF data
        logger.info("Converting data to PowerLASCOPF format")
        lascopf_data = process_lascopf_data(
            case_folder=case_folder,
            data_dict=case_data,
            settings=_settings
        )
        
        logger.info("Writing PowerLASCOPF data files")
        write_lascopf_data(lascopf_data, settings=_settings)
        
        logger.info(f"Completed scenario {scenario}")
    
    logger.info("\n\nAll scenarios completed successfully!\n")
    logger.info(f"Output files are in: {out_folder}")


if __name__ == "__main__":
    main()
