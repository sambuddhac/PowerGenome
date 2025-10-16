"""Functions for PowerLASCOPF.jl data export

This module provides conversion functions to export PowerGenome data
in formats compatible with PowerLASCOPF.jl for AC optimal power flow analysis.

PowerLASCOPF.jl typically requires:
- Generator/bus data with voltage and power specifications
- Network topology and line parameters (impedance, admittance, limits)
- Load data with real and reactive power
- Time-series generation and load profiles

The module is designed to be modular and extensible for different branches
and versions of PowerLASCOPF.jl.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from powergenome.util import snake_case_col

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PowerLASCOPFInputData:
    """A class to abstract the dataframes for PowerLASCOPF output files."""

    tag: str
    folder: Union[str, Path]
    file_name: str
    dataframe: pd.DataFrame

    def __post_init__(self):
        """Validate the data after initialization."""
        if not isinstance(self.folder, Path):
            object.__setattr__(self, "folder", Path(self.folder))
        if not self.file_name.endswith(".csv"):
            raise ValueError(f"file_name must end with .csv, got {self.file_name}")
        if not isinstance(self.dataframe, (pd.DataFrame, type(None))):
            raise TypeError(
                f"dataframe must be a pandas DataFrame or None, got {type(self.dataframe)}"
            )


def convert_generators_to_lascopf(
    gen_data: pd.DataFrame,
    settings: Optional[Dict] = None
) -> pd.DataFrame:
    """Convert PowerGenome generator data to PowerLASCOPF format.
    
    PowerLASCOPF expects generator data with electrical characteristics
    including voltage levels, power limits, and cost parameters.
    
    Parameters
    ----------
    gen_data : pd.DataFrame
        Generator data from PowerGenome with resource characteristics
    settings : Dict, optional
        Settings dictionary for customization
        
    Returns
    -------
    pd.DataFrame
        Generator data formatted for PowerLASCOPF
        
    Notes
    -----
    Expected PowerLASCOPF generator columns:
    - bus_id: Bus number where generator is connected
    - gen_id: Unique generator identifier
    - pg_min: Minimum active power output (MW)
    - pg_max: Maximum active power output (MW)
    - qg_min: Minimum reactive power output (MVAr)
    - qg_max: Maximum reactive power output (MVAr)
    - cost_coef_c2: Quadratic cost coefficient
    - cost_coef_c1: Linear cost coefficient
    - cost_coef_c0: Constant cost coefficient
    - ramp_up: Ramp up rate (MW/period)
    - ramp_down: Ramp down rate (MW/period)
    - startup_cost: Startup cost ($)
    - shutdown_cost: Shutdown cost ($)
    - min_up_time: Minimum up time (periods)
    - min_down_time: Minimum down time (periods)
    """
    if gen_data is None or gen_data.empty:
        logger.warning("Empty generator data provided for PowerLASCOPF conversion")
        return pd.DataFrame()
    
    logger.info("Converting generator data to PowerLASCOPF format")
    
    # Create a copy to avoid modifying original data
    lascopf_gen = gen_data.copy()
    
    # Map PowerGenome columns to PowerLASCOPF format
    column_mapping = {
        "Zone": "bus_id",
        "Resource": "gen_id",
        "Min_Power": "pg_min_fraction",
        "Existing_Cap_MW": "pg_max",
        "Cap_Size": "unit_size_mw",
    }
    
    # Apply column renaming where columns exist
    for pg_col, lascopf_col in column_mapping.items():
        if pg_col in lascopf_gen.columns:
            lascopf_gen = lascopf_gen.rename(columns={pg_col: lascopf_col})
    
    # Calculate absolute minimum power in MW
    if "pg_min_fraction" in lascopf_gen.columns and "pg_max" in lascopf_gen.columns:
        lascopf_gen["pg_min"] = (
            lascopf_gen["pg_min_fraction"] * lascopf_gen["pg_max"]
        )
    else:
        lascopf_gen["pg_min"] = 0.0
    
    # Set default reactive power limits if not provided
    # Typical range is ±30-40% of active power capacity
    if "pg_max" in lascopf_gen.columns:
        lascopf_gen["qg_min"] = -0.3 * lascopf_gen["pg_max"]
        lascopf_gen["qg_max"] = 0.3 * lascopf_gen["pg_max"]
    
    # Convert variable O&M and fuel costs to cost coefficients
    # Assuming linear cost function for now: cost = c1 * pg
    if "Var_OM_Cost_per_MWh" in lascopf_gen.columns:
        lascopf_gen["cost_coef_c1"] = lascopf_gen["Var_OM_Cost_per_MWh"]
    else:
        lascopf_gen["cost_coef_c1"] = 0.0
    
    # Quadratic and constant terms default to 0
    lascopf_gen["cost_coef_c2"] = 0.0
    lascopf_gen["cost_coef_c0"] = 0.0
    
    # Ramp rates
    if "Ramp_Up_Percentage" in lascopf_gen.columns and "pg_max" in lascopf_gen.columns:
        lascopf_gen["ramp_up"] = (
            lascopf_gen["Ramp_Up_Percentage"] * lascopf_gen["pg_max"]
        )
    else:
        lascopf_gen["ramp_up"] = lascopf_gen.get("pg_max", 0.0)
    
    if "Ramp_Dn_Percentage" in lascopf_gen.columns and "pg_max" in lascopf_gen.columns:
        lascopf_gen["ramp_down"] = (
            lascopf_gen["Ramp_Dn_Percentage"] * lascopf_gen["pg_max"]
        )
    else:
        lascopf_gen["ramp_down"] = lascopf_gen.get("pg_max", 0.0)
    
    # Startup/shutdown costs
    if "Start_Cost_per_MW" in lascopf_gen.columns and "pg_max" in lascopf_gen.columns:
        lascopf_gen["startup_cost"] = (
            lascopf_gen["Start_Cost_per_MW"] * lascopf_gen["pg_max"]
        )
    else:
        lascopf_gen["startup_cost"] = 0.0
    
    lascopf_gen["shutdown_cost"] = 0.0  # PowerGenome doesn't typically have this
    
    # Minimum up/down times
    if "Up_Time" in lascopf_gen.columns:
        lascopf_gen["min_up_time"] = lascopf_gen["Up_Time"]
    else:
        lascopf_gen["min_up_time"] = 0
    
    if "Down_Time" in lascopf_gen.columns:
        lascopf_gen["min_down_time"] = lascopf_gen["Down_Time"]
    else:
        lascopf_gen["min_down_time"] = 0
    
    # Select final columns for PowerLASCOPF
    required_cols = [
        "bus_id", "gen_id", "pg_min", "pg_max",
        "qg_min", "qg_max",
        "cost_coef_c2", "cost_coef_c1", "cost_coef_c0",
        "ramp_up", "ramp_down",
        "startup_cost", "shutdown_cost",
        "min_up_time", "min_down_time"
    ]
    
    # Keep only columns that exist
    output_cols = [col for col in required_cols if col in lascopf_gen.columns]
    
    return lascopf_gen[output_cols]


def convert_buses_to_lascopf(
    gen_data: pd.DataFrame,
    load_data: Optional[pd.DataFrame] = None,
    settings: Optional[Dict] = None
) -> pd.DataFrame:
    """Convert zone/region data to PowerLASCOPF bus data.
    
    Parameters
    ----------
    gen_data : pd.DataFrame
        Generator data containing zone information
    load_data : pd.DataFrame, optional
        Load data by zone
    settings : Dict, optional
        Settings dictionary for customization
        
    Returns
    -------
    pd.DataFrame
        Bus data formatted for PowerLASCOPF
        
    Notes
    -----
    Expected PowerLASCOPF bus columns:
    - bus_id: Unique bus identifier
    - bus_type: 1=PQ, 2=PV, 3=Ref (slack)
    - pd: Active power demand (MW)
    - qd: Reactive power demand (MVAr)
    - gs: Shunt conductance (per unit)
    - bs: Shunt susceptance (per unit)
    - area: Area number
    - vm: Voltage magnitude (per unit)
    - va: Voltage angle (degrees)
    - base_kv: Base voltage (kV)
    - zone: Zone number
    - vmax: Maximum voltage (per unit)
    - vmin: Minimum voltage (per unit)
    """
    if gen_data is None or gen_data.empty:
        logger.warning("Empty generator data provided for bus conversion")
        return pd.DataFrame()
    
    logger.info("Creating PowerLASCOPF bus data from zones")
    
    # Extract unique zones/buses from generator data
    if "Zone" in gen_data.columns:
        unique_zones = gen_data["Zone"].unique()
    elif "bus_id" in gen_data.columns:
        unique_zones = gen_data["bus_id"].unique()
    else:
        logger.error("No zone or bus_id column found in generator data")
        return pd.DataFrame()
    
    # Create bus dataframe
    buses = pd.DataFrame({
        "bus_id": sorted(unique_zones),
    })
    
    # Set first bus as reference (slack) bus, others as PV (generator) buses
    buses["bus_type"] = 2  # PV bus (generator bus)
    buses.loc[0, "bus_type"] = 3  # Reference/slack bus
    
    # Default electrical parameters
    buses["pd"] = 0.0  # Will be set from load data if available
    buses["qd"] = 0.0
    buses["gs"] = 0.0
    buses["bs"] = 0.0
    buses["area"] = 1
    buses["vm"] = 1.0  # Per unit voltage magnitude
    buses["va"] = 0.0  # Voltage angle in degrees
    buses["base_kv"] = 345.0  # Typical transmission voltage
    buses["zone"] = buses["bus_id"]
    buses["vmax"] = 1.05  # +/- 5% voltage limits typical
    buses["vmin"] = 0.95
    
    # Add load data if provided
    if load_data is not None and not load_data.empty:
        # Sum load across time if time-series data provided
        if "Time_Index" in load_data.columns or load_data.index.name == "Time_Index":
            if "Time_Index" in load_data.columns:
                load_data = load_data.set_index("Time_Index")
            # Average load across time periods for base case
            avg_load = load_data.mean()
        else:
            avg_load = load_data.iloc[0] if len(load_data) > 0 else pd.Series()
        
        # Map load to buses
        for col in avg_load.index:
            # Extract zone number from column name (e.g., "Load_MW_z1" -> 1)
            if "z" in str(col).lower():
                try:
                    zone_num = int(str(col).split("z")[-1].split("_")[0])
                    if zone_num in buses["bus_id"].values:
                        buses.loc[buses["bus_id"] == zone_num, "pd"] = avg_load[col]
                        # Assume power factor of 0.95, calculate reactive power
                        buses.loc[buses["bus_id"] == zone_num, "qd"] = (
                            avg_load[col] * 0.33  # tan(acos(0.95))
                        )
                except (ValueError, IndexError):
                    continue
    
    return buses


def convert_network_to_lascopf(
    network_data: pd.DataFrame,
    settings: Optional[Dict] = None
) -> pd.DataFrame:
    """Convert PowerGenome network/transmission data to PowerLASCOPF format.
    
    Parameters
    ----------
    network_data : pd.DataFrame
        Network transmission data from PowerGenome
    settings : Dict, optional
        Settings dictionary for customization
        
    Returns
    -------
    pd.DataFrame
        Branch/line data formatted for PowerLASCOPF
        
    Notes
    -----
    Expected PowerLASCOPF branch columns:
    - from_bus: From bus number
    - to_bus: To bus number
    - r: Resistance (per unit)
    - x: Reactance (per unit)
    - b: Total line charging susceptance (per unit)
    - rate_a: MVA rating A (long term)
    - rate_b: MVA rating B (short term)
    - rate_c: MVA rating C (emergency)
    - tap_ratio: Transformer off-nominal tap ratio
    - shift_angle: Transformer phase shift angle (degrees)
    - status: Branch status (1=in-service, 0=out-of-service)
    - angmin: Minimum angle difference (degrees)
    - angmax: Maximum angle difference (degrees)
    """
    if network_data is None or network_data.empty:
        logger.warning("Empty network data provided for PowerLASCOPF conversion")
        return pd.DataFrame()
    
    logger.info("Converting network data to PowerLASCOPF format")
    
    # Create a copy to avoid modifying original data
    lascopf_network = network_data.copy()
    
    # Map PowerGenome columns to PowerLASCOPF format
    column_mapping = {
        "Network_Lines": "line_name",
        "Line_Max_Flow_MW": "rate_a",
    }
    
    # Apply column renaming where columns exist
    for pg_col, lascopf_col in column_mapping.items():
        if pg_col in lascopf_network.columns:
            lascopf_network = lascopf_network.rename(columns={pg_col: lascopf_col})
    
    # Extract from_bus and to_bus from Network_Lines or z columns
    if "line_name" in lascopf_network.columns:
        # Try to extract from_bus and to_bus from line_name using robust regex patterns
        # Handles formats like "z1_z2", "z1z2", "bus1_bus2", "bus1-bus2", etc.
        extract_df = lascopf_network["line_name"].str.extract(
            r"(?:(?:z|bus)?(\d+)[_\- ]*(?:z|bus)?(\d+))", expand=True
        )
        extract_df = extract_df.astype(float)
        # If extraction fails (NaN), try fallback to z1/z2 columns or log a warning
        if extract_df.isnull().any(axis=None):
            logger.warning(
                "Could not reliably extract from_bus and to_bus from some line_name values. "
                "Falling back to z1/z2 columns if available."
            )
            if "z1" in lascopf_network.columns and "z2" in lascopf_network.columns:
                lascopf_network["from_bus"] = lascopf_network["z1"]
                lascopf_network["to_bus"] = lascopf_network["z2"]
            else:
                raise ValueError(
                    "Failed to extract from_bus and to_bus from line_name, and z1/z2 columns are not available."
                )
        else:
            lascopf_network[["from_bus", "to_bus"]] = extract_df
    elif "z1" in lascopf_network.columns and "z2" in lascopf_network.columns:
        lascopf_network["from_bus"] = lascopf_network["z1"]
        lascopf_network["to_bus"] = lascopf_network["z2"]
    
    # Default electrical parameters for transmission lines
    # Using typical values for high-voltage transmission
    # These should be customized based on actual line parameters
    
    # Resistance (r) and reactance (x) in per unit
    # Typical X/R ratio for transmission lines is 10:1
    if "distance_mile" in lascopf_network.columns:
        # Approximate: 0.01 ohm/mile for 345 kV lines
        # Convert to per unit assuming 100 MVA base
        lascopf_network["r"] = lascopf_network["distance_mile"] * 0.00001
        lascopf_network["x"] = lascopf_network["distance_mile"] * 0.0001
    else:
        lascopf_network["r"] = 0.001
        lascopf_network["x"] = 0.01
    
    # Line charging susceptance
    lascopf_network["b"] = 0.02
    
    # Power flow limits
    if "rate_a" not in lascopf_network.columns:
        lascopf_network["rate_a"] = 1000.0  # Default 1000 MVA
    
    lascopf_network["rate_b"] = lascopf_network["rate_a"] * 1.1  # 10% overload
    lascopf_network["rate_c"] = lascopf_network["rate_a"] * 1.2  # 20% emergency
    
    # Transformer parameters (default to regular line)
    lascopf_network["tap_ratio"] = 1.0
    lascopf_network["shift_angle"] = 0.0
    
    # Line status (all in service)
    lascopf_network["status"] = 1
    
    # Angle difference limits (typically ±30 degrees)
    lascopf_network["angmin"] = -30.0
    lascopf_network["angmax"] = 30.0
    
    # Select final columns for PowerLASCOPF
    required_cols = [
        "from_bus", "to_bus",
        "r", "x", "b",
        "rate_a", "rate_b", "rate_c",
        "tap_ratio", "shift_angle",
        "status",
        "angmin", "angmax"
    ]
    
    # Keep only columns that exist
    output_cols = [col for col in required_cols if col in lascopf_network.columns]
    
    return lascopf_network[output_cols]


def convert_load_to_lascopf(
    load_data: pd.DataFrame,
    settings: Optional[Dict] = None
) -> pd.DataFrame:
    """Convert PowerGenome load data to PowerLASCOPF time-series format.
    
    Parameters
    ----------
    load_data : pd.DataFrame
        Load time-series data from PowerGenome
    settings : Dict, optional
        Settings dictionary for customization
        
    Returns
    -------
    pd.DataFrame
        Load time-series formatted for PowerLASCOPF
        
    Notes
    -----
    Output format has columns for each bus load in MW:
    - period: Time period index
    - bus_1_p: Active power for bus 1 (MW)
    - bus_1_q: Reactive power for bus 1 (MVAr)
    - bus_2_p: Active power for bus 2 (MW)
    - bus_2_q: Reactive power for bus 2 (MVAr)
    - etc.
    """
    if load_data is None or load_data.empty:
        logger.warning("Empty load data provided for PowerLASCOPF conversion")
        return pd.DataFrame()
    
    logger.info("Converting load data to PowerLASCOPF format")
    
    # Create a copy to avoid modifying original data
    lascopf_load = load_data.copy()
    
    # Ensure Time_Index is in the dataframe
    if "Time_Index" in lascopf_load.columns:
        lascopf_load = lascopf_load.set_index("Time_Index")
    
    # Create period column
    lascopf_load.insert(0, "period", range(1, len(lascopf_load) + 1))
    
    # Rename load columns to PowerLASCOPF format
    # From "Load_MW_z1" to "bus_1_p"
    col_mapping = {}
    for col in lascopf_load.columns:
        if "Load_MW_z" in col or "Demand_MW_z" in col:
            # Extract zone number
            zone_num = col.split("z")[-1]
            col_mapping[col] = f"bus_{zone_num}_p"
    
    lascopf_load = lascopf_load.rename(columns=col_mapping)
    
    # Calculate reactive power (Q) from active power (P)
    # Assuming typical power factor of 0.95
    for col in lascopf_load.columns:
        if col.endswith("_p") and col != "period":
            bus_num = col.split("_")[1]
            lascopf_load[f"bus_{bus_num}_q"] = (
                lascopf_load[col] * 0.33  # tan(acos(0.95))
            )
    
    return lascopf_load


def convert_generation_profiles_to_lascopf(
    gen_variability: pd.DataFrame,
    gen_data: pd.DataFrame,
    settings: Optional[Dict] = None
) -> pd.DataFrame:
    """Convert PowerGenome generation profiles to PowerLASCOPF format.
    
    Parameters
    ----------
    gen_variability : pd.DataFrame
        Time-series generation capacity factors
    gen_data : pd.DataFrame
        Generator data with bus assignments
    settings : Dict, optional
        Settings dictionary for customization
        
    Returns
    -------
    pd.DataFrame
        Generation profiles formatted for PowerLASCOPF
        
    Notes
    -----
    Output format:
    - period: Time period index
    - gen_1_pmax: Available capacity for generator 1 (MW)
    - gen_2_pmax: Available capacity for generator 2 (MW)
    - etc.
    """
    if gen_variability is None or gen_variability.empty:
        logger.warning("Empty generation variability data provided")
        return pd.DataFrame()
    
    logger.info("Converting generation profiles to PowerLASCOPF format")
    
    # Create a copy
    lascopf_profiles = gen_variability.copy()
    
    # Ensure Time_Index is in the dataframe
    if "Time_Index" in lascopf_profiles.columns:
        lascopf_profiles = lascopf_profiles.set_index("Time_Index")
    
    # Create period column
    lascopf_profiles.insert(0, "period", range(1, len(lascopf_profiles) + 1))
    
    # Map resource names to generator IDs and convert capacity factors to MW
    if gen_data is not None and not gen_data.empty:
        for col in lascopf_profiles.columns:
            if col == "period":
                continue
            
            # Find matching generator
            matching_gen = gen_data[gen_data["Resource"] == col]
            if not matching_gen.empty:
                # Convert capacity factor to MW
                max_cap = matching_gen["Existing_Cap_MW"].iloc[0]
                lascopf_profiles[col] = lascopf_profiles[col] * max_cap
                
                # Rename to gen_X_pmax format
                gen_id = matching_gen.index[0] + 1  # 1-indexed
                lascopf_profiles = lascopf_profiles.rename(
                    columns={col: f"gen_{gen_id}_pmax"}
                )
    
    return lascopf_profiles


def process_lascopf_data(
    case_folder: Path,
    data_dict: Dict[str, pd.DataFrame],
    settings: Optional[Dict] = None
) -> List[PowerLASCOPFInputData]:
    """Process data dictionary and return list of PowerLASCOPF output files.
    
    Parameters
    ----------
    case_folder : Path
        Path to the case folder for outputs
    data_dict : Dict[str, pd.DataFrame]
        Dictionary containing PowerGenome data:
        - gen_data: Generator characteristics
        - gen_variability: Time-series generation profiles
        - demand_data: Load time-series
        - network: Transmission network data
    settings : Dict, optional
        Settings dictionary for customization
        
    Returns
    -------
    List[PowerLASCOPFInputData]
        List of data objects to be written as CSV files
    """
    logger.info("Processing data for PowerLASCOPF export")
    
    # Create output folder structure
    lascopf_folder = case_folder / "lascopf_data"
    lascopf_folder.mkdir(parents=True, exist_ok=True)
    
    output_data = []
    
    # Convert and add generator data
    if "gen_data" in data_dict and data_dict["gen_data"] is not None:
        gen_lascopf = convert_generators_to_lascopf(
            data_dict["gen_data"], settings
        )
        output_data.append(
            PowerLASCOPFInputData(
                tag="GENERATORS",
                folder=lascopf_folder,
                file_name="generators.csv",
                dataframe=gen_lascopf,
            )
        )
        
        # Create bus data from generators
        bus_data = convert_buses_to_lascopf(
            data_dict["gen_data"],
            data_dict.get("demand_data"),
            settings
        )
        output_data.append(
            PowerLASCOPFInputData(
                tag="BUSES",
                folder=lascopf_folder,
                file_name="buses.csv",
                dataframe=bus_data,
            )
        )
    
    # Convert and add network data
    if "network" in data_dict and data_dict["network"] is not None:
        network_lascopf = convert_network_to_lascopf(
            data_dict["network"], settings
        )
        output_data.append(
            PowerLASCOPFInputData(
                tag="BRANCHES",
                folder=lascopf_folder,
                file_name="branches.csv",
                dataframe=network_lascopf,
            )
        )
    
    # Convert and add load data
    if "demand_data" in data_dict and data_dict["demand_data"] is not None:
        load_lascopf = convert_load_to_lascopf(
            data_dict["demand_data"], settings
        )
        output_data.append(
            PowerLASCOPFInputData(
                tag="LOAD_TIMESERIES",
                folder=lascopf_folder,
                file_name="load_timeseries.csv",
                dataframe=load_lascopf,
            )
        )
    
    # Convert and add generation profiles
    if (
        "gen_variability" in data_dict 
        and data_dict["gen_variability"] is not None
        and "gen_data" in data_dict
    ):
        profiles_lascopf = convert_generation_profiles_to_lascopf(
            data_dict["gen_variability"],
            data_dict["gen_data"],
            settings
        )
        output_data.append(
            PowerLASCOPFInputData(
                tag="GEN_PROFILES",
                folder=lascopf_folder,
                file_name="generation_profiles.csv",
                dataframe=profiles_lascopf,
            )
        )
    
    logger.info(f"Created {len(output_data)} PowerLASCOPF data files")
    
    return output_data


def write_lascopf_data(
    data_list: List[PowerLASCOPFInputData],
    settings: Optional[Dict] = None
) -> None:
    """Write PowerLASCOPF data files to disk.
    
    Parameters
    ----------
    data_list : List[PowerLASCOPFInputData]
        List of PowerLASCOPF data objects to write
    settings : Dict, optional
        Settings dictionary for customization
    """
    logger.info("Writing PowerLASCOPF data files")
    
    for data in data_list:
        if data.dataframe is not None and not data.dataframe.empty:
            # Create folder if it doesn't exist
            data.folder.mkdir(parents=True, exist_ok=True)
            
            # Write CSV file
            output_path = data.folder / data.file_name
            data.dataframe.to_csv(output_path, index=False)
            logger.info(f"Wrote {data.tag} to {output_path}")
        else:
            logger.warning(f"Skipping empty dataframe for {data.tag}")
