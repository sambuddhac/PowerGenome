"""
Load and aggregate NREL renewable profile data for PowerLASCOPF.

This script processes one or more NREL renewable profile CSV files (Wind Toolkit, 
NSRDB, ATB, etc.) and converts them to PowerLASCOPF-compatible format.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def detect_column_mapping(
    df: pd.DataFrame, 
    site_col: Optional[str] = None,
    time_col: Optional[str] = None,
    output_col: Optional[str] = None
) -> dict:
    """
    Detect column names for site/location, timestamp, and generation/output.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame from NREL CSV file
    site_col : str, optional
        User-specified site/location column name
    time_col : str, optional
        User-specified timestamp column name
    output_col : str, optional
        User-specified generation/output column name
        
    Returns
    -------
    dict
        Mapping of standardized names to actual column names
        
    Raises
    ------
    ValueError
        If required columns cannot be detected
    """
    columns = df.columns.tolist()
    mapping = {}
    
    # Site/location column detection (only if explicitly requested or strongly indicated)
    if site_col and site_col in columns:
        mapping['site'] = site_col
    else:
        # Only detect site column if it's a clear match (not just partial)
        site_candidates = ['site_id', 'location_id', 'loc_id', 'bus_id', 
                          'cpa_id', 'gid']
        for candidate in site_candidates:
            matches = [col for col in columns if col.lower() == candidate.lower()]
            if matches:
                mapping['site'] = matches[0]
                logger.info(f"Detected site column: {matches[0]}")
                break
    
    # Time column detection
    if time_col and time_col in columns:
        mapping['time'] = time_col
    else:
        time_candidates = ['time', 'timestamp', 'datetime', 'date', 'hour', 
                          'time_index', 'period']
        for candidate in time_candidates:
            matches = [col for col in columns if candidate.lower() in col.lower()]
            if matches:
                mapping['time'] = matches[0]
                logger.info(f"Detected time column: {matches[0]}")
                break
    
    # Output/generation column detection
    if output_col and output_col in columns:
        mapping['output'] = output_col
    else:
        output_candidates = ['output', 'generation', 'power', 'mw', 'capacity_factor',
                           'cf', 'gen', 'renewable_mw']
        for candidate in output_candidates:
            matches = [col for col in columns if candidate.lower() in col.lower()]
            if matches:
                mapping['output'] = matches[0]
                logger.info(f"Detected output column: {matches[0]}")
                break
    
    # Validate required columns
    if 'time' not in mapping:
        # If no time column, check if index looks like time
        if df.index.name and any(t in str(df.index.name).lower() 
                                for t in ['time', 'date', 'hour']):
            mapping['time'] = df.index.name
            logger.info(f"Using index as time column: {df.index.name}")
        else:
            raise ValueError(
                f"Could not detect time column. Available columns: {columns}. "
                "Please specify with --time-col parameter."
            )
    
    # Check if this might be wide format (time column + multiple data columns)
    # In wide format, output column is not needed as each column represents a site
    if 'output' not in mapping and 'site' not in mapping:
        # This could be wide format - we'll handle it in load_nrel_profile
        pass
    elif 'output' not in mapping:
        raise ValueError(
            f"Could not detect output/generation column. Available columns: {columns}. "
            "Please specify with --output-col parameter."
        )
    
    return mapping


def load_nrel_profile(
    file_path: Union[str, Path],
    site_col: Optional[str] = None,
    time_col: Optional[str] = None,
    output_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Load a single NREL renewable profile CSV file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the NREL CSV file
    site_col : str, optional
        Name of site/location column
    time_col : str, optional
        Name of timestamp column
    output_col : str, optional
        Name of generation/output column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns: site_id, time_index, renewable_mw
    """
    file_path = Path(file_path)
    logger.info(f"Loading NREL profile from {file_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Detect column mapping
        col_mapping = detect_column_mapping(df, site_col, time_col, output_col)
        
        # Handle time as index if needed
        if col_mapping['time'] == df.index.name:
            df = df.reset_index()
        
        # Check if data is in wide format (multiple sites as columns)
        if 'site' not in col_mapping and 'output' not in col_mapping:
            # Wide format: time column + multiple site columns
            time_data = df[col_mapping['time']]
            site_cols = [c for c in df.columns if c != col_mapping['time']]
            
            if len(site_cols) == 0:
                raise ValueError("No site data columns found in wide format")
            
            logger.info(f"Detected wide format with {len(site_cols)} site columns")
            
            # Melt to long format
            df_long = df.melt(
                id_vars=[col_mapping['time']], 
                value_vars=site_cols,
                var_name='site_id',
                value_name='renewable_mw'
            )
            df_long = df_long.rename(columns={col_mapping['time']: 'time_index'})
            
        elif 'site' in col_mapping and 'output' in col_mapping:
            # Long format: explicit site, time, and output columns
            df_long = df[[col_mapping['site'], col_mapping['time'], col_mapping['output']]].copy()
            df_long.columns = ['site_id', 'time_index', 'renewable_mw']
        else:
            raise ValueError(
                "Ambiguous data format. Could not determine if data is in wide or long format. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Ensure proper data types
        df_long['renewable_mw'] = pd.to_numeric(df_long['renewable_mw'], errors='coerce')
        
        # Remove any NaN values
        initial_rows = len(df_long)
        df_long = df_long.dropna()
        if len(df_long) < initial_rows:
            logger.warning(f"Dropped {initial_rows - len(df_long)} rows with NaN values")
        
        logger.info(f"Processed {len(df_long)} rows for {df_long['site_id'].nunique()} sites")
        return df_long
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise


def aggregate_profiles(
    profile_files: List[Union[str, Path]],
    site_col: Optional[str] = None,
    time_col: Optional[str] = None,
    output_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Aggregate multiple NREL profile files into a single DataFrame.
    
    Parameters
    ----------
    profile_files : list of str or Path
        List of paths to NREL CSV files
    site_col : str, optional
        Name of site/location column (if same across all files)
    time_col : str, optional
        Name of timestamp column (if same across all files)
    output_col : str, optional
        Name of generation/output column (if same across all files)
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all sites and time periods
    """
    logger.info(f"Aggregating {len(profile_files)} profile files")
    
    profile_dfs = []
    for file_path in profile_files:
        df = load_nrel_profile(file_path, site_col, time_col, output_col)
        profile_dfs.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(profile_dfs, ignore_index=True)
    
    # Remove duplicates (same site and time)
    initial_rows = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['site_id', 'time_index'])
    if len(combined_df) < initial_rows:
        logger.warning(
            f"Removed {initial_rows - len(combined_df)} duplicate site-time combinations"
        )
    
    # Sort by time and site for consistency
    combined_df = combined_df.sort_values(['time_index', 'site_id']).reset_index(drop=True)
    
    logger.info(
        f"Aggregated to {len(combined_df)} total rows, "
        f"{combined_df['site_id'].nunique()} unique sites, "
        f"{combined_df['time_index'].nunique()} time periods"
    )
    
    return combined_df


def format_for_powerlascopf(
    df: pd.DataFrame,
    bus_id_map: Optional[dict] = None
) -> pd.DataFrame:
    """
    Format aggregated profile data for PowerLASCOPF.
    
    Parameters
    ----------
    df : pd.DataFrame
        Aggregated profile DataFrame with columns: site_id, time_index, renewable_mw
    bus_id_map : dict, optional
        Mapping from site_id to bus_id for PowerLASCOPF
        
    Returns
    -------
    pd.DataFrame
        PowerLASCOPF-compatible DataFrame
    """
    logger.info("Formatting output for PowerLASCOPF")
    
    output_df = df.copy()
    
    # Map site_id to bus_id if mapping provided
    if bus_id_map:
        output_df['bus_id'] = output_df['site_id'].map(bus_id_map)
        # Keep site_id as well for reference
        unmapped = output_df['bus_id'].isna().sum()
        if unmapped > 0:
            logger.warning(
                f"{unmapped} site_ids could not be mapped to bus_ids, "
                "using site_id as bus_id"
            )
            output_df['bus_id'] = output_df['bus_id'].fillna(output_df['site_id'])
    else:
        # Use site_id as bus_id if no mapping provided
        output_df['bus_id'] = output_df['site_id']
    
    # Reorder columns for PowerLASCOPF format
    output_df = output_df[['bus_id', 'time_index', 'renewable_mw']]
    
    logger.info(f"Formatted {len(output_df)} rows for PowerLASCOPF output")
    return output_df


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main function to load, aggregate, and export NREL renewable profiles.
    
    Parameters
    ----------
    args : argparse.Namespace, optional
        Command-line arguments (if None, will parse from sys.argv)
    """
    if args is None:
        args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("Starting NREL renewable profile processing")
    
    # Load and aggregate profiles
    combined_df = aggregate_profiles(
        args.input_files,
        site_col=args.site_col,
        time_col=args.time_col,
        output_col=args.output_col
    )
    
    # Load bus mapping if provided
    bus_id_map = None
    if args.bus_map:
        logger.info(f"Loading bus ID mapping from {args.bus_map}")
        bus_map_df = pd.read_csv(args.bus_map)
        # Assume first two columns are site_id and bus_id
        bus_id_map = dict(zip(bus_map_df.iloc[:, 0], bus_map_df.iloc[:, 1]))
        logger.info(f"Loaded {len(bus_id_map)} bus ID mappings")
    
    # Format for PowerLASCOPF
    output_df = format_for_powerlascopf(combined_df, bus_id_map)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    output_df.to_csv(output_path, index=False)
    logger.info(f"Wrote PowerLASCOPF-compatible output to {output_path}")
    logger.info(
        f"Summary: {output_df['bus_id'].nunique()} buses, "
        f"{output_df['time_index'].nunique()} time periods, "
        f"{len(output_df)} total records"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and aggregate NREL renewable profile CSV files for PowerLASCOPF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file with auto-detection
  python load_nrel_renewable_profiles.py wind_profiles.csv -o output.csv
  
  # Multiple files with explicit column names
  python load_nrel_renewable_profiles.py wind.csv solar.csv \\
    --site-col site_id --time-col timestamp --output-col capacity_factor \\
    -o combined_profiles.csv
  
  # With bus ID mapping
  python load_nrel_renewable_profiles.py profiles.csv \\
    --bus-map site_to_bus_mapping.csv -o output.csv
        """
    )
    
    parser.add_argument(
        'input_files',
        nargs='+',
        help='One or more NREL renewable profile CSV files'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file path for PowerLASCOPF-compatible data'
    )
    
    parser.add_argument(
        '--site-col',
        help='Name of site/location column (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--time-col',
        help='Name of timestamp column (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--output-col',
        help='Name of generation/output column (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--bus-map',
        help='CSV file mapping site IDs to bus IDs (first column: site_id, second: bus_id)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    main()
