"""
Load EIA Hourly Demand Data

This script reads EIA hourly demand data from CSV and/or JSON files and converts
them into PowerLASCOPF-compatible format.

Input formats supported:
- CSV: "Region Code","Timestamp (Hour Ending)","Demand (MWh)","Percent Change from Prior Hour"
- JSON: {"respondent":"US48","dataType":"D","Timestamp (Hour Ending)":"10/11/2025 1 p.m. EDT","value":426315.93,"percentChange":"0%"}

Output format:
- CSV with Time_Index column and Load_MW_{region} columns for each region
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_eia_timestamp(timestamp_str: str) -> pd.Timestamp:
    """
    Parse EIA timestamp format to pandas Timestamp.
    
    EIA format example: "10/11/2025 1 p.m. EDT"
    
    Parameters
    ----------
    timestamp_str : str
        EIA formatted timestamp string
        
    Returns
    -------
    pd.Timestamp
        Parsed timestamp
    """
    try:
        # EIA timestamps include timezone abbreviations that pandas may not parse directly
        # Remove timezone abbreviation and parse the datetime
        # Example: "10/11/2025 1 p.m. EDT" -> "10/11/2025 1 p.m."
        parts = timestamp_str.rsplit(' ', 1)
        if len(parts) == 2:
            timestamp_str_no_tz = parts[0]
        else:
            timestamp_str_no_tz = timestamp_str
        
        # Try different format patterns
        # Handle both "p.m." and "a.m." with and without periods
        formats_to_try = [
            '%m/%d/%Y %I %p.',  # "10/11/2025 1 p.m."
            '%m/%d/%Y %I %p',   # "10/11/2025 1 pm"
            '%m/%d/%Y %I:%M %p.',  # "10/11/2025 1:00 p.m."
            '%m/%d/%Y %I:%M %p',   # "10/11/2025 1:00 pm"
        ]
        
        for fmt in formats_to_try:
            try:
                return pd.to_datetime(timestamp_str_no_tz, format=fmt)
            except ValueError:
                continue
        
        # If none of the explicit formats work, try pandas flexible parsing
        return pd.to_datetime(timestamp_str_no_tz)
        
    except Exception as e:
        logger.error(f"Could not parse timestamp '{timestamp_str}': {e}")
        raise ValueError(f"Could not parse timestamp '{timestamp_str}': {e}") from e


def read_csv_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read EIA hourly demand data from CSV file.
    
    Expected columns:
    - "Region Code": Region identifier (e.g., "US48")
    - "Timestamp (Hour Ending)": Hour ending timestamp
    - "Demand (MWh)": Demand in MWh
    - "Percent Change from Prior Hour": (optional, not used)
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: region, timestamp, demand_mwh
    """
    logger.info(f"Reading CSV file: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ["Region Code", "Timestamp (Hour Ending)", "Demand (MWh)"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract and rename columns
        result = pd.DataFrame({
            'region': df['Region Code'],
            'timestamp': df['Timestamp (Hour Ending)'].apply(parse_eia_timestamp),
            'demand_mwh': df['Demand (MWh)']
        })
        
        logger.info(f"Successfully read {len(result)} records from CSV")
        return result
        
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise RuntimeError(f"Failed to read CSV file {file_path}: {e}") from e


def read_json_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Read EIA hourly demand data from JSON file.
    
    Expected format (one JSON object per line or array of objects):
    {"respondent":"US48","dataType":"D","Timestamp (Hour Ending)":"10/11/2025 1 p.m. EDT","value":426315.93,"percentChange":"0%"}
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to JSON file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: region, timestamp, demand_mwh
    """
    logger.info(f"Reading JSON file: {file_path}")
    
    try:
        # Try reading as JSON lines first
        with open(file_path, 'r') as f:
            content = f.read().strip()
            
        # Check if it's a JSON array or JSON lines
        if content.startswith('['):
            # JSON array
            data = json.loads(content)
        else:
            # JSON lines (one object per line)
            data = [json.loads(line) for line in content.split('\n') if line.strip()]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Map JSON fields to standard columns
        # Handle different possible field names
        region_col = None
        timestamp_col = None
        value_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'respondent' in col_lower or 'region' in col_lower:
                region_col = col
            elif 'timestamp' in col_lower or 'hour ending' in col_lower:
                timestamp_col = col
            elif 'value' in col_lower or 'demand' in col_lower:
                value_col = col
        
        if not all([region_col, timestamp_col, value_col]):
            raise ValueError(
                f"Could not identify required fields in JSON. "
                f"Found columns: {list(df.columns)}"
            )
        
        # Extract and rename columns
        result = pd.DataFrame({
            'region': df[region_col],
            'timestamp': df[timestamp_col].apply(parse_eia_timestamp),
            'demand_mwh': df[value_col]
        })
        
        logger.info(f"Successfully read {len(result)} records from JSON")
        return result
        
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        raise RuntimeError(f"Failed to read JSON file {file_path}: {e}") from e


def aggregate_demand_data(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate demand data from multiple DataFrames.
    
    Combines data from multiple sources, handling multiple regions and time periods.
    
    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames with columns: region, timestamp, demand_mwh
        
    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with same structure
    """
    logger.info(f"Aggregating {len(dataframes)} dataframes")
    
    if not dataframes:
        raise ValueError("No dataframes to aggregate")
    
    # Concatenate all dataframes
    combined = pd.concat(dataframes, ignore_index=True)
    
    # Remove duplicates (same region and timestamp)
    # If duplicates exist, take the mean of demand values
    combined = combined.groupby(['region', 'timestamp'], as_index=False).agg({
        'demand_mwh': 'mean'
    })
    
    logger.info(f"Aggregated to {len(combined)} total records across {combined['region'].nunique()} regions")
    
    return combined


def create_powerlascopf_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert aggregated demand data to PowerLASCOPF format.
    
    Output format:
    - Time_Index: Integer index starting from 1
    - Load_MW_{region}: Demand in MW for each region
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: region, timestamp, demand_mwh
        
    Returns
    -------
    pd.DataFrame
        PowerLASCOPF-compatible DataFrame
    """
    logger.info("Converting to PowerLASCOPF format")
    
    # Sort by timestamp and region
    df = df.sort_values(['timestamp', 'region']).copy()
    
    # Pivot to wide format: each region becomes a column
    pivot_df = df.pivot(index='timestamp', columns='region', values='demand_mwh')
    
    # Reset index to get timestamp as a column
    pivot_df = pivot_df.reset_index()
    
    # Sort by timestamp
    pivot_df = pivot_df.sort_values('timestamp')
    
    # Create Time_Index (1-based indexing)
    pivot_df['Time_Index'] = range(1, len(pivot_df) + 1)
    
    # Rename region columns to Load_MW_{region} format
    region_cols = [col for col in pivot_df.columns if col != 'timestamp' and col != 'Time_Index']
    rename_dict = {col: f'Load_MW_{col}' for col in region_cols}
    pivot_df = pivot_df.rename(columns=rename_dict)
    
    # Reorder columns: Time_Index first, then Load_MW columns
    load_cols = [col for col in pivot_df.columns if col.startswith('Load_MW_')]
    final_cols = ['Time_Index'] + sorted(load_cols)
    
    result = pivot_df[final_cols].copy()
    
    # Ensure Time_Index is integer
    result['Time_Index'] = result['Time_Index'].astype(int)
    
    logger.info(f"Created PowerLASCOPF format with {len(result)} time steps and {len(load_cols)} regions")
    
    return result


def load_eia_hourly_demand(
    input_files: List[Union[str, Path]],
    output_file: Union[str, Path]
) -> None:
    """
    Main function to load EIA hourly demand data and convert to PowerLASCOPF format.
    
    Parameters
    ----------
    input_files : List[Union[str, Path]]
        List of input CSV and/or JSON files
    output_file : Union[str, Path]
        Path to output CSV file
    """
    logger.info(f"Starting EIA hourly demand conversion with {len(input_files)} input files")
    
    # Read all input files
    dataframes = []
    for file_path in input_files:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                df = read_csv_file(file_path)
            elif suffix == '.json':
                df = read_json_file(file_path)
            else:
                logger.warning(f"Unsupported file format: {suffix} for file {file_path}")
                continue
            
            dataframes.append(df)
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No data was successfully loaded from input files")
    
    # Aggregate all data
    combined_df = aggregate_demand_data(dataframes)
    
    # Convert to PowerLASCOPF format
    output_df = create_powerlascopf_output(combined_df)
    
    # Write output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_df.to_csv(output_path, index=False)
    logger.info(f"Successfully wrote output to: {output_path}")
    logger.info(f"Output shape: {output_df.shape[0]} rows, {output_df.shape[1]} columns")


def main():
    """Command-line interface for the script."""
    parser = argparse.ArgumentParser(
        description="Convert EIA hourly demand data to PowerLASCOPF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single CSV file
  python load_eia_hourly_demand.py input.csv output.csv
  
  # Multiple files (CSV and JSON)
  python load_eia_hourly_demand.py input1.csv input2.json output.csv
  
  # With explicit input flag
  python load_eia_hourly_demand.py -i file1.csv -i file2.json -o output.csv
        """
    )
    
    parser.add_argument(
        'files',
        nargs='*',
        help='Input files (CSV and/or JSON) followed by output file. '
             'Last argument is treated as output file.'
    )
    
    parser.add_argument(
        '-i', '--input',
        action='append',
        dest='input_files',
        help='Input file (can be specified multiple times)'
    )
    
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine input and output files
    input_files = []
    output_file = None
    
    if args.input_files and args.output_file:
        # Explicit input/output flags
        input_files = args.input_files
        output_file = args.output_file
    elif args.files:
        # Positional arguments: last one is output, rest are inputs
        if len(args.files) < 2:
            parser.error("At least one input file and one output file are required")
        input_files = args.files[:-1]
        output_file = args.files[-1]
    else:
        parser.error("No input or output files specified")
    
    try:
        load_eia_hourly_demand(input_files, output_file)
        logger.info("Conversion completed successfully")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise RuntimeError("Conversion failed during EIA hourly demand data processing") from e


if __name__ == "__main__":
    main()
