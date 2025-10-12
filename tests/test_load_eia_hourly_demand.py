"""
Test functions for load_eia_hourly_demand.py script
"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Import functions from the script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "powergenome" / "scripts" / "powerlascopf_data_helpers"))
from load_eia_hourly_demand import (
    parse_eia_timestamp,
    read_csv_file,
    read_json_file,
    aggregate_demand_data,
    create_powerlascopf_output,
    load_eia_hourly_demand,
)


def test_parse_eia_timestamp():
    """Test parsing of EIA timestamp formats."""
    # Test with p.m. format
    ts1 = parse_eia_timestamp("10/11/2025 1 p.m. EDT")
    assert ts1.month == 10
    assert ts1.day == 11
    assert ts1.year == 2025
    assert ts1.hour == 13  # 1 p.m. is 13:00
    
    # Test with a.m. format
    ts2 = parse_eia_timestamp("10/11/2025 3 a.m. PDT")
    assert ts2.hour == 3
    
    # Test without timezone
    ts3 = parse_eia_timestamp("10/11/2025 5 p.m.")
    assert ts3.hour == 17


def test_read_csv_file():
    """Test reading EIA demand data from CSV file."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('"Region Code","Timestamp (Hour Ending)","Demand (MWh)","Percent Change from Prior Hour"\n')
        f.write('US48,"10/11/2025 1 a.m. EDT",426315.93,0%\n')
        f.write('US48,"10/11/2025 2 a.m. EDT",415234.12,"-2.6%"\n')
        f.write('CA,"10/11/2025 1 a.m. PDT",45123.67,0%\n')
        temp_path = f.name
    
    try:
        df = read_csv_file(temp_path)
        
        # Check columns
        assert 'region' in df.columns
        assert 'timestamp' in df.columns
        assert 'demand_mwh' in df.columns
        
        # Check data
        assert len(df) == 3
        assert 'US48' in df['region'].values
        assert 'CA' in df['region'].values
        
        # Check demand values
        us_demand = df[df['region'] == 'US48'].iloc[0]['demand_mwh']
        assert us_demand == 426315.93
        
    finally:
        Path(temp_path).unlink()


def test_read_json_file():
    """Test reading EIA demand data from JSON file."""
    # Create a temporary JSON file with JSON lines format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"respondent":"US48","dataType":"D","Timestamp (Hour Ending)":"10/11/2025 1 a.m. EDT","value":426315.93,"percentChange":"0%"}\n')
        f.write('{"respondent":"CA","dataType":"D","Timestamp (Hour Ending)":"10/11/2025 1 a.m. PDT","value":45123.67,"percentChange":"0%"}\n')
        temp_path = f.name
    
    try:
        df = read_json_file(temp_path)
        
        # Check columns
        assert 'region' in df.columns
        assert 'timestamp' in df.columns
        assert 'demand_mwh' in df.columns
        
        # Check data
        assert len(df) == 2
        assert 'US48' in df['region'].values
        assert 'CA' in df['region'].values
        
    finally:
        Path(temp_path).unlink()


def test_aggregate_demand_data():
    """Test aggregating demand data from multiple DataFrames."""
    # Create sample dataframes
    df1 = pd.DataFrame({
        'region': ['US48', 'US48'],
        'timestamp': pd.to_datetime(['2025-10-11 01:00', '2025-10-11 02:00']),
        'demand_mwh': [100.0, 105.0]
    })
    
    df2 = pd.DataFrame({
        'region': ['CA', 'CA'],
        'timestamp': pd.to_datetime(['2025-10-11 01:00', '2025-10-11 02:00']),
        'demand_mwh': [50.0, 52.0]
    })
    
    result = aggregate_demand_data([df1, df2])
    
    # Check result
    assert len(result) == 4  # 2 regions × 2 timestamps
    assert result['region'].nunique() == 2
    assert 'US48' in result['region'].values
    assert 'CA' in result['region'].values


def test_create_powerlascopf_output():
    """Test converting aggregated data to PowerLASCOPF format."""
    # Create sample data
    df = pd.DataFrame({
        'region': ['US48', 'CA', 'US48', 'CA'],
        'timestamp': pd.to_datetime([
            '2025-10-11 01:00', '2025-10-11 01:00',
            '2025-10-11 02:00', '2025-10-11 02:00'
        ]),
        'demand_mwh': [100.0, 50.0, 105.0, 52.0]
    })
    
    result = create_powerlascopf_output(df)
    
    # Check columns
    assert 'Time_Index' in result.columns
    assert 'Load_MW_US48' in result.columns
    assert 'Load_MW_CA' in result.columns
    
    # Check data
    assert len(result) == 2  # 2 time steps
    assert result['Time_Index'].tolist() == [1, 2]
    assert result['Load_MW_US48'].iloc[0] == 100.0
    assert result['Load_MW_CA'].iloc[0] == 50.0


def test_load_eia_hourly_demand_integration():
    """Integration test for the complete workflow."""
    # Create temporary input files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('"Region Code","Timestamp (Hour Ending)","Demand (MWh)","Percent Change from Prior Hour"\n')
        f.write('US48,"10/11/2025 1 a.m. EDT",100.0,0%\n')
        f.write('US48,"10/11/2025 2 a.m. EDT",105.0,"5%"\n')
        csv_path = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"respondent":"CA","Timestamp (Hour Ending)":"10/11/2025 1 a.m. PDT","value":50.0}\n')
        f.write('{"respondent":"CA","Timestamp (Hour Ending)":"10/11/2025 2 a.m. PDT","value":52.0}\n')
        json_path = f.name
    
    # Create temporary output file path
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_path = f.name
    
    try:
        # Run the main function
        load_eia_hourly_demand([csv_path, json_path], output_path)
        
        # Verify output file exists
        assert Path(output_path).exists()
        
        # Read and verify output
        output_df = pd.read_csv(output_path)
        
        # Check structure
        assert 'Time_Index' in output_df.columns
        assert 'Load_MW_US48' in output_df.columns
        assert 'Load_MW_CA' in output_df.columns
        
        # Check data
        assert len(output_df) == 2
        assert output_df['Time_Index'].tolist() == [1, 2]
        
    finally:
        Path(csv_path).unlink()
        Path(json_path).unlink()
        Path(output_path).unlink()


def test_error_handling_missing_file():
    """Test that missing files are handled gracefully."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_path = f.name
    
    try:
        # Try to load from non-existent file
        with pytest.raises(ValueError, match="No data was successfully loaded"):
            load_eia_hourly_demand(['/nonexistent/file.csv'], output_path)
    finally:
        if Path(output_path).exists():
            Path(output_path).unlink()


def test_error_handling_invalid_csv():
    """Test that invalid CSV format is handled."""
    # Create a CSV with missing required columns
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('col1,col2\n')
        f.write('val1,val2\n')
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Missing required columns"):
            read_csv_file(temp_path)
    finally:
        Path(temp_path).unlink()
