"""Tests for PowerLASCOPF export functionality"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from powergenome.PowerLASCOPF import (
    PowerLASCOPFInputData,
    convert_buses_to_lascopf,
    convert_generation_profiles_to_lascopf,
    convert_generators_to_lascopf,
    convert_load_to_lascopf,
    convert_network_to_lascopf,
    process_lascopf_data,
    write_lascopf_data,
)


@pytest.fixture
def sample_gen_data():
    """Create sample generator data for testing."""
    return pd.DataFrame({
        "Resource": ["Coal_1", "NG_CC_1", "Wind_1", "Solar_1"],
        "Zone": [1, 1, 2, 2],
        "Existing_Cap_MW": [500.0, 300.0, 200.0, 150.0],
        "Min_Power": [0.4, 0.3, 0.0, 0.0],
        "Var_OM_Cost_per_MWh": [2.5, 3.0, 0.5, 0.2],
        "Ramp_Up_Percentage": [0.5, 0.8, 1.0, 1.0],
        "Ramp_Dn_Percentage": [0.5, 0.8, 1.0, 1.0],
        "Start_Cost_per_MW": [50.0, 30.0, 0.0, 0.0],
        "Up_Time": [4, 2, 0, 0],
        "Down_Time": [4, 2, 0, 0],
    })


@pytest.fixture
def sample_network_data():
    """Create sample network data for testing."""
    return pd.DataFrame({
        "Network_Lines": ["z1_to_z2", "z2_to_z3"],
        "Line_Max_Flow_MW": [1000.0, 800.0],
        "distance_mile": [100.0, 150.0],
    })


@pytest.fixture
def sample_load_data():
    """Create sample load time-series data for testing."""
    return pd.DataFrame({
        "Time_Index": [1, 2, 3],
        "Load_MW_z1": [1000.0, 1100.0, 1050.0],
        "Load_MW_z2": [800.0, 850.0, 820.0],
    })


@pytest.fixture
def sample_gen_variability():
    """Create sample generation variability data for testing."""
    return pd.DataFrame({
        "Time_Index": [1, 2, 3],
        "Coal_1": [1.0, 1.0, 1.0],
        "NG_CC_1": [1.0, 1.0, 1.0],
        "Wind_1": [0.3, 0.5, 0.7],
        "Solar_1": [0.0, 0.8, 0.6],
    })


class TestPowerLASCOPFInputData:
    """Tests for PowerLASCOPFInputData dataclass."""

    def test_valid_initialization(self, tmp_path):
        """Test valid initialization of PowerLASCOPFInputData."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        data = PowerLASCOPFInputData(
            tag="TEST",
            folder=tmp_path,
            file_name="test.csv",
            dataframe=df,
        )
        assert data.tag == "TEST"
        assert data.folder == tmp_path
        assert data.file_name == "test.csv"
        assert data.dataframe.equals(df)

    def test_invalid_file_extension(self, tmp_path):
        """Test that non-CSV file names raise ValueError."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(ValueError, match="file_name must end with .csv"):
            PowerLASCOPFInputData(
                tag="TEST",
                folder=tmp_path,
                file_name="test.txt",
                dataframe=df,
            )

    def test_path_conversion(self):
        """Test that string folder paths are converted to Path objects."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        data = PowerLASCOPFInputData(
            tag="TEST",
            folder="test_folder",
            file_name="test.csv",
            dataframe=df,
        )
        assert isinstance(data.folder, Path)


class TestConvertGenerators:
    """Tests for convert_generators_to_lascopf function."""

    def test_basic_conversion(self, sample_gen_data):
        """Test basic generator data conversion."""
        result = convert_generators_to_lascopf(sample_gen_data)
        
        assert not result.empty
        assert "bus_id" in result.columns
        assert "gen_id" in result.columns
        assert "pg_min" in result.columns
        assert "pg_max" in result.columns
        assert "qg_min" in result.columns
        assert "qg_max" in result.columns

    def test_power_limits(self, sample_gen_data):
        """Test that power limits are calculated correctly."""
        result = convert_generators_to_lascopf(sample_gen_data)
        
        # Check pg_max matches input capacity
        assert result.loc[0, "pg_max"] == 500.0
        
        # Check pg_min calculated from Min_Power fraction
        assert result.loc[0, "pg_min"] == 500.0 * 0.4  # 200 MW
        
        # Check reactive power limits
        assert result.loc[0, "qg_min"] == -0.3 * 500.0
        assert result.loc[0, "qg_max"] == 0.3 * 500.0

    def test_cost_conversion(self, sample_gen_data):
        """Test cost coefficient conversion."""
        result = convert_generators_to_lascopf(sample_gen_data)
        
        # Check linear cost coefficient
        assert result.loc[0, "cost_coef_c1"] == 2.5
        assert result.loc[1, "cost_coef_c1"] == 3.0
        
        # Check quadratic and constant terms default to 0
        assert (result["cost_coef_c2"] == 0.0).all()
        assert (result["cost_coef_c0"] == 0.0).all()

    def test_ramp_rates(self, sample_gen_data):
        """Test ramp rate conversion."""
        result = convert_generators_to_lascopf(sample_gen_data)
        
        # Check ramp rates in MW
        assert result.loc[0, "ramp_up"] == 500.0 * 0.5  # 250 MW
        assert result.loc[0, "ramp_down"] == 500.0 * 0.5

    def test_startup_costs(self, sample_gen_data):
        """Test startup cost conversion."""
        result = convert_generators_to_lascopf(sample_gen_data)
        
        # Check total startup cost
        assert result.loc[0, "startup_cost"] == 50.0 * 500.0  # $25,000

    def test_empty_dataframe(self):
        """Test handling of empty generator dataframe."""
        result = convert_generators_to_lascopf(pd.DataFrame())
        assert result.empty

    def test_none_input(self):
        """Test handling of None input."""
        result = convert_generators_to_lascopf(None)
        assert result.empty


class TestConvertBuses:
    """Tests for convert_buses_to_lascopf function."""

    def test_basic_bus_creation(self, sample_gen_data):
        """Test basic bus data creation from generator zones."""
        result = convert_buses_to_lascopf(sample_gen_data)
        
        assert not result.empty
        assert "bus_id" in result.columns
        assert "bus_type" in result.columns
        assert len(result) == 2  # Two unique zones

    def test_bus_types(self, sample_gen_data):
        """Test that bus types are assigned correctly."""
        result = convert_buses_to_lascopf(sample_gen_data)
        
        # First bus should be reference (type 3)
        assert result.loc[0, "bus_type"] == 3
        
        # Other buses should be PV (type 2)
        assert (result.loc[1:, "bus_type"] == 2).all()

    def test_voltage_parameters(self, sample_gen_data):
        """Test voltage parameter defaults."""
        result = convert_buses_to_lascopf(sample_gen_data)
        
        assert (result["vm"] == 1.0).all()  # Per unit voltage
        assert (result["va"] == 0.0).all()  # Voltage angle
        assert (result["vmax"] == 1.05).all()
        assert (result["vmin"] == 0.95).all()
        assert (result["base_kv"] == 345.0).all()

    def test_load_integration(self, sample_gen_data, sample_load_data):
        """Test that load data is integrated into bus data."""
        result = convert_buses_to_lascopf(sample_gen_data, sample_load_data)
        
        # Check that load is added to buses
        # Average of [1000, 1100, 1050] = 1050
        assert result.loc[result["bus_id"] == 1, "pd"].iloc[0] == pytest.approx(1050.0)

    def test_empty_generator_data(self):
        """Test handling of empty generator data."""
        result = convert_buses_to_lascopf(pd.DataFrame())
        assert result.empty


class TestConvertNetwork:
    """Tests for convert_network_to_lascopf function."""

    def test_basic_conversion(self, sample_network_data):
        """Test basic network data conversion."""
        result = convert_network_to_lascopf(sample_network_data)
        
        assert not result.empty
        assert "from_bus" in result.columns
        assert "to_bus" in result.columns
        assert "r" in result.columns
        assert "x" in result.columns

    def test_bus_extraction(self, sample_network_data):
        """Test extraction of from/to buses from line names."""
        result = convert_network_to_lascopf(sample_network_data)
        
        # Check first line: z1_to_z2
        assert result.loc[0, "from_bus"] == 1.0
        assert result.loc[0, "to_bus"] == 2.0

    def test_electrical_parameters(self, sample_network_data):
        """Test electrical parameter calculations."""
        result = convert_network_to_lascopf(sample_network_data)
        
        # Check impedance scales with distance
        assert result.loc[0, "r"] < result.loc[1, "r"]  # Shorter line has less resistance
        assert result.loc[0, "x"] < result.loc[1, "x"]

    def test_power_limits(self, sample_network_data):
        """Test power flow limit assignment."""
        result = convert_network_to_lascopf(sample_network_data)
        
        # Check ratings
        assert result.loc[0, "rate_a"] == 1000.0
        assert result.loc[0, "rate_b"] == 1100.0  # 10% overload
        assert result.loc[0, "rate_c"] == 1200.0  # 20% emergency

    def test_line_status(self, sample_network_data):
        """Test that all lines are in service by default."""
        result = convert_network_to_lascopf(sample_network_data)
        
        assert (result["status"] == 1).all()

    def test_empty_network(self):
        """Test handling of empty network data."""
        result = convert_network_to_lascopf(pd.DataFrame())
        assert result.empty


class TestConvertLoad:
    """Tests for convert_load_to_lascopf function."""

    def test_basic_conversion(self, sample_load_data):
        """Test basic load data conversion."""
        result = convert_load_to_lascopf(sample_load_data)
        
        assert not result.empty
        assert "period" in result.columns
        assert "bus_1_p" in result.columns
        assert "bus_2_p" in result.columns

    def test_period_numbering(self, sample_load_data):
        """Test that periods are numbered starting from 1."""
        result = convert_load_to_lascopf(sample_load_data)
        
        assert result["period"].iloc[0] == 1
        assert result["period"].iloc[-1] == len(sample_load_data)

    def test_reactive_power(self, sample_load_data):
        """Test reactive power calculation."""
        result = convert_load_to_lascopf(sample_load_data)
        
        # Check Q calculated from P with power factor 0.95
        assert "bus_1_q" in result.columns
        assert result.loc[0, "bus_1_q"] == pytest.approx(1000.0 * 0.33, rel=0.01)

    def test_empty_load(self):
        """Test handling of empty load data."""
        result = convert_load_to_lascopf(pd.DataFrame())
        assert result.empty


class TestConvertGenerationProfiles:
    """Tests for convert_generation_profiles_to_lascopf function."""

    def test_basic_conversion(self, sample_gen_variability, sample_gen_data):
        """Test basic generation profile conversion."""
        result = convert_generation_profiles_to_lascopf(
            sample_gen_variability, sample_gen_data
        )
        
        assert not result.empty
        assert "period" in result.columns

    def test_capacity_factor_to_mw(self, sample_gen_variability, sample_gen_data):
        """Test conversion from capacity factors to MW."""
        result = convert_generation_profiles_to_lascopf(
            sample_gen_variability, sample_gen_data
        )
        
        # Wind capacity factor 0.3 * 200 MW capacity = 60 MW
        # Note: column naming depends on gen_data index
        wind_cols = [c for c in result.columns if "gen" in c and c != "period"]
        # At least one profile should have values scaled by capacity
        assert any(result[col].max() > 1.0 for col in wind_cols)

    def test_empty_profiles(self, sample_gen_data):
        """Test handling of empty generation profiles."""
        result = convert_generation_profiles_to_lascopf(
            pd.DataFrame(), sample_gen_data
        )
        assert result.empty


class TestProcessLASCOPFData:
    """Tests for process_lascopf_data function."""

    def test_complete_processing(
        self, tmp_path, sample_gen_data, sample_network_data,
        sample_load_data, sample_gen_variability
    ):
        """Test complete data processing pipeline."""
        data_dict = {
            "gen_data": sample_gen_data,
            "network": sample_network_data,
            "demand_data": sample_load_data,
            "gen_variability": sample_gen_variability,
        }
        
        result = process_lascopf_data(tmp_path, data_dict)
        
        # Check that all expected files are created
        assert len(result) > 0
        tags = [item.tag for item in result]
        assert "GENERATORS" in tags
        assert "BUSES" in tags
        assert "BRANCHES" in tags
        assert "LOAD_TIMESERIES" in tags

    def test_partial_data(self, tmp_path, sample_gen_data):
        """Test processing with only generator data."""
        data_dict = {"gen_data": sample_gen_data}
        
        result = process_lascopf_data(tmp_path, data_dict)
        
        # Should still create generators and buses
        assert len(result) >= 2
        tags = [item.tag for item in result]
        assert "GENERATORS" in tags
        assert "BUSES" in tags

    def test_empty_data(self, tmp_path):
        """Test processing with empty data dictionary."""
        data_dict = {}
        
        result = process_lascopf_data(tmp_path, data_dict)
        
        # Should return empty list or list with empty dataframes
        assert isinstance(result, list)


class TestWriteLASCOPFData:
    """Tests for write_lascopf_data function."""

    def test_file_creation(self, tmp_path):
        """Test that CSV files are created correctly."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        data_list = [
            PowerLASCOPFInputData(
                tag="TEST",
                folder=tmp_path,
                file_name="test.csv",
                dataframe=df,
            )
        ]
        
        write_lascopf_data(data_list)
        
        # Check file was created
        assert (tmp_path / "test.csv").exists()
        
        # Check content
        result = pd.read_csv(tmp_path / "test.csv")
        assert result.equals(df)

    def test_empty_dataframe_skipped(self, tmp_path, caplog):
        """Test that empty dataframes are skipped with warning."""
        data_list = [
            PowerLASCOPFInputData(
                tag="EMPTY",
                folder=tmp_path,
                file_name="empty.csv",
                dataframe=pd.DataFrame(),
            )
        ]
        
        with caplog.at_level(logging.WARNING):
            write_lascopf_data(data_list)
        
        # File should not be created
        assert not (tmp_path / "empty.csv").exists()
        
        # Warning should be logged
        assert "Skipping empty dataframe" in caplog.text

    def test_folder_creation(self, tmp_path):
        """Test that output folders are created if they don't exist."""
        subfolder = tmp_path / "new_folder" / "subfolder"
        df = pd.DataFrame({"col1": [1, 2, 3]})
        data_list = [
            PowerLASCOPFInputData(
                tag="TEST",
                folder=subfolder,
                file_name="test.csv",
                dataframe=df,
            )
        ]
        
        write_lascopf_data(data_list)
        
        # Check folder and file were created
        assert subfolder.exists()
        assert (subfolder / "test.csv").exists()


class TestIntegration:
    """Integration tests for complete export workflow."""

    def test_full_export_workflow(
        self, tmp_path, sample_gen_data, sample_network_data,
        sample_load_data, sample_gen_variability
    ):
        """Test complete export workflow from data dict to files."""
        data_dict = {
            "gen_data": sample_gen_data,
            "network": sample_network_data,
            "demand_data": sample_load_data,
            "gen_variability": sample_gen_variability,
        }
        
        # Process data
        lascopf_data = process_lascopf_data(tmp_path, data_dict)
        
        # Write data
        write_lascopf_data(lascopf_data)
        
        # Check all expected files exist
        lascopf_folder = tmp_path / "lascopf_data"
        assert (lascopf_folder / "generators.csv").exists()
        assert (lascopf_folder / "buses.csv").exists()
        assert (lascopf_folder / "branches.csv").exists()
        assert (lascopf_folder / "load_timeseries.csv").exists()
        
        # Verify file contents are non-empty
        gens = pd.read_csv(lascopf_folder / "generators.csv")
        assert len(gens) == 4  # Four generators in sample data

    def test_data_consistency(
        self, tmp_path, sample_gen_data, sample_network_data
    ):
        """Test that exported data maintains consistency."""
        data_dict = {
            "gen_data": sample_gen_data,
            "network": sample_network_data,
        }
        
        lascopf_data = process_lascopf_data(tmp_path, data_dict)
        write_lascopf_data(lascopf_data)
        
        # Load exported data
        lascopf_folder = tmp_path / "lascopf_data"
        gens = pd.read_csv(lascopf_folder / "generators.csv")
        buses = pd.read_csv(lascopf_folder / "buses.csv")
        branches = pd.read_csv(lascopf_folder / "branches.csv")
        
        # Check consistency
        # All generator buses should exist in bus list
        gen_buses = set(gens["bus_id"].unique())
        bus_ids = set(buses["bus_id"].unique())
        assert gen_buses.issubset(bus_ids)
        
        # All branch buses should exist
        branch_buses = set(branches["from_bus"].unique()) | set(branches["to_bus"].unique())
        assert branch_buses.issubset(bus_ids)
