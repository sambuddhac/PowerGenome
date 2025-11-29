# PowerLASCOPF Export Guide

This guide explains how to use PowerGenome to export data in PowerLASCOPF.jl-compatible format for AC optimal power flow analysis.

## Overview

PowerLASCOPF.jl is a Julia package for solving optimal power flow (OPF) problems with AC power flow constraints. This export functionality converts PowerGenome's generation, load, and network data into the format required by PowerLASCOPF.jl.

## What Gets Exported

The PowerLASCOPF export creates the following data files:

### 1. Generators (`generators.csv`)
Generator technical and economic parameters:
- **bus_id**: Bus number where generator is connected
- **gen_id**: Unique generator identifier
- **pg_min**: Minimum active power output (MW)
- **pg_max**: Maximum active power output (MW)
- **qg_min**: Minimum reactive power output (MVAr)
- **qg_max**: Maximum reactive power output (MVAr)
- **cost_coef_c2**: Quadratic cost coefficient ($/MW²)
- **cost_coef_c1**: Linear cost coefficient ($/MW)
- **cost_coef_c0**: Constant cost coefficient ($)
- **ramp_up**: Ramp up rate (MW/period)
- **ramp_down**: Ramp down rate (MW/period)
- **startup_cost**: Startup cost ($)
- **shutdown_cost**: Shutdown cost ($)
- **min_up_time**: Minimum up time (periods)
- **min_down_time**: Minimum down time (periods)

### 2. Buses (`buses.csv`)
Electrical bus specifications:
- **bus_id**: Unique bus identifier
- **bus_type**: 1=PQ (load), 2=PV (generator), 3=Ref (slack)
- **pd**: Active power demand (MW)
- **qd**: Reactive power demand (MVAr)
- **gs**: Shunt conductance (per unit)
- **bs**: Shunt susceptance (per unit)
- **area**: Area number
- **vm**: Voltage magnitude (per unit)
- **va**: Voltage angle (degrees)
- **base_kv**: Base voltage (kV)
- **zone**: Zone number
- **vmax**: Maximum voltage (per unit)
- **vmin**: Minimum voltage (per unit)

### 3. Branches (`branches.csv`)
Transmission line and transformer parameters:
- **from_bus**: From bus number
- **to_bus**: To bus number
- **r**: Resistance (per unit)
- **x**: Reactance (per unit)
- **b**: Total line charging susceptance (per unit)
- **rate_a**: MVA rating A (long term)
- **rate_b**: MVA rating B (short term)
- **rate_c**: MVA rating C (emergency)
- **tap_ratio**: Transformer off-nominal tap ratio
- **shift_angle**: Transformer phase shift angle (degrees)
- **status**: Branch status (1=in-service, 0=out-of-service)
- **angmin**: Minimum angle difference (degrees)
- **angmax**: Maximum angle difference (degrees)

### 4. Load Time Series (`load_timeseries.csv`)
Hourly load profiles by bus:
- **period**: Time period index
- **bus_1_p**: Active power for bus 1 (MW)
- **bus_1_q**: Reactive power for bus 1 (MVAr)
- **bus_2_p**: Active power for bus 2 (MW)
- **bus_2_q**: Reactive power for bus 2 (MVAr)
- etc.

### 5. Generation Profiles (`generation_profiles.csv`)
Available generation capacity over time:
- **period**: Time period index
- **gen_1_pmax**: Available capacity for generator 1 (MW)
- **gen_2_pmax**: Available capacity for generator 2 (MW)
- etc.

## Usage

### Command Line Interface

The simplest way to export PowerLASCOPF data is using the command line:

```bash
python -m powergenome.run_lascopf_export --settings_file my_settings --results_folder lascopf_output
```

Or with the shorthand:

```bash
python -m powergenome.run_lascopf_export -sf my_settings -rf lascopf_output
```

### Command Line Options

- `-sf, --settings_file`: Path to your PowerGenome settings file (YAML)
- `-rf, --results_folder`: Name of folder for output files
- `--no-current-gens`: Don't include existing generators
- `--no-gens`: Skip generator data export
- `--no-load`: Skip load profile export
- `--no-transmission`: Skip transmission network export
- `--sort-gens`: Sort generators alphabetically within each region

### Python API

You can also use the export functions directly in Python:

```python
from pathlib import Path
from powergenome.PowerLASCOPF import (
    process_lascopf_data,
    write_lascopf_data,
)
from powergenome.generators import GeneratorClusters
from powergenome.load_profiles import make_final_load_curves

# Assuming you have PowerGenome data already loaded
data_dict = {
    "gen_data": gen_data,
    "gen_variability": gen_variability,
    "demand_data": load_data,
    "network": transmission_data,
}

# Convert to PowerLASCOPF format
case_folder = Path("my_case")
lascopf_data = process_lascopf_data(
    case_folder=case_folder,
    data_dict=data_dict,
    settings=settings
)

# Write output files
write_lascopf_data(lascopf_data, settings=settings)
```

## Data Mapping and Assumptions

### Generator Data Mapping

| PowerGenome Field | PowerLASCOPF Field | Notes |
|-------------------|-------------------|-------|
| Zone | bus_id | Direct mapping |
| Resource | gen_id | Generator name/identifier |
| Min_Power | pg_min | Converted from fraction to MW |
| Existing_Cap_MW | pg_max | Maximum capacity |
| Var_OM_Cost_per_MWh | cost_coef_c1 | Linear cost coefficient |
| Ramp_Up_Percentage | ramp_up | Converted to MW/period |
| Ramp_Dn_Percentage | ramp_down | Converted to MW/period |
| Start_Cost_per_MW | startup_cost | Converted to total $ |
| Up_Time | min_up_time | Direct mapping |
| Down_Time | min_down_time | Direct mapping |

### Default Values and Assumptions

1. **Reactive Power Limits**: Set to ±30% of active power capacity (typical for thermal generators)
2. **Quadratic Cost Term**: Set to 0 (linear cost function assumed)
3. **Shutdown Costs**: Set to 0 (not typically tracked in PowerGenome)
4. **Voltage Parameters**: 
   - Base voltage: 345 kV (typical transmission level)
   - Voltage limits: 0.95 to 1.05 per unit (±5%)
   - Initial voltage: 1.0 per unit
5. **Transmission Line Electrical Parameters**:
   - Resistance (r): ~0.00001 per unit per mile
   - Reactance (x): ~0.0001 per unit per mile
   - X/R ratio: ~10 (typical for transmission)
   - Line charging: 0.02 per unit
6. **Power Factor**: 0.95 assumed for calculating reactive power from active power
7. **Angle Limits**: ±30 degrees (typical stability constraint)

### Bus Type Assignment

- First bus (numerically) is assigned as reference/slack bus (type 3)
- All other buses with generators are PV buses (type 2)
- Buses with only load would be PQ buses (type 1)

## Known Limitations and Data Gaps

### 1. Missing Electrical Parameters

PowerGenome focuses on capacity expansion planning and doesn't track all parameters needed for AC power flow:

- **Line Impedances**: Calculated using typical values and distances. Real impedances should be obtained from utility planning studies or WECC/ERCOT datasets.
- **Transformer Parameters**: Default values used. Actual tap ratios and phase shifts need to be specified for transformers.
- **Shunt Elements**: Not included in base PowerGenome data. Add manually if needed for voltage control.

### 2. Reactive Power

PowerGenome primarily models active power (MW):

- Reactive power limits are estimated based on typical generator capabilities
- Load reactive power calculated assuming 0.95 power factor
- No explicit reactive power resources (capacitors, reactors, SVCs) included

### 3. Voltage Levels

PowerGenome aggregates regions without tracking voltage levels:

- All buses assumed at transmission voltage (345 kV)
- No sub-transmission or distribution modeling
- Voltage constraints may need adjustment for specific studies

### 4. Network Topology

PowerGenome uses simplified inter-regional transmission:

- Full network topology not captured
- May need to add detailed intra-regional transmission
- Radial connections and contingencies not represented

### 5. Generator Location

Generators are aggregated to zone level:

- Exact bus locations within zones not specified
- All generators in a zone mapped to single bus
- May need disaggregation for detailed studies

## Customization

### Adding Custom Electrical Parameters

You can provide custom electrical parameters through the settings file:

```yaml
lascopf_custom_params:
  # Custom line parameters (overrides calculated values)
  line_parameters:
    z1_z2:  # Line name
      r: 0.0015  # Resistance (p.u.)
      x: 0.015   # Reactance (p.u.)
      b: 0.025   # Susceptance (p.u.)
  
  # Custom bus voltages
  bus_voltages:
    1: 345.0  # kV
    2: 500.0  # kV
  
  # Custom generator reactive power limits
  gen_q_limits:
    Coal_1:
      qg_min: -50  # MVAr
      qg_max: 150  # MVAr
```

### Extending the Export Module

The export module is designed to be modular and extensible. To add new data exports:

1. Create a new conversion function in `PowerLASCOPF.py`:
```python
def convert_my_data_to_lascopf(my_data: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    # Your conversion logic
    return converted_data
```

2. Add it to the `process_lascopf_data` function:
```python
output_data.append(
    PowerLASCOPFInputData(
        tag="MY_DATA",
        folder=lascopf_folder,
        file_name="my_data.csv",
        dataframe=convert_my_data_to_lascopf(data_dict["my_data"], settings),
    )
)
```

## Additional Data Sources

### Obtaining Detailed Network Data

For production-quality AC power flow studies, you'll need detailed network data:

1. **WECC/ERCOT Models**: 
   - Contact regional coordinators for planning models
   - Models include full topology and electrical parameters
   - Often available for academic research

2. **Utility Planning Data**:
   - Request from local utilities
   - Typically requires NDA
   - Most accurate for specific regions

3. **IEEE Test Systems**:
   - IEEE 14, 30, 57, 118 bus systems
   - Good for methodology testing
   - Not representative of actual systems

4. **PowerWorld Cases**:
   - Available from PowerWorld website
   - Include full electrical parameters
   - Can be converted to PowerLASCOPF format

### Sample Code for MATPOWER Conversion

If you have MATPOWER case files, you can convert them to PowerLASCOPF format:

```python
import scipy.io as sio

# Load MATPOWER case
mat_case = sio.loadmat('case118.mat')
mpc = mat_case['mpc']

# Extract bus data
bus_data = pd.DataFrame(
    mpc['bus'][0, 0],
    columns=['bus_id', 'bus_type', 'pd', 'qd', 'gs', 'bs', 
             'area', 'vm', 'va', 'base_kv', 'zone', 'vmax', 'vmin']
)

# Extract branch data
branch_data = pd.DataFrame(
    mpc['branch'][0, 0],
    columns=['from_bus', 'to_bus', 'r', 'x', 'b', 
             'rate_a', 'rate_b', 'rate_c', 'tap_ratio', 
             'shift_angle', 'status', 'angmin', 'angmax']
)

# Use with PowerGenome generator data
lascopf_data = process_lascopf_data(
    case_folder=case_folder,
    data_dict={
        "gen_data": pg_gen_data,
        "buses": bus_data,  # Use MATPOWER buses
        "network": branch_data,  # Use MATPOWER branches
    }
)
```

## Compatibility with PowerLASCOPF.jl Branches

This export module is designed to work with multiple versions of PowerLASCOPF.jl:

### Standard Branch
The default export format follows standard optimal power flow data conventions compatible with the main PowerLASCOPF.jl branch.

### Development Branches
For development branches with different data requirements, you can:

1. Modify conversion functions in `PowerLASCOPF.py`
2. Use custom settings to specify format variations
3. Override default parameters through settings file

Example for custom format:

```python
from powergenome.PowerLASCOPF import convert_generators_to_lascopf

# Custom conversion with different column names
def convert_for_dev_branch(gen_data, settings):
    base_conversion = convert_generators_to_lascopf(gen_data, settings)
    # Apply dev branch specific changes
    base_conversion = base_conversion.rename(columns={
        'gen_id': 'generator_name',
        'pg_max': 'max_power_output'
    })
    return base_conversion
```

## Validation and Testing

### Checking Export Outputs

After exporting, validate your data:

```python
import pandas as pd

# Load exported files
gens = pd.read_csv('lascopf_data/generators.csv')
buses = pd.read_csv('lascopf_data/buses.csv')
branches = pd.read_csv('lascopf_data/branches.csv')

# Check for missing values
print(f"Missing values in generators: {gens.isnull().sum().sum()}")
print(f"Missing values in buses: {buses.isnull().sum().sum()}")
print(f"Missing values in branches: {branches.isnull().sum().sum()}")

# Check power balance
total_gen_capacity = gens['pg_max'].sum()
max_load = pd.read_csv('lascopf_data/load_timeseries.csv').iloc[:, 1::2].sum(axis=1).max()
print(f"Generation capacity: {total_gen_capacity:.1f} MW")
print(f"Peak load: {max_load:.1f} MW")
print(f"Reserve margin: {(total_gen_capacity / max_load - 1) * 100:.1f}%")

# Check network connectivity
bus_ids = set(buses['bus_id'])
from_buses = set(branches['from_bus'])
to_buses = set(branches['to_bus'])
all_branch_buses = from_buses | to_buses
isolated_buses = bus_ids - all_branch_buses
print(f"Isolated buses: {isolated_buses}")
```

### Running in PowerLASCOPF.jl

To test your exported data in Julia:

```julia
using PowerLASCOPF, CSV, DataFrames

# Load data
gens = CSV.read("lascopf_data/generators.csv", DataFrame)
buses = CSV.read("lascopf_data/buses.csv", DataFrame)
branches = CSV.read("lascopf_data/branches.csv", DataFrame)
load_ts = CSV.read("lascopf_data/load_timeseries.csv", DataFrame)

# Create PowerLASCOPF case
case = PowerLASCOPFCase(
    generators=gens,
    buses=buses,
    branches=branches,
    load_timeseries=load_ts
)

# Solve OPF
results = solve_opf(case)
```

## Troubleshooting

### Common Issues

1. **"No zone or bus_id column found"**
   - Ensure your settings file specifies `model_regions`
   - Check that generator data includes zone information

2. **"Empty generator data provided"**
   - Verify `--no-gens` flag is not set
   - Check that generator clustering completed successfully

3. **"Missing network data"**
   - Ensure transmission data is included in PowerGenome run
   - Check `--no-transmission` flag is not set

4. **Voltage convergence issues in PowerLASCOPF**
   - Adjust voltage limits in bus data
   - Check for isolated buses or islands
   - Verify reactive power limits are reasonable

5. **Infeasible OPF**
   - Check that total generation capacity exceeds peak load
   - Verify transmission limits are not too restrictive
   - Review generator minimum output constraints

## Support and Contributing

For questions or issues:

1. Check the PowerGenome [Wiki](https://github.com/PowerGenome/PowerGenome/wiki)
2. Open an issue on [GitHub](https://github.com/PowerGenome/PowerGenome/issues)
3. Join the [PowerGenome groups.io](https://groups.io/g/powergenome) discussion

To contribute improvements:

1. Fork the repository
2. Make your changes
3. Add tests if applicable
4. Submit a pull request

## References

- PowerGenome Documentation: https://github.com/PowerGenome/PowerGenome/wiki
- PowerLASCOPF.jl: [Add actual URL when available]
- MATPOWER: https://matpower.org/
- PandaPower: https://www.pandapower.org/
