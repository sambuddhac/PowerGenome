# Additional Data Sources for PowerLASCOPF Export

This document provides sample code and guidance for incorporating additional data sources into your PowerLASCOPF exports.

## Overview

PowerGenome's default data may not include all electrical parameters needed for AC optimal power flow. This guide shows how to integrate data from external sources.

## 1. MATPOWER Case Files

MATPOWER is a widely-used power system simulation package with many test cases.

### Converting MATPOWER to PowerLASCOPF

```python
import scipy.io as sio
import pandas as pd
from pathlib import Path
from powergenome.PowerLASCOPF import PowerLASCOPFInputData, write_lascopf_data

def convert_matpower_case(mat_file, output_folder):
    """Convert MATPOWER .mat file to PowerLASCOPF format.
    
    Parameters
    ----------
    mat_file : str or Path
        Path to MATPOWER .mat case file
    output_folder : str or Path
        Output folder for CSV files
    """
    # Load MATPOWER case
    mat_case = sio.loadmat(mat_file)
    mpc = mat_case['mpc']
    
    # Extract and convert bus data
    bus_cols = ['bus_id', 'bus_type', 'pd', 'qd', 'gs', 'bs', 
                'area', 'vm', 'va', 'base_kv', 'zone', 'vmax', 'vmin']
    buses = pd.DataFrame(mpc['bus'][0, 0], columns=bus_cols)
    
    # Extract and convert branch data
    branch_cols = ['from_bus', 'to_bus', 'r', 'x', 'b', 
                   'rate_a', 'rate_b', 'rate_c', 'tap_ratio', 
                   'shift_angle', 'status', 'angmin', 'angmax']
    branches = pd.DataFrame(mpc['branch'][0, 0], columns=branch_cols)
    
    # Extract and convert generator data
    gen_cols = ['bus_id', 'pg', 'qg', 'qg_max', 'qg_min', 'vg', 
                'mbase', 'status', 'pg_max', 'pg_min']
    gens = pd.DataFrame(mpc['gen'][0, 0], columns=gen_cols)
    gens['gen_id'] = [f"gen_{i+1}" for i in range(len(gens))]
    
    # Add cost data if available
    if 'gencost' in mpc.dtype.names:
        gencost = mpc['gencost'][0, 0]
        # MATPOWER format: [model_type, startup, shutdown, n, c_n, ..., c_0]
        gens['startup_cost'] = gencost[:, 1]
        gens['shutdown_cost'] = gencost[:, 2]
        # Polynomial costs - extract coefficients
        n_cost_terms = gencost[:, 3].astype(int)
        gens['cost_coef_c2'] = 0.0
        gens['cost_coef_c1'] = 0.0
        gens['cost_coef_c0'] = 0.0
        for i, n in enumerate(n_cost_terms):
            if n >= 1:
                gens.loc[i, 'cost_coef_c0'] = gencost[i, 4]
            if n >= 2:
                gens.loc[i, 'cost_coef_c1'] = gencost[i, 5]
            if n >= 3:
                gens.loc[i, 'cost_coef_c2'] = gencost[i, 6]
    
    # Write to CSV files
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    buses.to_csv(output_path / 'buses.csv', index=False)
    branches.to_csv(output_path / 'branches.csv', index=False)
    gens.to_csv(output_path / 'generators.csv', index=False)
    
    print(f"Converted MATPOWER case to {output_folder}")
    print(f"  Buses: {len(buses)}")
    print(f"  Branches: {len(branches)}")
    print(f"  Generators: {len(gens)}")

# Example usage
convert_matpower_case('case118.mat', 'lascopf_output/case118')
```

## 2. PandaPower Networks

PandaPower is a Python-based power system analysis tool.

### Converting PandaPower to PowerLASCOPF

```python
import pandapower as pp
import pandas as pd
from pathlib import Path

def convert_pandapower_net(net, output_folder):
    """Convert PandaPower network to PowerLASCOPF format.
    
    Parameters
    ----------
    net : pandapower.Network
        PandaPower network object
    output_folder : str or Path
        Output folder for CSV files
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert buses
    buses = pd.DataFrame({
        'bus_id': net.bus.index,
        'bus_type': 1,  # Default to PQ
        'pd': 0.0,
        'qd': 0.0,
        'gs': 0.0,
        'bs': 0.0,
        'area': net.bus.zone if 'zone' in net.bus.columns else 1,
        'vm': net.bus.vn_kv / net.bus.vn_kv.iloc[0],  # Normalize to p.u.
        'va': 0.0,
        'base_kv': net.bus.vn_kv,
        'zone': net.bus.zone if 'zone' in net.bus.columns else 1,
        'vmax': net.bus.max_vm_pu,
        'vmin': net.bus.min_vm_pu,
    })
    
    # Set bus types based on generators and external grid
    if len(net.gen) > 0:
        gen_buses = net.gen.bus.unique()
        buses.loc[buses['bus_id'].isin(gen_buses), 'bus_type'] = 2  # PV
    if len(net.ext_grid) > 0:
        slack_buses = net.ext_grid.bus.unique()
        buses.loc[buses['bus_id'].isin(slack_buses), 'bus_type'] = 3  # Slack
    
    # Add loads to buses
    if len(net.load) > 0:
        load_by_bus = net.load.groupby('bus').agg({
            'p_mw': 'sum',
            'q_mvar': 'sum'
        })
        for bus_id in load_by_bus.index:
            buses.loc[buses['bus_id'] == bus_id, 'pd'] = load_by_bus.loc[bus_id, 'p_mw']
            buses.loc[buses['bus_id'] == bus_id, 'qd'] = load_by_bus.loc[bus_id, 'q_mvar']
    
    # Convert lines and transformers to branches
    branches_list = []
    
    # Lines
    if len(net.line) > 0:
        for idx, line in net.line.iterrows():
            branch = {
                'from_bus': line.from_bus,
                'to_bus': line.to_bus,
                'r': line.r_ohm_per_km * line.length_km / (net.bus.loc[line.from_bus, 'vn_kv'] ** 2 / 100),  # p.u.
                'x': line.x_ohm_per_km * line.length_km / (net.bus.loc[line.from_bus, 'vn_kv'] ** 2 / 100),  # p.u.
                'b': line.c_nf_per_km * line.length_km * (net.bus.loc[line.from_bus, 'vn_kv'] ** 2 / 100) * 1e-9 * 2 * 3.14159 * 50,  # p.u.
                'rate_a': line.max_i_ka * net.bus.loc[line.from_bus, 'vn_kv'] * 1.732,  # MVA
                'rate_b': line.max_i_ka * net.bus.loc[line.from_bus, 'vn_kv'] * 1.732 * 1.1,
                'rate_c': line.max_i_ka * net.bus.loc[line.from_bus, 'vn_kv'] * 1.732 * 1.2,
                'tap_ratio': 1.0,
                'shift_angle': 0.0,
                'status': 1 if line.in_service else 0,
                'angmin': -30.0,
                'angmax': 30.0,
            }
            branches_list.append(branch)
    
    # Transformers
    if len(net.trafo) > 0:
        for idx, trafo in net.trafo.iterrows():
            branch = {
                'from_bus': trafo.hv_bus,
                'to_bus': trafo.lv_bus,
                'r': trafo.vk_percent / 100.0 * 0.1,  # Approximate
                'x': trafo.vk_percent / 100.0,  # p.u.
                'b': 0.0,
                'rate_a': trafo.sn_mva,
                'rate_b': trafo.sn_mva * 1.1,
                'rate_c': trafo.sn_mva * 1.2,
                'tap_ratio': trafo.tap_pos / trafo.tap_neutral if hasattr(trafo, 'tap_pos') else 1.0,
                'shift_angle': trafo.shift_degree if 'shift_degree' in trafo else 0.0,
                'status': 1 if trafo.in_service else 0,
                'angmin': -30.0,
                'angmax': 30.0,
            }
            branches_list.append(branch)
    
    branches = pd.DataFrame(branches_list)
    
    # Convert generators
    gens_list = []
    if len(net.gen) > 0:
        for idx, gen in net.gen.iterrows():
            gen_data = {
                'bus_id': gen.bus,
                'gen_id': f"gen_{idx}",
                'pg_min': gen.min_p_mw,
                'pg_max': gen.max_p_mw,
                'qg_min': gen.min_q_mvar,
                'qg_max': gen.max_q_mvar,
                'cost_coef_c2': 0.0,
                'cost_coef_c1': 1.0,  # Default cost
                'cost_coef_c0': 0.0,
                'ramp_up': gen.max_p_mw,  # No ramp limits in PandaPower by default
                'ramp_down': gen.max_p_mw,
                'startup_cost': 0.0,
                'shutdown_cost': 0.0,
                'min_up_time': 0,
                'min_down_time': 0,
            }
            gens_list.append(gen_data)
    
    gens = pd.DataFrame(gens_list)
    
    # Write to CSV
    buses.to_csv(output_path / 'buses.csv', index=False)
    branches.to_csv(output_path / 'branches.csv', index=False)
    gens.to_csv(output_path / 'generators.csv', index=False)
    
    print(f"Converted PandaPower network to {output_folder}")
    print(f"  Buses: {len(buses)}")
    print(f"  Branches: {len(branches)}")
    print(f"  Generators: {len(gens)}")

# Example usage
net = pp.networks.case118()  # Load IEEE 118-bus system
convert_pandapower_net(net, 'lascopf_output/pp_case118')
```

## 3. WECC Planning Models

Western Electricity Coordinating Council (WECC) provides planning models.

### Loading WECC Data

```python
import pandas as pd
from pathlib import Path

def load_wecc_data(wecc_folder, output_folder):
    """Load and convert WECC planning model data.
    
    WECC data typically comes in multiple CSV or RAW files.
    This is a template - adjust based on your specific WECC dataset.
    
    Parameters
    ----------
    wecc_folder : str or Path
        Folder containing WECC data files
    output_folder : str or Path
        Output folder for PowerLASCOPF format
    """
    wecc_path = Path(wecc_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Example: Load bus data from WECC format
    # Adjust column names based on actual WECC format
    wecc_buses = pd.read_csv(wecc_path / 'buses.csv')
    buses = pd.DataFrame({
        'bus_id': wecc_buses['BUS_NUMBER'],
        'bus_type': wecc_buses['IDE'],  # 1=PQ, 2=PV, 3=Slack
        'pd': wecc_buses['PLOAD_MW'],
        'qd': wecc_buses['QLOAD_MVAR'],
        'gs': wecc_buses['SHUNT_G'],
        'bs': wecc_buses['SHUNT_B'],
        'area': wecc_buses['AREA'],
        'vm': wecc_buses['VM_PU'],
        'va': wecc_buses['VA_DEG'],
        'base_kv': wecc_buses['BASE_KV'],
        'zone': wecc_buses['ZONE'],
        'vmax': 1.05,
        'vmin': 0.95,
    })
    
    # Load branch data
    wecc_branches = pd.read_csv(wecc_path / 'branches.csv')
    branches = pd.DataFrame({
        'from_bus': wecc_branches['FROM_BUS'],
        'to_bus': wecc_branches['TO_BUS'],
        'r': wecc_branches['R_PU'],
        'x': wecc_branches['X_PU'],
        'b': wecc_branches['B_PU'],
        'rate_a': wecc_branches['RATE_A'],
        'rate_b': wecc_branches['RATE_B'],
        'rate_c': wecc_branches['RATE_C'],
        'tap_ratio': wecc_branches.get('TAP', 1.0),
        'shift_angle': wecc_branches.get('SHIFT', 0.0),
        'status': 1,
        'angmin': -30.0,
        'angmax': 30.0,
    })
    
    # Load generator data
    wecc_gens = pd.read_csv(wecc_path / 'generators.csv')
    gens = pd.DataFrame({
        'bus_id': wecc_gens['BUS'],
        'gen_id': wecc_gens['GEN_ID'],
        'pg_min': wecc_gens['PMIN_MW'],
        'pg_max': wecc_gens['PMAX_MW'],
        'qg_min': wecc_gens['QMIN_MVAR'],
        'qg_max': wecc_gens['QMAX_MVAR'],
        'cost_coef_c2': wecc_gens.get('COST_C2', 0.0),
        'cost_coef_c1': wecc_gens['COST_C1'],
        'cost_coef_c0': wecc_gens.get('COST_C0', 0.0),
        'ramp_up': wecc_gens.get('RAMP_UP', wecc_gens['PMAX_MW']),
        'ramp_down': wecc_gens.get('RAMP_DN', wecc_gens['PMAX_MW']),
        'startup_cost': wecc_gens.get('STARTUP_COST', 0.0),
        'shutdown_cost': 0.0,
        'min_up_time': wecc_gens.get('MIN_UP_TIME', 0),
        'min_down_time': wecc_gens.get('MIN_DN_TIME', 0),
    })
    
    # Write converted data
    buses.to_csv(output_path / 'buses.csv', index=False)
    branches.to_csv(output_path / 'branches.csv', index=False)
    gens.to_csv(output_path / 'generators.csv', index=False)
    
    print(f"Converted WECC data to {output_folder}")

# Example usage
# load_wecc_data('path/to/wecc/data', 'lascopf_output/wecc_case')
```

## 4. Integrating with PowerGenome Exports

You can merge external electrical parameters with PowerGenome data:

```python
from powergenome.PowerLASCOPF import process_lascopf_data, write_lascopf_data
import pandas as pd
from pathlib import Path

def merge_external_network_data(pg_data_dict, external_network_file, output_folder):
    """Merge PowerGenome data with external network parameters.
    
    Parameters
    ----------
    pg_data_dict : dict
        PowerGenome data dictionary
    external_network_file : str
        CSV file with detailed network parameters
    output_folder : str or Path
        Output folder
    """
    # Load external network data with detailed electrical parameters
    external_network = pd.read_csv(external_network_file)
    
    # Merge with PowerGenome transmission data
    # Match based on from/to zones
    if 'network' in pg_data_dict:
        pg_network = pg_data_dict['network'].copy()
        
        # Extract zone information
        if 'Network_Lines' in pg_network.columns:
            pg_network[['from_zone', 'to_zone']] = (
                pg_network['Network_Lines']
                .str.extract(r'(\w+)_to_(\w+)')
            )
        
        # Merge electrical parameters
        merged = pg_network.merge(
            external_network,
            left_on=['from_zone', 'to_zone'],
            right_on=['from_zone', 'to_zone'],
            how='left'
        )
        
        # Use external values where available, PowerGenome defaults otherwise
        for col in ['r', 'x', 'b']:
            if f'{col}_external' in merged.columns:
                merged[col] = merged[f'{col}_external'].fillna(merged.get(col, 0.01))
        
        pg_data_dict['network'] = merged
    
    # Process and write
    lascopf_data = process_lascopf_data(
        case_folder=Path(output_folder),
        data_dict=pg_data_dict
    )
    write_lascopf_data(lascopf_data)

# Example usage
# merge_external_network_data(
#     pg_data_dict={'gen_data': gens, 'network': transmission},
#     external_network_file='detailed_network_params.csv',
#     output_folder='lascopf_output'
# )
```

## 5. Time-Series Load Data from EIA

Integrate EIA hourly load data:

```python
import pandas as pd
from pathlib import Path

def load_eia_hourly_demand(eia_file, zone_mapping, output_folder):
    """Load hourly demand from EIA and format for PowerLASCOPF.
    
    Parameters
    ----------
    eia_file : str
        Path to EIA hourly demand file
    zone_mapping : dict
        Mapping from EIA BA codes to PowerGenome zones
    output_folder : str or Path
        Output folder
    """
    # Load EIA data
    # Typical EIA format: columns are BA codes, index is datetime
    eia_demand = pd.read_csv(eia_file, index_col=0, parse_dates=True)
    
    # Map to PowerGenome zones
    demand_by_zone = pd.DataFrame()
    for eia_ba, pg_zone in zone_mapping.items():
        if eia_ba in eia_demand.columns:
            col_name = f"bus_{pg_zone}_p"
            demand_by_zone[col_name] = eia_demand[eia_ba]
    
    # Add period index
    demand_by_zone.insert(0, 'period', range(1, len(demand_by_zone) + 1))
    
    # Calculate reactive power
    p_cols = [c for c in demand_by_zone.columns if c.endswith('_p')]
    for p_col in p_cols:
        zone = p_col.split('_')[1]
        q_col = f"bus_{zone}_q"
        demand_by_zone[q_col] = demand_by_zone[p_col] * 0.33  # Power factor 0.95
    
    # Write output
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    demand_by_zone.to_csv(output_path / 'load_timeseries.csv', index=False)
    
    print(f"Processed EIA demand data: {len(demand_by_zone)} hours")

# Example usage
# zone_mapping = {
#     'CISO': '1',  # California to Zone 1
#     'PNM': '2',   # New Mexico to Zone 2
#     # etc.
# }
# load_eia_hourly_demand('eia_demand.csv', zone_mapping, 'lascopf_output')
```

## 6. Renewable Generation Profiles from NREL

```python
import pandas as pd
import h5py
from pathlib import Path

def load_nrel_renewable_profiles(wind_file, solar_file, resource_mapping, output_folder):
    """Load NREL renewable profiles and format for PowerLASCOPF.
    
    Parameters
    ----------
    wind_file : str
        Path to NREL wind profile HDF5 file
    solar_file : str
        Path to NREL solar profile HDF5 file
    resource_mapping : dict
        Mapping from NREL site IDs to PowerGenome generator IDs
    output_folder : str or Path
        Output folder
    """
    profiles = pd.DataFrame()
    
    # Load wind profiles
    with h5py.File(wind_file, 'r') as f:
        for nrel_site, gen_id in resource_mapping['wind'].items():
            if nrel_site in f:
                cf = f[nrel_site]['cf_profile'][:]
                profiles[f'gen_{gen_id}_pmax'] = cf
    
    # Load solar profiles
    with h5py.File(solar_file, 'r') as f:
        for nrel_site, gen_id in resource_mapping['solar'].items():
            if nrel_site in f:
                cf = f[nrel_site]['cf_profile'][:]
                profiles[f'gen_{gen_id}_pmax'] = cf
    
    # Add period index
    profiles.insert(0, 'period', range(1, len(profiles) + 1))
    
    # Write output
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output_path / 'generation_profiles.csv', index=False)
    
    print(f"Processed renewable profiles: {len(profiles)} hours, {len(profiles.columns)-1} resources")

# Example usage
# resource_mapping = {
#     'wind': {'site_123': 1, 'site_456': 2},
#     'solar': {'site_789': 3}
# }
# load_nrel_renewable_profiles(
#     'nrel_wind.h5',
#     'nrel_solar.h5',
#     resource_mapping,
#     'lascopf_output'
# )
```

## Summary

These examples show how to:

1. Convert standard power system formats (MATPOWER, PandaPower)
2. Integrate regional planning model data (WECC)
3. Add detailed electrical parameters to PowerGenome exports
4. Load time-series data from external sources (EIA, NREL)

Choose the approach that matches your data sources and study requirements. The PowerLASCOPF export module is designed to be flexible and accommodate diverse data inputs.

## Additional Resources

- **MATPOWER**: https://matpower.org/
- **PandaPower**: https://www.pandapower.org/
- **NREL Wind Toolkit**: https://www.nrel.gov/grid/wind-toolkit.html
- **NREL Solar Toolkit**: https://nsrdb.nrel.gov/
- **EIA Data**: https://www.eia.gov/opendata/
- **WECC**: https://www.wecc.org/
