import scipy.io as sio
import pandas as pd
from pathlib import Path
import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MATPOWER .mat file to PowerLASCOPF input CSVs.")
    parser.add_argument("mat_file", help="Path to MATPOWER .mat case file")
    parser.add_argument("output_folder", help="Output directory for PowerLASCOPF CSVs")
    args = parser.parse_args()
    convert_matpower_case(args.mat_file, args.output_folder)