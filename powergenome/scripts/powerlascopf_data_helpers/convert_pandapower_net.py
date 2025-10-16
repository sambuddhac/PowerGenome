import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def load_pandapower_network(input_file):
    """Load a PandaPower network from pickle or JSON file.

    Parameters
    ----------
    input_file : str or Path
        Path to PandaPower network file (.p or .json)

    Returns
    -------
    pandapower.Network
        Loaded PandaPower network

    Raises
    ------
    ValueError
        If file format is not supported or file cannot be loaded
    """
    input_path = Path(input_file)
    file_ext = input_path.suffix.lower()

    try:
        import pandapower as pp
    except ImportError:
        raise ImportError(
            "pandapower is required to use this script. "
            "Install it with: pip install pandapower"
        )

    try:
        if file_ext == ".p":
            print(f"Loading PandaPower network from pickle file: {input_file}")
            # Try pandapower's from_pickle first
            try:
                net = pp.from_pickle(str(input_path))
            except (pickle.UnpicklingError, AttributeError, OSError) as e:
                # Fall back to direct pickle load
                with open(input_path, "rb") as f:
                    net = pickle.load(f)
        elif file_ext == ".json":
            print(f"Loading PandaPower network from JSON file: {input_file}")
            net = pp.from_json(str(input_path))
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                "Supported formats are .p (pickle) and .json"
            )

        # Validate that we have a valid pandapower network
        # PandaPower networks can be dict-like or have attributes
        if isinstance(net, dict):
            if "bus" not in net:
                raise ValueError(
                    "Loaded object is not a valid PandaPower network. "
                    "Missing 'bus' table."
                )
        elif not hasattr(net, "bus"):
            raise ValueError(
                "Loaded object is not a valid PandaPower network. "
                "Expected pandapower network with 'bus' attribute."
            )

        print("PandaPower network loaded successfully")
        return net
    except Exception as e:
        raise ValueError(f"Error loading PandaPower network: {str(e)}")


def convert_buses(net):
    """Convert PandaPower bus data to PowerLASCOPF format.

    Parameters
    ----------
    net : pandapower.Network
        PandaPower network

    Returns
    -------
    pd.DataFrame
        Bus data in PowerLASCOPF format
    """
    buses_df = net.bus.copy()

    # Map PandaPower bus attributes to PowerLASCOPF format
    # PandaPower uses: name, vn_kv, type, zone, in_service
    # PowerLASCOPF uses: bus_id, bus_type, pd, qd, gs, bs, area, vm, va, base_kv, zone, vmax, vmin

    bus_data = pd.DataFrame()
    bus_data["bus_id"] = buses_df.index + 1  # 1-indexed

    # Map bus type: PandaPower uses 'b' (busbar), 'n' (node), 'm' (muff)
    # PowerLASCOPF uses: 1 (PQ), 2 (PV), 3 (ref/slack)
    # Default to PQ bus (type 1) unless identified otherwise
    bus_data["bus_type"] = 1

    # Initialize demand (pd, qd) - will be filled from load data
    bus_data["pd"] = 0.0
    bus_data["qd"] = 0.0

    # Shunt susceptance and conductance
    if "shunt" in net:
        for idx, shunt in net.shunt.iterrows():
            if shunt["in_service"]:
                bus_idx = shunt["bus"]
                bus_data.loc[bus_data["bus_id"] == bus_idx + 1, "gs"] = shunt.get(
                    "p_mw", 0.0
                )
                bus_data.loc[bus_data["bus_id"] == bus_idx + 1, "bs"] = shunt.get(
                    "q_mvar", 0.0
                )

    if "gs" not in bus_data.columns:
        bus_data["gs"] = 0.0
    if "bs" not in bus_data.columns:
        bus_data["bs"] = 0.0

    # Area and zone - use zone if available
    bus_data["area"] = buses_df.get("zone", 1)
    bus_data["zone"] = buses_df.get("zone", 1)

    # Voltage magnitude and angle - use nominal values
    bus_data["vm"] = 1.0  # per unit
    bus_data["va"] = 0.0  # degrees

    # Base voltage
    bus_data["base_kv"] = buses_df["vn_kv"]

    # Voltage limits
    bus_data["vmax"] = buses_df.get("max_vm_pu", 1.1)
    bus_data["vmin"] = buses_df.get("min_vm_pu", 0.9)

    # Add load data to buses
    if "load" in net:
        for idx, load in net.load.iterrows():
            if load["in_service"]:
                bus_idx = load["bus"]
                # Aggregate loads at the same bus
                current_pd = bus_data.loc[
                    bus_data["bus_id"] == bus_idx + 1, "pd"
                ].values
                current_qd = bus_data.loc[
                    bus_data["bus_id"] == bus_idx + 1, "qd"
                ].values
                if len(current_pd) > 0:
                    bus_data.loc[bus_data["bus_id"] == bus_idx + 1, "pd"] = (
                        current_pd[0] + load["p_mw"]
                    )
                    bus_data.loc[bus_data["bus_id"] == bus_idx + 1, "qd"] = (
                        current_qd[0] + load["q_mvar"]
                    )

    # Identify slack bus from ext_grid
    if "ext_grid" in net and len(net.ext_grid) > 0:
        for idx, ext_grid in net.ext_grid.iterrows():
            if ext_grid["in_service"]:
                slack_bus_idx = ext_grid["bus"]
                bus_data.loc[bus_data["bus_id"] == slack_bus_idx + 1, "bus_type"] = 3

    print(f"Converted {len(bus_data)} buses")
    return bus_data


def convert_branches(net):
    """Convert PandaPower line and transformer data to PowerLASCOPF format.

    Parameters
    ----------
    net : pandapower.Network
        PandaPower network

    Returns
    -------
    pd.DataFrame
        Branch data in PowerLASCOPF format (lines and transformers combined)
    """
    branches_list = []

    # Process lines
    if "line" in net and len(net.line) > 0:
        for idx, line in net.line.iterrows():
            if line["in_service"]:
                # Get from_bus voltage for base impedance calculation
                from_bus_vn = net.bus.loc[line["from_bus"], "vn_kv"]
                base_mva = 100.0  # Standard base MVA

                # Calculate base impedance: Z_base = V^2 / S_base
                z_base = from_bus_vn**2 / base_mva

                # Convert ohms to per unit
                r_pu = (
                    line["r_ohm_per_km"] * line["length_km"] / z_base
                    if z_base > 0
                    else 0.0
                )
                x_pu = (
                    line["x_ohm_per_km"] * line["length_km"] / z_base
                    if z_base > 0
                    else 0.0
                )

                # Susceptance: B = 2*pi*f*C, convert to per unit
                # c_nf_per_km is in nanofarads per km
                # For 60 Hz: B = 2*pi*60*C_nf*1e-9 * length_km
                # Then convert to per unit: B_pu = B * Z_base
                freq = 60.0  # Hz
                b_pu = (
                    2
                    * 3.14159
                    * freq
                    * line["c_nf_per_km"]
                    * 1e-9
                    * line["length_km"]
                    * z_base
                )

                # Calculate MVA rating from current limit
                # S = sqrt(3) * V * I
                rate_mva = 1.732 * from_bus_vn * line["max_i_ka"]

                branch = {
                    "from_bus": line["from_bus"] + 1,  # 1-indexed
                    "to_bus": line["to_bus"] + 1,  # 1-indexed
                    "r": r_pu,
                    "x": x_pu,
                    "b": b_pu,
                    "rate_a": rate_mva,
                    "rate_b": rate_mva * (line.get("max_loading_percent", 100) / 100),
                    "rate_c": rate_mva * (line.get("max_loading_percent", 100) / 100),
                    "tap_ratio": 0.0,  # No tap for lines
                    "shift_angle": 0.0,  # No phase shift for lines
                    "status": 1,  # In service
                    "angmin": -360.0,
                    "angmax": 360.0,
                }
                branches_list.append(branch)

    # Process transformers
    if "trafo" in net and len(net.trafo) > 0:
        for idx, trafo in net.trafo.iterrows():
            if trafo["in_service"]:
                # Calculate impedance in per unit
                # vk_percent is the short circuit voltage in percent
                # vkr_percent is the real part
                r_pu = (
                    (trafo["vkr_percent"] / 100)
                    if pd.notna(trafo["vkr_percent"])
                    else 0
                )
                x_pu_sq = (trafo["vk_percent"] / 100) ** 2 - r_pu**2
                x_pu = x_pu_sq**0.5 if x_pu_sq > 0 else 0

                branch = {
                    "from_bus": trafo["hv_bus"] + 1,  # 1-indexed
                    "to_bus": trafo["lv_bus"] + 1,  # 1-indexed
                    "r": r_pu,
                    "x": x_pu,
                    "b": 0.0,  # Transformers typically have negligible shunt susceptance
                    "rate_a": trafo["sn_mva"],  # MVA rating
                    "rate_b": trafo["sn_mva"],
                    "rate_c": trafo["sn_mva"],
                    "tap_ratio": trafo["vn_hv_kv"]
                    / trafo["vn_lv_kv"],  # Nominal tap ratio
                    "shift_angle": trafo.get("shift_degree", 0.0),
                    "status": 1,  # In service
                    "angmin": -360.0,
                    "angmax": 360.0,
                }
                branches_list.append(branch)

    branches_df = pd.DataFrame(branches_list)
    line_count = len(net.line) if "line" in net else 0
    trafo_count = len(net.trafo) if "trafo" in net else 0
    in_service_line_count = (
        net.line["in_service"].sum() if "line" in net and len(net.line) > 0 else 0
    )
    in_service_trafo_count = (
        net.trafo["in_service"].sum() if "trafo" in net and len(net.trafo) > 0 else 0
    )
    print(
        f"Converted {len(branches_df)} branches "
        f"({in_service_line_count} of {line_count} lines, "
        f"{in_service_trafo_count} of {trafo_count} transformers in service)"
    )
    return branches_df


def convert_generators(net):
    """Convert PandaPower generator data to PowerLASCOPF format.

    Parameters
    ----------
    net : pandapower.Network
        PandaPower network

    Returns
    -------
    pd.DataFrame
        Generator data in PowerLASCOPF format
    """
    gens_list = []

    # Process generators (gen table)
    if "gen" in net and len(net.gen) > 0:
        for idx, gen in net.gen.iterrows():
            if gen["in_service"]:
                generator = {
                    "gen_id": f"gen_{idx + 1}",
                    "bus_id": gen["bus"] + 1,  # 1-indexed
                    "pg": gen["p_mw"],  # Active power output
                    "qg": gen.get("q_mvar", 0.0),  # Reactive power output
                    "qg_max": gen.get("max_q_mvar", gen.get("q_mvar", 0.0)),
                    "qg_min": gen.get("min_q_mvar", -gen.get("q_mvar", 0.0)),
                    "vg": gen.get("vm_pu", 1.0),  # Voltage setpoint
                    "mbase": gen.get("sn_mva", 100.0),  # Base MVA
                    "status": 1,  # In service
                    "pg_max": gen.get("max_p_mw", gen["p_mw"]),
                    "pg_min": gen.get("min_p_mw", 0.0),
                }
                gens_list.append(generator)

    # Process static generators (sgen table) - typically renewable sources
    if "sgen" in net and len(net.sgen) > 0:
        for idx, sgen in net.sgen.iterrows():
            if sgen["in_service"]:
                generator = {
                    "gen_id": f"sgen_{idx + 1}",
                    "bus_id": sgen["bus"] + 1,  # 1-indexed
                    "pg": sgen["p_mw"],  # Active power output
                    "qg": sgen.get("q_mvar", 0.0),  # Reactive power output
                    "qg_max": sgen.get("max_q_mvar", sgen.get("q_mvar", 0.0)),
                    "qg_min": sgen.get("min_q_mvar", -sgen.get("q_mvar", 0.0)),
                    "vg": 1.0,  # Static generators don't control voltage
                    "mbase": sgen.get("sn_mva", 100.0),  # Base MVA
                    "status": 1,  # In service
                    "pg_max": sgen.get("max_p_mw", sgen["p_mw"]),
                    "pg_min": sgen.get("min_p_mw", 0.0),
                }
                gens_list.append(generator)

    # Process external grid connections as generators
    if "ext_grid" in net and len(net.ext_grid) > 0:
        for idx, ext_grid in net.ext_grid.iterrows():
            if ext_grid["in_service"]:
                generator = {
                    "gen_id": f"ext_grid_{idx + 1}",
                    "bus_id": ext_grid["bus"] + 1,  # 1-indexed
                    "pg": 0.0,  # Will be determined by power flow
                    "qg": 0.0,  # Will be determined by power flow
                    "qg_max": ext_grid.get("max_q_mvar", 9999.0),
                    "qg_min": ext_grid.get("min_q_mvar", -9999.0),
                    "vg": ext_grid.get("vm_pu", 1.0),  # Voltage setpoint
                    "mbase": 100.0,  # Base MVA
                    "status": 1,  # In service
                    "pg_max": ext_grid.get("max_p_mw", 9999.0),
                    "pg_min": ext_grid.get("min_p_mw", -9999.0),
                }
                gens_list.append(generator)

    gens_df = pd.DataFrame(gens_list)

    # Add cost data placeholders (PandaPower doesn't have built-in cost data)
    gens_df["startup_cost"] = 0.0
    gens_df["shutdown_cost"] = 0.0
    gens_df["cost_coef_c2"] = 0.0
    gens_df["cost_coef_c1"] = 0.0
    gens_df["cost_coef_c0"] = 0.0

    gen_count = len(net.gen) if "gen" in net else 0
    sgen_count = len(net.sgen) if "sgen" in net else 0
    ext_grid_count = len(net.ext_grid) if "ext_grid" in net else 0
    print(
        f"Converted {len(gens_df)} generators "
        f"({gen_count} gen, {sgen_count} sgen, {ext_grid_count} ext_grid)"
    )
    return gens_df


def convert_pandapower_net(input_file, output_folder):
    """Convert PandaPower network to PowerLASCOPF CSV format.

    Parameters
    ----------
    input_file : str or Path
        Path to PandaPower network file (.p or .json)
    output_folder : str or Path
        Output folder for CSV files

    Raises
    ------
    ValueError
        If input file format is not supported or conversion fails
    """
    # Load PandaPower network
    net = load_pandapower_network(input_file)

    # Convert network components
    print("\nConverting network components...")
    buses = convert_buses(net)
    branches = convert_branches(net)
    generators = convert_generators(net)

    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write CSV files
    print(f"\nWriting output files to {output_folder}...")
    buses.to_csv(output_path / "buses.csv", index=False)
    print(f"  Wrote buses.csv ({len(buses)} buses)")

    branches.to_csv(output_path / "branches.csv", index=False)
    print(f"  Wrote branches.csv ({len(branches)} branches)")

    generators.to_csv(output_path / "generators.csv", index=False)
    print(f"  Wrote generators.csv ({len(generators)} generators)")

    print(f"\nConversion complete! Files saved to {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PandaPower network to PowerLASCOPF input CSVs."
    )
    parser.add_argument(
        "input_file",
        help="Path to PandaPower network file (.p for pickle or .json for JSON)",
    )
    parser.add_argument(
        "output_folder", help="Output directory for PowerLASCOPF CSV files"
    )
    args = parser.parse_args()

    try:
        convert_pandapower_net(args.input_file, args.output_folder)
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)
