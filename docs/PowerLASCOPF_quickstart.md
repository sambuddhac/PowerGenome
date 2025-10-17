# PowerLASCOPF Export Quick Start

This is a quick reference for exporting PowerGenome data to PowerLASCOPF.jl format.

## 5-Minute Start

1. **Prepare your PowerGenome settings** (existing settings file works)

2. **Run the export**:
```bash
python -m powergenome.run_lascopf_export -sf my_settings -rf lascopf_output
```

3. **Find your data**:
```
lascopf_output/
  scenario_name/
    lascopf_data/
      generators.csv
      buses.csv
      branches.csv
      load_timeseries.csv
      generation_profiles.csv
```

## What You Get

- **generators.csv**: Generator capacities, costs, and operating limits
- **buses.csv**: Electrical bus specifications with voltage limits
- **branches.csv**: Transmission lines with impedances and flow limits
- **load_timeseries.csv**: Hourly load by bus (MW and MVAr)
- **generation_profiles.csv**: Available generation capacity over time

## Key Differences from GenX Export

| Feature | GenX | PowerLASCOPF |
|---------|------|--------------|
| Power Flow | DC (simplified) | AC (detailed) |
| Voltage | Not modeled | Per-unit voltage at each bus |
| Reactive Power | No | Yes (MVAr) |
| Line Impedance | Simplified | Resistance + Reactance |
| Use Case | Capacity expansion | Operations & dispatch |

## Command Options

```bash
# Basic export
python -m powergenome.run_lascopf_export -sf settings -rf output

# Skip existing generators
python -m powergenome.run_lascopf_export -sf settings -rf output --no-current-gens

# Export only load data (no generators)
python -m powergenome.run_lascopf_export -sf settings -rf output --no-gens

# Skip transmission network
python -m powergenome.run_lascopf_export -sf settings -rf output --no-transmission
```

## Important Assumptions

⚠️ **Default electrical parameters** are used where PowerGenome data doesn't include them:

- **Reactive power limits**: ±30% of active power capacity
- **Line impedances**: Estimated from distances (0.0001 Ω/mile reactance)
- **Base voltage**: 345 kV transmission
- **Power factor**: 0.95 for loads
- **Voltage limits**: 0.95 to 1.05 per unit

For production studies, you should **replace these with actual electrical parameters** from:
- Utility planning models
- WECC/ERCOT datasets
- FERC Form 715 data

## Quick Validation

Check your exports are reasonable:

```python
import pandas as pd

gens = pd.read_csv('lascopf_output/scenario/lascopf_data/generators.csv')
loads = pd.read_csv('lascopf_output/scenario/lascopf_data/load_timeseries.csv')

print(f"Total generation capacity: {gens['pg_max'].sum():.0f} MW")
print(f"Peak load: {loads.iloc[:, 1::2].sum(axis=1).max():.0f} MW")
print(f"Number of generators: {len(gens)}")
print(f"Number of time periods: {len(loads)}")
```

## Next Steps

- Read the [full export guide](PowerLASCOPF_export_guide.md) for detailed documentation
- Customize electrical parameters in your settings file
- Validate output in PowerLASCOPF.jl
- Report issues or request features on GitHub

## Common Issues

**"Missing electrical parameters"**: This is expected. Default values are used. Override with custom settings for your study region.

**"Voltage convergence failures"**: Check that load and generation are balanced in each zone. May need to adjust voltage limits.

**"Empty output files"**: Verify your PowerGenome settings include the required data sections (generators, load, transmission).

## Settings File Example

Your existing PowerGenome settings should work. No special configuration needed for basic export.

For custom electrical parameters, add:

```yaml
# Optional: Custom PowerLASCOPF parameters
lascopf_custom_params:
  base_voltage_kv: 345
  power_factor: 0.95
  voltage_limits:
    vmax: 1.05
    vmin: 0.95
```

See the [full guide](PowerLASCOPF_export_guide.md) for all options.
