# PowerLASCOPF Export

This directory contains scripts and documentation for exporting PowerGenome data in PowerLASCOPF.jl-compatible format.

## Files

- **PowerLASCOPF.py**: Core module with conversion functions for each data type
- **run_lascopf_export.py**: Command-line interface for PowerLASCOPF export
- **docs/PowerLASCOPF_quickstart.md**: Quick start guide (5-minute overview)
- **docs/PowerLASCOPF_export_guide.md**: Comprehensive user guide with detailed documentation
- **docs/PowerLASCOPF_additional_data_sources.md**: Sample code for integrating additional data sources

## Quick Start

Export your PowerGenome data to PowerLASCOPF format:

```bash
python -m powergenome.run_lascopf_export -sf my_settings -rf lascopf_output
```

See [docs/PowerLASCOPF_quickstart.md](docs/PowerLASCOPF_quickstart.md) for more.

## What Gets Exported

1. **generators.csv** - Generator characteristics and costs
2. **buses.csv** - Electrical bus specifications
3. **branches.csv** - Transmission network with impedances
4. **load_timeseries.csv** - Hourly load profiles (P and Q)
5. **generation_profiles.csv** - Time-varying generation availability

## Key Features

- **Modular Design**: Easy to extend for different PowerLASCOPF.jl branches
- **Flexible Data Mapping**: Handles missing electrical parameters with sensible defaults
- **Multiple Data Sources**: Sample code for MATPOWER, PandaPower, WECC, EIA, NREL
- **Comprehensive Documentation**: User guides and known limitations clearly documented

## Documentation

- **Quick Start**: [docs/PowerLASCOPF_quickstart.md](docs/PowerLASCOPF_quickstart.md)
- **Full Guide**: [docs/PowerLASCOPF_export_guide.md](docs/PowerLASCOPF_export_guide.md)
- **Additional Data Sources**: [docs/PowerLASCOPF_additional_data_sources.md](docs/PowerLASCOPF_additional_data_sources.md)

## Known Limitations

PowerGenome focuses on capacity expansion planning and doesn't track all parameters needed for AC power flow:

- Line impedances estimated from distances
- Reactive power limits based on typical generator capabilities  
- Default voltage levels and constraints
- Simplified network topology

See the [full guide](docs/PowerLASCOPF_export_guide.md#known-limitations-and-data-gaps) for details and how to address these gaps.

## Compatibility

Designed to work with both stable and development branches of PowerLASCOPF.jl. The modular architecture allows easy customization for format variations.

## Support

- PowerGenome Wiki: https://github.com/PowerGenome/PowerGenome/wiki
- Issue Tracker: https://github.com/PowerGenome/PowerGenome/issues
- Discussion Forum: https://groups.io/g/powergenome

## Contributing

Contributions welcome! See the main PowerGenome README for contribution guidelines.
