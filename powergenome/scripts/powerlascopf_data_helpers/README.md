# PowerLASCOPF Data Helpers

This directory contains utility scripts for preparing data inputs for PowerLASCOPF models.

## Scripts

### load_nrel_renewable_profiles.py

Load and aggregate NREL renewable profile data (Wind Toolkit, NSRDB, ATB, etc.) and convert to PowerLASCOPF-compatible format.

#### Features

- **Flexible column detection**: Automatically detects site/location, timestamp, and generation/output columns
- **Multiple format support**: Handles both wide format (time × sites) and long format (explicit site, time, output columns)
- **Multi-file aggregation**: Combines data from multiple CSV files into a single output
- **Bus ID mapping**: Optional mapping from site IDs to bus IDs
- **Robust error handling**: Clear error messages and logging

#### Usage

Basic usage with auto-detection:
```bash
python load_nrel_renewable_profiles.py wind_profiles.csv -o output.csv
```

Multiple files with explicit column names:
```bash
python load_nrel_renewable_profiles.py wind.csv solar.csv \
  --site-col site_id --time-col timestamp --output-col capacity_factor \
  -o combined_profiles.csv
```

With bus ID mapping:
```bash
python load_nrel_renewable_profiles.py profiles.csv \
  --bus-map site_to_bus_mapping.csv -o output.csv
```

Enable verbose logging:
```bash
python load_nrel_renewable_profiles.py profiles.csv -o output.csv -v
```

#### Input Formats

**Wide format** (multiple sites as columns):
```csv
time_index,site_1,site_2,site_3
1,0.15,0.22,0.18
2,0.25,0.30,0.28
...
```

**Long format** (explicit site, time, output columns):
```csv
site_id,timestamp,capacity_factor
site_1,1,0.15
site_1,2,0.25
site_2,1,0.22
site_2,2,0.30
...
```

#### Output Format

PowerLASCOPF-compatible CSV with columns:
- `bus_id`: Bus/site identifier
- `time_index`: Time period identifier
- `renewable_mw`: Generation/capacity factor value

```csv
bus_id,time_index,renewable_mw
bus_101,1,0.15
bus_102,1,0.22
bus_101,2,0.25
bus_102,2,0.30
...
```

#### Bus Mapping File

Optional CSV file to map site IDs to bus IDs:
```csv
site_id,bus_id
site_1,bus_101
site_2,bus_102
site_3,bus_103
...
```

### convert_matpower_case.py

Convert MATPOWER .mat files to PowerLASCOPF CSV format.

See script documentation for usage details.
