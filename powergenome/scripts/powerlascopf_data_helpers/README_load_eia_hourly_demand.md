# load_eia_hourly_demand.py

A script to convert EIA hourly demand data from CSV and/or JSON formats to PowerLASCOPF-compatible CSV format.

## Overview

This script reads EIA (U.S. Energy Information Administration) hourly demand data in multiple formats and converts it to a standardized PowerLASCOPF format suitable for power system optimization modeling.

## Input Formats

### CSV Format
```csv
"Region Code","Timestamp (Hour Ending)","Demand (MWh)","Percent Change from Prior Hour"
US48,"10/11/2025 1 p.m. EDT",426315.93,0%
```

Required columns:
- `Region Code`: Region identifier (e.g., "US48", "CA")
- `Timestamp (Hour Ending)`: Timestamp in format "MM/DD/YYYY H a.m./p.m. TZ"
- `Demand (MWh)`: Demand value in MWh
- `Percent Change from Prior Hour`: Optional, not used

### JSON Format

**JSON Lines** (one object per line):
```json
{"respondent":"US48","dataType":"D","Timestamp (Hour Ending)":"10/11/2025 1 p.m. EDT","value":426315.93,"percentChange":"0%"}
```

**JSON Array**:
```json
[
  {"respondent":"US48","dataType":"D","Timestamp (Hour Ending)":"10/11/2025 1 p.m. EDT","value":426315.93,"percentChange":"0%"},
  {"respondent":"US48","dataType":"D","Timestamp (Hour Ending)":"10/11/2025 2 p.m. EDT","value":430123.45,"percentChange":"0.89%"}
]
```

Required fields:
- `respondent` or similar: Region identifier
- `Timestamp (Hour Ending)`: Timestamp string
- `value`: Demand value

## Output Format

PowerLASCOPF-compatible CSV with:
- `Time_Index`: Sequential hour index (1, 2, 3, ...)
- `Load_MW_{region}`: Demand in MW for each region (e.g., `Load_MW_US48`, `Load_MW_CA`)

Example output:
```csv
Time_Index,Load_MW_CA,Load_MW_US48
1,45123.67,426315.93
2,43876.23,415234.12
3,42567.89,408921.45
```

## Usage

### Basic Usage

Single CSV file:
```bash
python load_eia_hourly_demand.py input.csv output.csv
```

Multiple files (CSV and JSON):
```bash
python load_eia_hourly_demand.py input1.csv input2.json output.csv
```

### Using Flags

Explicit input/output flags:
```bash
python load_eia_hourly_demand.py -i file1.csv -i file2.json -o output.csv
```

Multiple input files:
```bash
python load_eia_hourly_demand.py \
  -i region1_data.csv \
  -i region2_data.csv \
  -i region3_data.json \
  -o combined_demand.csv
```

### Verbose Logging

```bash
python load_eia_hourly_demand.py -v input.csv output.csv
```

## Features

- **Multi-format Support**: Reads both CSV and JSON formats
- **Multi-region Support**: Aggregates data from multiple regions into a single output
- **Multi-file Support**: Combines data from multiple input files
- **Timestamp Parsing**: Handles various EIA timestamp formats with timezone abbreviations
- **Data Validation**: Checks for required columns/fields and reports errors
- **Duplicate Handling**: Averages duplicate entries for same region/timestamp
- **Logging**: Comprehensive logging of all operations

## Error Handling

The script provides detailed error messages for common issues:
- Missing input files
- Invalid CSV/JSON format
- Missing required columns/fields
- Timestamp parsing errors
- Empty datasets

## Requirements

- Python 3.6+
- pandas

## Module Functions

The script can also be imported and used programmatically:

```python
from load_eia_hourly_demand import (
    read_csv_file,
    read_json_file,
    aggregate_demand_data,
    create_powerlascopf_output,
    load_eia_hourly_demand
)

# Read individual files
csv_data = read_csv_file('input.csv')
json_data = read_json_file('input.json')

# Aggregate data
combined = aggregate_demand_data([csv_data, json_data])

# Convert to PowerLASCOPF format
output = create_powerlascopf_output(combined)

# Or use the complete workflow
load_eia_hourly_demand(['input1.csv', 'input2.json'], 'output.csv')
```

## Examples

### Example 1: Single Region from CSV
```bash
python load_eia_hourly_demand.py us48_2025.csv demand_us48.csv
```

### Example 2: Multiple Regions from Multiple Files
```bash
python load_eia_hourly_demand.py \
  -i california.csv \
  -i texas.json \
  -i newyork.csv \
  -o western_demand.csv
```

### Example 3: Combining Historical Data
```bash
python load_eia_hourly_demand.py \
  historical_2023.csv \
  historical_2024.csv \
  forecast_2025.json \
  all_years_demand.csv
```

## Notes

- All timestamps are converted to a consistent format for sorting
- Timezone information is preserved during parsing but not in the final output
- Output rows are sorted by timestamp to ensure correct Time_Index ordering
- The script handles missing data points by omitting them from the output
- Load values are in MWh in both input and output (column name uses MW for compatibility)

## Support

For issues or questions, please refer to the PowerGenome repository documentation.
