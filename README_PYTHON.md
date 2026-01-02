# Financial Statement Analysis - Python Implementation

## Project Overview

Python implementation of the Financial Statement Analysis examining **The Effect of Mandatory CSR Disclosure (Directive 2014/95/EU) on Corporate Profitability**.

## Project Structure

### Folders

- `input/` - Contains input data files
- `output/` - Contains all generated results (created automatically)

### Input Files

- `combined_data.csv` - Primary dataset (only file used by the script)
- `combined_data.dta` - Original Stata file (reference only)
- `other_variables.xlsx` - Excel data (already merged into CSV)

### Output Files (generated automatically)

- `fsa_analysis_plots.png` - Visualizations
- `regression_results.txt` - Detailed regression output
- `descriptive_statistics.csv` - Summary statistics
- `correlation_matrix.csv` - Correlation analysis
- `cleaned_data.csv` - Processed dataset

## Installation & Setup

### Step 1: Install Python

Make sure you have Python 3.8 or higher installed:

```bash
python --version
```

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scipy, matplotlib, seaborn, statsmodels, linearmodels

### Step 3: Run the Analysis

```bash
python fsa_analysis.py
```

## What the Script Does

1. **Data Loading & Preprocessing**

   - Loads `input/combined_data.csv`
   - Removes invalid observations
   - Filters to 2012-2021 period
   - Final dataset: 31,168 observations
2. **Variable Construction**

   - Creates ROA, leverage, log_assets, market_to_book, csr_indicator
3. **Analysis**

   - Descriptive statistics and correlations
   - OLS regression (baseline)
   - Fixed effects regression (main model)
   - Winsorization and robustness checks
4. **Outputs**

   - All results exported to `output/` folder
   - Includes visualizations, regression results, and cleaned data

## Troubleshooting

**Module not found error:**

```bash
pip install package_name
```

**CSV file not found:**

- Ensure `combined_data.csv` is in the `input/` folder
