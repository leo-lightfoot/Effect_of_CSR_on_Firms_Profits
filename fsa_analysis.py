"""
Financial Statement Analysis: The Effect of Mandatory CSR Disclosure on Corporate Profitability
Description: Python implementation of CSR disclosure impact analysis using panel data regression
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print("="*80)
print("Financial Statement Analysis: CSR Disclosure Impact on Profitability")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")

# Load the combined_data.csv (converted from Stata's combined_data.dta)
# This file already has Excel data merged and initial variables created by Stata
df = pd.read_csv('input/combined_data.csv')
print(f"   Loaded combined_data.csv: {len(df)} observations, {len(df.columns)} variables")
print(f"   This file includes Excel merge and Stata-created variables")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n2. Data Preprocessing...")

# Convert year to numeric (needed for Excel data)
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Drop rows with negative sales, total_assets, or ESG scores (where these columns exist)
# Note: Excel-only rows will have NaN for these, so we use .notna() to check first
print("   Removing invalid observations...")
initial_rows = len(df)

# Only filter where the column exists and is not null
if 'sales' in df.columns:
    df = df[(df['sales'].isna()) | (df['sales'] >= 0)]
if 'total_assets' in df.columns:
    df = df[(df['total_assets'].isna()) | (df['total_assets'] >= 0)]
if 'ESG_score' in df.columns:
    df = df[(df['ESG_score'].isna()) | (df['ESG_score'] >= 0)]

print(f"   Dropped {initial_rows - len(df)} observations with negative values")

# Filter to 2012-2021 time period (do this AFTER merge, like Stata)
df = df[(df['year'] >= 2012) & (df['year'] <= 2021)]
print(f"   Filtered to 2012-2021: {len(df)} observations")

# Remove duplicates based on firm (ric) and year
# Keep first occurrence (Excel data comes first in concat, so Excel values are prioritized)
df = df.drop_duplicates(subset=['ric', 'year'], keep='first')
print(f"   After removing duplicates: {len(df)} observations")

# ============================================================================
# 3. VARIABLE CONSTRUCTION
# ============================================================================
print("\n3. Creating Variables...")

# Some variables already exist in combined_data (leverage, lag_assets, firm_id)
# Create or recreate the variables needed for analysis

# Leverage (may already exist, but recreate to be sure)
if 'leverage' not in df.columns or df['leverage'].isna().all():
    df['leverage'] = df['total_assets'] / df['total_liabilities']
    print("   Created leverage variable")
else:
    print("   Leverage variable already exists from Stata")

# Log of Total Assets
df['log_assets'] = np.log(df['total_assets'])

# Market-to-Book Ratio
df['market_to_book'] = df['market_value'] / df['total_assets']

# ROA = Operating Income / Total Assets
df['ROA'] = df['operating_income'] / df['total_assets']

# CSR Indicator: Binary variable (1 if firm reported CSR, 0 otherwise)
# Following Stata's logic: missing CSR_report is treated as NO CSR (=0), not as missing
df['CSR_report_clean'] = df['CSR_report'].fillna('').str.upper().str.strip()
df['csr_indicator'] = 0  # Default to 0 (No CSR)
df.loc[df['CSR_report_clean'] == 'Y', 'csr_indicator'] = 1  # Only Y becomes 1
# Note: This matches Stata's approach where missing/blank is treated as No CSR

# Firm identifier (may already exist)
if 'firm_id' not in df.columns:
    df['firm_id'] = pd.Categorical(df['ric']).codes

# Sort by firm and year
df = df.sort_values(['ric', 'year'])

# Lagged assets (may already exist from Stata)
if 'lag_assets' not in df.columns or df['lag_assets'].isna().all():
    df['lag_assets'] = df.groupby('ric')['total_assets'].shift(1)

# Drop rows with missing values in key variables
print("   Removing observations with missing key variables...")
key_vars = ['ROA', 'leverage', 'log_assets', 'market_to_book', 'csr_indicator', 'year', 'total_assets']
df_clean = df.dropna(subset=key_vars)

# Remove infinite values that may result from division
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=key_vars)

print(f"   Final cleaned dataset: {len(df_clean)} observations")

# ============================================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("4. DESCRIPTIVE STATISTICS")
print("="*80)

# Select key variables for summary statistics
summary_vars = ['leverage', 'ROA', 'log_assets', 'market_to_book', 'csr_indicator']
desc_stats = df_clean[summary_vars].describe()

print("\nSummary Statistics:")
print(desc_stats)

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. PAIRWISE CORRELATIONS")
print("="*80)

# Calculate correlation matrix
corr_matrix = df_clean[summary_vars].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Calculate p-values for correlations
def calculate_pvalues(df):
    """Calculate p-values for correlation matrix"""
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = stats.pearsonr(df[r], df[c])[1]
    return pvalues

pvalues = calculate_pvalues(df_clean[summary_vars])
print("\nP-values for correlations:")
print(pvalues)

# ============================================================================
# 6. OLS REGRESSION (WITHOUT FIXED EFFECTS)
# ============================================================================
print("\n" + "="*80)
print("6. OLS REGRESSION (Baseline Model)")
print("="*80)

# Prepare variables for OLS
X_ols = df_clean[['leverage', 'market_to_book', 'log_assets', 'csr_indicator']].copy()
X_ols = sm.add_constant(X_ols)  # Add intercept
y_ols = df_clean['ROA']

# Run OLS regression
ols_model = sm.OLS(y_ols, X_ols).fit()
print("\nOLS Regression Results:")
print(ols_model.summary())

# Test for heteroskedasticity (Breusch-Pagan test)
print("\nBreusch-Pagan Test for Heteroskedasticity:")
bp_test = het_breuschpagan(ols_model.resid, ols_model.model.exog)
print(f"   LM Statistic: {bp_test[0]:.4f}")
print(f"   P-value: {bp_test[1]:.4f}")
if bp_test[1] < 0.05:
    print("   Result: Heteroskedasticity detected (p < 0.05)")
else:
    print("   Result: No heteroskedasticity detected (p >= 0.05)")

# ============================================================================
# 7. FIXED EFFECTS REGRESSION
# ============================================================================
print("\n" + "="*80)
print("7. FIXED EFFECTS REGRESSION (Panel Data Model)")
print("="*80)

# Prepare panel data
df_panel = df_clean.copy()
df_panel = df_panel.set_index(['ric', 'year'])

# Dependent variable
y_panel = df_panel['ROA']

# Independent variables
X_panel = df_panel[['leverage', 'market_to_book', 'log_assets', 'csr_indicator']]

# Run fixed effects regression with entity (firm) and time (year) effects
fe_model = PanelOLS(y_panel, X_panel, entity_effects=True, time_effects=True)
fe_results = fe_model.fit(cov_type='clustered', cluster_entity=True)

print("\nFixed Effects Regression Results:")
print(fe_results)

# ============================================================================
# 8. WINSORIZATION (OUTLIER HANDLING)
# ============================================================================
print("\n" + "="*80)
print("8. WINSORIZATION (Handling Outliers)")
print("="*80)

# Create a copy for winsorized data
df_winsor = df_clean.copy()

# Winsorize key variables at 1st and 99th percentiles
vars_to_winsorize = ['ROA', 'leverage', 'market_to_book', 'log_assets']

print("\nWinsorizing variables at 1st and 99th percentiles...")
for var in vars_to_winsorize:
    df_winsor[var] = winsorize(df_winsor[var], limits=[0.01, 0.01])
    print(f"   Winsorized: {var}")

# ============================================================================
# 9. FIXED EFFECTS REGRESSION ON WINSORIZED DATA
# ============================================================================
print("\n" + "="*80)
print("9. FIXED EFFECTS REGRESSION (Winsorized Data)")
print("="*80)

# Prepare winsorized panel data
df_panel_winsor = df_winsor.copy()
df_panel_winsor = df_panel_winsor.set_index(['ric', 'year'])

# Dependent variable
y_panel_winsor = df_panel_winsor['ROA']

# Independent variables
X_panel_winsor = df_panel_winsor[['leverage', 'market_to_book', 'log_assets', 'csr_indicator']]

# Run fixed effects regression
fe_model_winsor = PanelOLS(y_panel_winsor, X_panel_winsor, entity_effects=True, time_effects=True)
fe_results_winsor = fe_model_winsor.fit(cov_type='clustered', cluster_entity=True)

print("\nFixed Effects Regression Results (Winsorized):")
print(fe_results_winsor)

# ============================================================================
# 10. VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("10. CREATING VISUALIZATIONS")
print("="*80)

# Set style for better-looking plots
sns.set_style("whitegrid")

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Scatter plot: ROA vs Market-to-Book Ratio
axes[0, 0].scatter(df_winsor['market_to_book'], df_winsor['ROA'],
                   alpha=0.3, s=10, color='purple')
axes[0, 0].set_xlabel('Market-to-Book Ratio')
axes[0, 0].set_ylabel('ROA')
axes[0, 0].set_title('ROA vs Market-to-Book Ratio')
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=0.5)

# 2. Box plot: ROA by CSR Indicator
df_winsor['CSR_Status'] = df_winsor['csr_indicator'].map({0: 'No CSR Report', 1: 'CSR Report'})
df_winsor.boxplot(column='ROA', by='CSR_Status', ax=axes[0, 1])
axes[0, 1].set_xlabel('CSR Reporting Status')
axes[0, 1].set_ylabel('ROA')
axes[0, 1].set_title('ROA Distribution by CSR Reporting Status')
axes[0, 1].get_figure().suptitle('')  # Remove automatic title

# 3. Scatter plot: ROA vs Log Assets
axes[1, 0].scatter(df_winsor['log_assets'], df_winsor['ROA'],
                   alpha=0.3, s=10, color='blue')
axes[1, 0].set_xlabel('Log of Total Assets')
axes[1, 0].set_ylabel('ROA')
axes[1, 0].set_title('ROA vs Firm Size (Log Assets)')
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=0.5)

# 4. Correlation heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('output/fsa_analysis_plots.png', dpi=300, bbox_inches='tight')
print("\n   Plots saved as 'output/fsa_analysis_plots.png'")

# ============================================================================
# 11. SUMMARY OF KEY FINDINGS
# ============================================================================
print("\n" + "="*80)
print("11. KEY FINDINGS SUMMARY")
print("="*80)

print("\nMain Results from Fixed Effects Regression (Winsorized Data):")
print("-" * 60)

# Extract coefficients and p-values
for var in ['csr_indicator', 'leverage', 'market_to_book', 'log_assets']:
    coef = fe_results_winsor.params[var]
    pval = fe_results_winsor.pvalues[var]
    tstat = fe_results_winsor.tstats[var]

    significance = ""
    if pval < 0.001:
        significance = "***"
    elif pval < 0.01:
        significance = "**"
    elif pval < 0.05:
        significance = "*"

    print(f"\n{var:20s}: {coef:>10.6f} {significance}")
    print(f"{'':20s}  (t-stat: {tstat:>6.2f}, p-value: {pval:.4f})")

print("\n" + "-" * 60)
print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05")

print("\n\nInterpretation:")
print("-" * 60)
print("1. CSR Indicator: A negative coefficient suggests that mandatory CSR")
print("   disclosure is associated with LOWER profitability (ROA) in the short term.")
print("\n2. Log Assets: Larger firms tend to have HIGHER profitability.")
print("\n3. Market-to-Book: Firms with higher market valuations relative to book")
print("   value tend to have LOWER profitability.")
print("\n4. Leverage: The relationship between leverage and profitability varies")
print("   depending on the model specification.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# ============================================================================
# 12. EXPORT RESULTS
# ============================================================================
print("\n12. Exporting Results...")

# Save regression results to text file
with open('output/regression_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FINANCIAL STATEMENT ANALYSIS RESULTS\n")
    f.write("="*80 + "\n\n")

    f.write("1. OLS REGRESSION (Baseline)\n")
    f.write("-"*80 + "\n")
    f.write(str(ols_model.summary()))
    f.write("\n\n")

    f.write("2. FIXED EFFECTS REGRESSION (Original Data)\n")
    f.write("-"*80 + "\n")
    f.write(str(fe_results))
    f.write("\n\n")

    f.write("3. FIXED EFFECTS REGRESSION (Winsorized Data)\n")
    f.write("-"*80 + "\n")
    f.write(str(fe_results_winsor))

print("   Regression results saved to 'output/regression_results.txt'")

# Save descriptive statistics
desc_stats.to_csv('output/descriptive_statistics.csv')
print("   Descriptive statistics saved to 'output/descriptive_statistics.csv'")

# Save correlation matrix
corr_matrix.to_csv('output/correlation_matrix.csv')
print("   Correlation matrix saved to 'output/correlation_matrix.csv'")

# Save cleaned dataset
df_clean.to_csv('output/cleaned_data.csv', index=False)
print("   Cleaned dataset saved to 'output/cleaned_data.csv'")

print("\n" + "="*80)
print("All results exported successfully!")
print("="*80)
