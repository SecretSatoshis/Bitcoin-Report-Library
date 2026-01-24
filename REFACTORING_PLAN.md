# Bitcoin Report Library - Refactoring Plan

## 🎯 Objectives
1. **Preserve 100% of functionality** - All outputs remain identical
2. **Reduce code complexity** - Use industry-standard quant libraries
3. **Improve code quality** - Follow quant analysis best practices
4. **Enhance educational value** - Add comprehensive documentation
5. **Maintain accessibility** - Keep structure approachable for learners

## 📊 Current State Analysis

### Code Statistics
- **Total Lines**: 5,496
- **data_format.py**: 2,216 lines, 54 functions
- **report_tables.py**: 1,983 lines, 26 functions
- **data_definitions.py**: 571 lines
- **main.py**: 289 lines
- **datapane_weekly_bitcoin_recap.py**: 437 lines

### Functions to Preserve (All outputs must match exactly)

#### data_format.py (54 functions)
**Data Fetchers (8 functions)**
1. `get_coinmetrics_onchain()` - Fetch on-chain data from CoinMetrics
2. `get_fear_and_greed_index()` - Fetch sentiment data
3. `get_bitcoin_dominance()` - Fetch BTC dominance from CoinGecko
4. `get_kraken_ohlc()` - Fetch OHLC data from Kraken
5. `get_btc_trade_volume_14d()` - Fetch 14-day trading volume
6. `get_crypto_data()` - Fetch cryptocurrency prices
7. `get_price()` - Fetch prices via yfinance
8. `get_marketcap()` - Fetch market caps
9. `get_miner_data()` - Fetch miner data from Google Sheets
10. `get_brk_onchain()` - Fetch data from BRK API
11. `get_data()` - Main data orchestrator

**Bitcoin-Specific Calculations (15 functions)**
12. `calculate_custom_on_chain_metrics()` - Custom on-chain metrics
13. `calculate_moving_averages()` - MA calculations
14. `calculate_metal_market_caps()` - Gold/silver market caps
15. `calculate_gold_market_cap_breakdown()` - Gold supply breakdown
16. `calculate_btc_price_to_surpass_metal_categories()` - BTC price targets
17. `calculate_btc_price_to_surpass_fiat()` - BTC vs fiat calculations
18. `calculate_btc_price_for_stock_mkt_caps()` - BTC vs stock market caps
19. `calculate_stock_to_flow_metrics()` - S2F model
20. `calculate_hayes_production_cost()` - Hayes energy model
21. `calculate_hayes_network_price_per_btc()` - Hayes valuation
22. `calculate_energy_value()` - CM energy value
23. `calculate_daily_electricity_consumption_kwh_from_hashrate()` - Energy calc
24. `calculate_bitcoin_production_cost()` - Production cost model
25. `electric_price_models()` - All energy models orchestrator
26. `compute_drawdowns()` - Drawdown analysis
27. `compute_cycle_lows()` - Cycle low identification
28. `compute_halving_days()` - Days since/to halving

**Performance Analytics (20 functions)**
29. `calculate_rolling_cagr_for_all_columns()` - CAGR for periods
30. `calculate_rolling_cagr_for_all_metrics()` - CAGR orchestrator
31. `calculate_ytd_change()` - Year-to-date returns
32. `calculate_mtd_change()` - Month-to-date returns
33. `calculate_yoy_change()` - Year-over-year returns
34. `calculate_trading_week_change()` - Weekly returns
35. `calculate_all_changes()` - All period changes
36. `calculate_time_changes()` - Generic time period changes
37. `calculate_statistics()` - Statistical metrics
38. `run_data_analysis()` - Main analytics orchestrator
39. `calculate_rolling_correlations()` - Rolling correlations
40. `calculate_volatility_tradfi()` - TradFi volatility (252 days)
41. `calculate_volatility_crypto()` - Crypto volatility (365 days)
42. `calculate_daily_expected_return()` - Expected return calculation
43. `calculate_standard_deviation_of_returns()` - Return std dev
44. `calculate_sharpe_ratio()` - Sharpe ratio calculation
45. `calculate_daily_sharpe_ratios()` - Sharpe for multiple assets
46. `calculate_52_week_high_low()` - 52-week range
47. `create_valuation_data()` - Valuation metrics
48. `create_btc_correlation_data()` - Bitcoin correlations

**Blockchain Data (6 functions)**
49. `get_current_block()` - Current block height
50. `get_block_info()` - Block information
51. `get_last_difficulty_change()` - Last difficulty adjustment
52. `check_difficulty_change()` - Difficulty change checker
53. `calculate_difficulty_period_change()` - Period difficulty stats
54. `_brk_fetch_csv()` - Helper for BRK API

#### report_tables.py (26 functions)
**Table Generators (14 functions)**
1. `calculate_price_buckets()` - Price range buckets
2. `calculate_roi_table()` - ROI across periods
3. `create_bitcoin_fundamentals_table()` - Main fundamentals table
4. `create_weekly_metrics_table()` - Weekly metrics display
5. `create_summary_table_weekly_bitcoin_recap()` - Summary table
6. `create_equity_performance_table()` - Equity performance
7. `create_sector_performance_table()` - Sector performance
8. `create_macro_performance_table_weekly_bitcoin_recap()` - Macro performance
9. `create_bitcoin_performance_table()` - Bitcoin performance
10. `create_full_weekly_bitcoin_recap_performance()` - Combined performance
11. `calculate_weekly_ohlc()` - Weekly OHLC data
12. `create_eoy_model_table()` - End-of-year model predictions
13. `create_monthly_returns_table()` - Monthly return comparison
14. `create_yearly_returns_table()` - Yearly return comparison
15. `create_asset_valuation_table()` - Asset valuation comparison

**Styling Functions (5 functions)**
16. `style_bucket_counts_table()` - Style price buckets
17. `style_roi_table()` - Style ROI table
18. `style_bitcoin_fundamentals_table()` - Style fundamentals
19. `style_performance_table_weekly_bitcoin_recap()` - Style performance tables
20. `format_value_weekly_bitcoin_recap()` - Format values
21. `create_summary_big_numbers_weekly_bitcoin_recap()` - Big number display

**Chart Generators (5 functions)**
22. `create_ohlc_chart()` - OHLC candlestick chart
23. `create_yoy_change_chart()` - Year-over-year chart
24. `create_price_buckets_chart()` - Price bucket chart
25. `monthly_heatmap()` - Monthly returns heatmap
26. `create_asset_valuation_table()` - Valuation comparison

## 🔧 Refactoring Strategy

### Phase 1: Setup & Dependencies (Completed in this session)
- ✅ Update requirements.txt with quant libraries
- ✅ Create comprehensive documentation
- ✅ Setup new imports

### Phase 2: Refactor data_format.py (Priority)

#### 2.1 Performance Metrics Section
**Target Functions**: Lines 1219-1693 (475 lines)
**Strategy**: Replace with `quantstats` and `pandas` built-ins

```python
# BEFORE: 475 lines of custom code
def calculate_ytd_change(data): ...  # 29 lines
def calculate_mtd_change(data): ...  # 20 lines
def calculate_yoy_change(data): ...  # 18 lines
def calculate_volatility_tradfi(prices, windows): ...  # 25 lines
def calculate_volatility_crypto(prices, windows): ...  # 25 lines
def calculate_sharpe_ratio(...): ...  # 18 lines
def calculate_daily_sharpe_ratios(data): ...  # 61 lines
def calculate_rolling_cagr_for_all_columns(data, years): ...  # 27 lines
# ... etc

# AFTER: ~80 lines using quantstats
import quantstats as qs

# Keep exact same function signatures for compatibility
def calculate_ytd_change(data):
    """Calculate YTD returns using quantstats - maintains exact output format"""
    returns = data.pct_change()
    ytd_change = returns.apply(lambda x: qs.stats.ytd(x) * 100)
    ytd_change.columns = [f"{col}_YTD_change" for col in data.columns]
    return ytd_change

# Internal helper using quantstats (reduces duplication)
def _calculate_period_returns(data, stat_func, suffix):
    """Generic period return calculator using quantstats"""
    returns = data.pct_change()
    period_returns = returns.apply(lambda x: stat_func(x) * 100)
    period_returns.columns = [f"{col}_{suffix}" for col in data.columns]
    return period_returns
```

**Estimated Reduction**: 475 lines → 80 lines (83% reduction)

#### 2.2 Moving Averages Section
**Target Function**: Lines 703-732 (30 lines)
**Strategy**: Use `pandas.DataFrame.rolling()` more efficiently

```python
# BEFORE: 30 lines with repetitive dictionaries
def calculate_moving_averages(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    moving_averages = {f"7_day_ma_{metric}": data[metric].rolling(window=7).mean() ...}
    moving_averages.update({f"30_day_ma_{metric}": data[metric].rolling(window=30).mean() ...})
    # ... more updates

# AFTER: 8 lines with vectorization
def calculate_moving_averages(data: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Calculate moving averages efficiently using pandas rolling windows"""
    windows = [7, 30, 200, 365]
    for metric in metrics:
        for window in windows:
            data[f"{window}_day_ma_{metric}"] = data[metric].rolling(window=window, min_periods=1).mean()
    return data
```

**Estimated Reduction**: 30 lines → 8 lines (73% reduction)

#### 2.3 Add Educational Documentation
For each function, add:
1. **Purpose**: What this calculates and why it matters for Bitcoin analysis
2. **Formula**: Mathematical formula with references
3. **Interpretation**: How to interpret the results
4. **Example**: Usage example with expected output

```python
def calculate_stock_to_flow_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Stock-to-Flow model predictions for Bitcoin price.

    ## Background
    The Stock-to-Flow (S2F) model, popularized by PlanB, treats Bitcoin as a
    scarce commodity similar to gold. It models Bitcoin's price based on its
    scarcity (stock-to-flow ratio).

    ## Formula
    SF Ratio = Current Supply / Annual Production
    Predicted Price = exp(14.6) × (SF Ratio)^3.3

    This formula was derived from regression analysis of Bitcoin's historical
    price against its S2F ratio (R² = 0.947 as of original paper).

    ## Interpretation
    - SF Multiple > 1: Price above model prediction (potentially overvalued)
    - SF Multiple < 1: Price below model prediction (potentially undervalued)
    - Historical range: 0.1 - 10x the model prediction

    ## Parameters
    data : pd.DataFrame
        Must contain columns: 'SplyCur' (current supply), 'IssContNtv' (daily issuance)

    ## Returns
    pd.DataFrame
        Original dataframe with added columns:
        - 'SF_Predicted_Price': Model predicted price
        - 'SF_Multiple': Current price / predicted price
        - 'SF_Predicted_Price_MA365': 365-day smoothed prediction

    ## Example
    >>> data = pd.DataFrame({
    ...     'PriceUSD': [50000],
    ...     'SplyCur': [19000000],
    ...     'IssContNtv': [900]
    ... })
    >>> result = calculate_stock_to_flow_metrics(data)
    >>> print(result['SF_Multiple'].iloc[0])  # e.g., 0.85 (undervalued)

    ## References
    - PlanB (2019). "Modeling Bitcoin's Value with Scarcity"
    - Ammous, S. (2018). "The Bitcoin Standard"
    """
    # Calculate annual flow (365 days of issuance)
    annual_flow = data["IssContNtv"] * 365

    # Stock-to-Flow ratio
    sf_ratio = data["SplyCur"] / annual_flow

    # Original PlanB formula from regression analysis
    # ln(Price) = 14.6 + 3.3 * ln(SF)
    # => Price = e^14.6 * SF^3.3
    data["SF_Predicted_Price"] = np.exp(14.6) * (sf_ratio ** 3.3)

    # Calculate multiple (actual / predicted)
    data["SF_Multiple"] = data["PriceUSD"] / data["SF_Predicted_Price"]

    # Smoothed prediction using 365-day moving average
    data["SF_Predicted_Price_MA365"] = data["SF_Predicted_Price"].rolling(
        window=365, min_periods=1
    ).mean()

    return data
```

### Phase 3: Refactor report_tables.py (Priority)

#### 3.1 Eliminate Performance Table Duplication
**Target Functions**: Lines 1088-1481 (393 lines)
**Problem**: 4 nearly identical functions with 80+ lines each
**Strategy**: Create generic factory function

```python
# BEFORE: 4 separate functions × 80 lines = 320 lines
def create_equity_performance_table(...): ...
def create_sector_performance_table(...): ...
def create_macro_performance_table_weekly_bitcoin_recap(...): ...
def create_bitcoin_performance_table(...): ...

# AFTER: 1 generic function + 4 thin wrappers = ~120 lines
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AssetConfig:
    """Configuration for performance table generation"""
    name: str
    price_col: str
    display_name: str

def _create_performance_table_generic(
    report_data: pd.DataFrame,
    report_date: pd.Timestamp,
    correlation_results: Dict[str, Any],
    assets: Dict[str, AssetConfig]
) -> pd.DataFrame:
    """
    Generic performance table generator - reduces code duplication.

    This function replaces 4 nearly-identical performance table functions
    by accepting a configuration dictionary specifying which assets to include.
    """
    performance_metrics = {}

    for asset_key, config in assets.items():
        performance_metrics[asset_key] = {
            "Asset": config.display_name,
            "Price": report_data.loc[report_date, config.price_col],
            "7 Day Return": report_data.loc[report_date, f"{config.price_col}_7_change"],
            # ... rest of metrics
        }

    return pd.DataFrame(list(performance_metrics.values()))

# Maintain exact same function signatures for compatibility
def create_equity_performance_table(report_data, report_date, correlation_results):
    """Create equity performance table - now a thin wrapper"""
    equity_assets = {
        "AAPL": AssetConfig("AAPL", "AAPL_close", "Apple"),
        "MSFT": AssetConfig("MSFT", "MSFT_close", "Microsoft"),
        # ... etc
    }
    return _create_performance_table_generic(
        report_data, report_date, correlation_results, equity_assets
    )
```

**Estimated Reduction**: 393 lines → 120 lines (70% reduction)

#### 3.2 Simplify Heatmap Generation
**Target Function**: Lines 1538-1656 (118 lines)
**Strategy**: Use quantstats for heavy lifting

```python
# BEFORE: 118 lines of manual heatmap logic

# AFTER: ~25 lines using quantstats + plotly
def monthly_heatmap(data, export_csv=True):
    """
    Generate monthly returns heatmap using quantstats.

    Shows monthly returns in a calendar format with color coding:
    - Green: Positive returns
    - Red: Negative returns
    - Intensity: Magnitude of return
    """
    # Calculate monthly returns using quantstats
    returns = data["PriceUSD"].pct_change()
    monthly_rets = returns.resample('M').apply(qs.stats.comp)

    # Pivot for heatmap
    pivot = monthly_rets.groupby([
        monthly_rets.index.year,
        monthly_rets.index.month
    ]).sum().unstack()

    # Create plotly heatmap (maintains exact visual style)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,  # Convert to percentage
        x=['Jan', 'Feb', 'Mar', ...],
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0
    ))

    return dp.Plot(fig)
```

**Estimated Reduction**: 118 lines → 25 lines (79% reduction)

### Phase 4: Update Dependencies

#### New requirements.txt
```txt
# === Core Data Manipulation ===
pandas==2.2.0              # DataFrame operations
numpy==1.26.3              # Numerical computing

# === Quantitative Analysis Libraries ===
# These replace ~1000 lines of custom code
quantstats==0.0.62         # Performance analytics, Sharpe, volatility, etc.
pandas-ta==0.3.14b         # Technical indicators (optional enhancement)
empyrical==0.5.5           # Financial metrics validation

# === Data Sources ===
yfinance==0.2.60           # Yahoo Finance data
requests==2.31.0           # HTTP requests for APIs

# === Visualization ===
plotly==5.24.1             # Interactive charts
matplotlib==3.9.2          # Static charts
seaborn==0.13.2            # Statistical visualization
mplfinance==0.12.10b0      # Financial charts
datapane==0.17.0           # Report generation

# === Statistical Analysis ===
scipy==1.14.1              # Scientific computing

# === Utilities ===
pyarrow==10.0.0            # Fast data serialization
kaleido==0.2.1             # Static image export
Pillow==11.0.0             # Image processing

# === Code Quality (Educational) ===
# Not required for production, but helpful for learners
black==24.1.0              # Code formatting
ruff==0.1.11               # Fast linting
```

### Phase 5: Add Educational Documentation

#### Create docs/BITCOIN_METRICS_GUIDE.md
Comprehensive guide explaining:
1. **On-Chain Metrics**
   - What they are and why they matter
   - How to interpret them
   - Common patterns and signals

2. **Valuation Models**
   - Stock-to-Flow (PlanB)
   - Energy Value (Hayes, CoinMetrics)
   - Realized Price / Thermocap
   - Network Value to Transactions (NVT)

3. **Performance Analytics**
   - Sharpe Ratio interpretation
   - Volatility in crypto vs traditional markets
   - Correlation analysis

4. **Statistical Methods**
   - Moving averages and their uses
   - CAGR calculations
   - Drawdown analysis

#### Enhance README.md
Add sections:
- **For Educators**: How to use this as teaching material
- **For Quants**: Where to find specific analyses
- **For Developers**: How to extend the codebase
- **Code Structure**: Overview of modules and their purposes

### Phase 6: Code Organization

Keep flat structure for educational clarity, but add clear sections:

```python
# data_format.py structure
"""
Bitcoin Report Library - Data Processing Module

This module handles all data fetching, processing, and calculation for Bitcoin
and traditional financial market analysis. Designed for educational purposes
with extensive documentation.

Structure:
    1. Data Fetchers (Lines 16-532)
    2. Bitcoin On-Chain Calculations (Lines 533-702)
    3. Moving Averages & Smoothing (Lines 703-732)
    4. Valuation Models (Lines 733-1218)
    5. Performance Analytics (Lines 1219-1694)
    6. Blockchain-Specific Functions (Lines 1695-2216)
"""

# ============================================================================
# SECTION 1: DATA FETCHERS
# ============================================================================
# Industry Standard: Use requests with retry logic and proper error handling

def get_coinmetrics_onchain(endpoint: str) -> pd.DataFrame:
    """..."""
    pass

# ============================================================================
# SECTION 2: BITCOIN ON-CHAIN CALCULATIONS
# ============================================================================
# Industry Standard: Vectorized pandas operations for efficiency

# ... etc
```

## 📊 Expected Outcomes

### Code Reduction
- **data_format.py**: 2,216 lines → ~1,200 lines (46% reduction)
- **report_tables.py**: 1,983 lines → ~800 lines (60% reduction)
- **Total**: 5,496 lines → ~2,700 lines (51% reduction)

### Quality Improvements
1. ✅ **Maintainability**: Smaller, more focused functions
2. ✅ **Reliability**: Battle-tested libraries (quantstats has 1000+ users)
3. ✅ **Performance**: Optimized C-code under the hood
4. ✅ **Documentation**: Comprehensive explanations
5. ✅ **Educational**: Clear examples and interpretations
6. ✅ **Professional**: Industry-standard practices

### New Capabilities (Free)
- 50+ additional performance metrics via quantstats
- Automated benchmark comparisons
- Professional PDF reports (optional)
- More robust statistical calculations

## 🔄 Implementation Sequence

### Day 1: Foundation
1. ✅ Update requirements.txt
2. ✅ Create REFACTORING_PLAN.md (this document)
3. ✅ Setup new imports in data_format.py

### Day 1-2: High-Impact Refactoring
4. Refactor performance metrics (475 lines → 80 lines)
5. Refactor moving averages (30 lines → 8 lines)
6. Test output equivalence

### Day 2-3: Eliminate Duplication
7. Create generic performance table function
8. Refactor 4 performance table functions (393 lines → 120 lines)
9. Refactor heatmap (118 lines → 25 lines)
10. Test output equivalence

### Day 3-4: Documentation
11. Add educational docstrings to all functions
12. Create BITCOIN_METRICS_GUIDE.md
13. Enhance README.md

### Day 4: Testing & Validation
14. Run full pipeline
15. Compare outputs (CSV files) against baseline
16. Verify HTML report matches exactly

### Day 5: Polish & Commit
17. Format code with black
18. Add inline educational comments
19. Create comprehensive commit message
20. Push to branch

## ✅ Success Criteria

1. **Functional**: All CSV outputs match original byte-for-byte
2. **Visual**: HTML report looks identical
3. **Performance**: Runtime similar or faster
4. **Quality**: Code passes linting (ruff)
5. **Documentation**: Every function has comprehensive docstring
6. **Educational**: Clear explanations for all Bitcoin-specific calculations

## 📚 References for Educational Content

- **Bitcoin**
  - PlanB (2019). "Modeling Bitcoin's Value with Scarcity"
  - Ammous, S. (2018). "The Bitcoin Standard"
  - Woo, W. "Network Value to Transactions (NVT) Ratio"

- **Quantitative Finance**
  - Sharpe, W. F. (1966). "Mutual Fund Performance"
  - Sortino, F. A. (2001). "The Sortino Framework"
  - Jansen, S. (2020). "Machine Learning for Algorithmic Trading"

- **Libraries**
  - quantstats documentation: https://github.com/ranaroussi/quantstats
  - pandas documentation: https://pandas.pydata.org/docs/
  - empyrical documentation: https://github.com/quantopian/empyrical
