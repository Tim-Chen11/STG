"""
ETL Pipeline Configuration
Central configuration for data processing parameters
"""

from datetime import datetime

# Data source paths
DATA_SOURCES = {
    'base_path': 'data',
    'fred': {
        'path': 'FRED',
        'files': {
            'mortgage_rate': {'file': 'mortgage_rate.csv', 'date_col': 'date', 'value_col': 'mortgage_rate'},
            'cpi': {'file': 'cpi.csv', 'date_col': 'date', 'value_col': 'cpi'},
            'unemployment_rate': {'file': 'unemployment_rate.csv', 'date_col': 'date', 'value_col': 'unemployment_rate'},
            'building_permits': {'file': 'building_permits.csv', 'date_col': 'date', 'value_col': 'building_permits'},
            'housing_starts': {'file': 'housing_starts.csv', 'date_col': 'date', 'value_col': 'housing_starts'},
            'income_median': {'file': 'income_median.csv', 'date_col': 'date', 'value_col': 'income_median'},
            'median_sales_price': {'file': 'median_sales_price.csv', 'date_col': 'date', 'value_col': 'median_sales_price'}
        }
    },
    'zillow': {
        'path': 'Zillow',
        'files': {
            'days_on_market': {'file': 'days_on_market.parquet', 'date_col': 'Date'},
            'for_sale_listings': {'file': 'for_sale_listings.parquet', 'date_col': 'Date'},
            'home_values': {'file': 'home_values.parquet', 'date_col': 'Date'},
            'home_values_forecasts': {'file': 'home_values_forecasts.parquet', 'date_col': 'Date'},
            'new_construction': {'file': 'new_construction.parquet', 'date_col': 'Date'},
            'rentals': {'file': 'rentals.parquet', 'date_col': 'Date'},
            'sales': {'file': 'sales.parquet', 'date_col': 'Date'}
        }
    },
    'census': {
        'path': 'USCENSUS',
        'pattern': '*.xls*'
    }
}

# Time period definitions for analysis
TIME_PERIODS = {
    'low_rate': {
        'start': '2020-01-01',
        'end': '2021-12-31',
        'description': 'Low interest rate period during COVID-19'
    },
    'transition': {
        'start': '2022-01-01',
        'end': '2022-12-31',
        'description': 'Transition period with rising rates'
    },
    'high_rate': {
        'start': '2023-01-01',
        'end': '2024-12-31',
        'description': 'High interest rate environment'
    },
    'pre_covid': {
        'start': '2018-01-01',
        'end': '2019-12-31',
        'description': 'Pre-COVID baseline period'
    }
}

# Transformation parameters
TRANSFORM_PARAMS = {
    'resampling': {
        'monthly': 'M',
        'quarterly': 'Q',
        'annual': 'Y'
    },
    'rolling_windows': {
        'short': 3,  # 3-month moving average
        'medium': 6,  # 6-month moving average
        'long': 12   # 12-month moving average
    },
    'yoy_periods': {
        'monthly': 12,
        'quarterly': 4,
        'annual': 1
    }
}

# Regional aggregation levels
AGGREGATION_LEVELS = {
    'state': 'State',
    'metro': 'Metro',
    'county': 'County',
    'zip': 'Region'  # Zillow uses 'Region' for zip codes
}

# Cooling indicator weights
INDICATOR_WEIGHTS = {
    'days_on_market': 0.35,
    'price_cuts': 0.25,
    'inventory': 0.20,
    'price_growth': 0.20
}

# Output configuration
OUTPUT_CONFIG = {
    'processed_data_path': 'processed_data',
    'analysis_results_path': 'analysis_results',
    'visualizations_path': 'visualizations',
    'reports_path': 'reports',
    'formats': {
        'dataframe': 'parquet',
        'series': 'parquet',
        'dict': 'json',
        'metadata': 'json'
    }
}

# Analysis thresholds
ANALYSIS_THRESHOLDS = {
    'cooling_indicators': {
        'extreme_cooling': 50,    # > 50 cooling index
        'strong_cooling': 25,      # 25-50 cooling index
        'moderate_cooling': 10,    # 10-25 cooling index
        'minimal_cooling': 0,      # 0-10 cooling index
        'no_cooling': -10          # < 0 cooling index (still hot)
    },
    'price_changes': {
        'rapid_appreciation': 15,  # > 15% YoY
        'strong_appreciation': 10, # 10-15% YoY
        'moderate_appreciation': 5,# 5-10% YoY
        'flat': 0,                 # -5% to 5% YoY
        'depreciation': -5         # < -5% YoY
    }
}

# Data quality checks
QUALITY_CHECKS = {
    'min_records': 100,           # Minimum records for analysis
    'max_missing_pct': 0.3,       # Maximum 30% missing data
    'outlier_std_devs': 3,        # Flag values > 3 std devs
    'date_range_check': True,     # Verify date ranges
    'duplicate_check': True       # Check for duplicate records
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 150,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'husl',
    'save_formats': ['png', 'svg']
}