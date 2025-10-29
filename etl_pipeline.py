"""
Modular ETL Pipeline for Real Estate Market Analysis
Handles FRED, Zillow, and US Census data processing
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataExtractor:
    """Handles data extraction from various sources"""
    
    def __init__(self, base_path='data'):
        self.base_path = Path(base_path)
        self.fred_path = self.base_path / 'FRED'
        self.zillow_path = self.base_path / 'Zillow'
        self.census_path = self.base_path / 'USCENSUS'
        
    def extract_fred_data(self):
        """Extract all FRED economic indicator data"""
        logger.info("Extracting FRED data...")
        
        fred_data = {}
        fred_files = {
            'mortgage_rate': 'mortgage_rate.csv',
            'cpi': 'cpi.csv',
            'unemployment_rate': 'unemployment_rate.csv',
            'building_permits': 'building_permits.csv',
            'housing_starts': 'housing_starts.csv',
            'income_median': 'income_median.csv',
            'median_sales_price': 'median_sales_price.csv'
        }
        
        for key, filename in fred_files.items():
            filepath = self.fred_path / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                fred_data[key] = df
                logger.info(f"  - Extracted {key}: {len(df)} records")
            else:
                logger.warning(f"  - File not found: {filepath}")
                
        return fred_data
    
    def extract_zillow_data(self):
        """Extract all Zillow regional real estate data"""
        logger.info("Extracting Zillow data...")
        
        zillow_data = {}
        zillow_files = {
            'days_on_market': 'days_on_market.parquet',
            'for_sale_listings': 'for_sale_listings.parquet',
            'home_values': 'home_values.parquet',
            'home_values_forecasts': 'home_values_forecasts.parquet',
            'new_construction': 'new_construction.parquet',
            'rentals': 'rentals.parquet',
            'sales': 'sales.parquet'
        }
        
        for key, filename in zillow_files.items():
            filepath = self.zillow_path / filename
            if filepath.exists():
                df = pd.read_parquet(filepath)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                zillow_data[key] = df
                logger.info(f"  - Extracted {key}: {len(df)} records")
            else:
                logger.warning(f"  - File not found: {filepath}")
                
        return zillow_data
    
    def extract_census_data(self):
        """Extract US Census migration data"""
        logger.info("Extracting Census data...")
        
        census_data = {}
        census_files = [f for f in self.census_path.glob('*.xls*')]
        
        for filepath in census_files:
            year = None
            for yr in range(2010, 2024):
                if str(yr) in filepath.name:
                    year = yr
                    break
            
            if year:
                try:
                    df = pd.read_excel(filepath, sheet_name=0)
                    census_data[f'migration_{year}'] = df
                    logger.info(f"  - Extracted migration data for {year}")
                except Exception as e:
                    logger.error(f"  - Error reading {filepath.name}: {e}")
                    
        return census_data


class DataTransformer:
    """Handles all data transformations"""
    
    def __init__(self):
        self.time_periods = {
            'low_rate': {'start': '2020-01-01', 'end': '2021-12-31'},
            'transition': {'start': '2022-01-01', 'end': '2022-12-31'},
            'high_rate': {'start': '2023-01-01', 'end': '2024-12-31'}
        }
    
    def transform_fred_data(self, fred_data):
        """Transform FRED data for analysis"""
        logger.info("Transforming FRED data...")
        
        transformed = {}
        
        # Mortgage rates - weekly to monthly
        if 'mortgage_rate' in fred_data:
            df = fred_data['mortgage_rate'].set_index('date')
            transformed['mortgage_rate_weekly'] = df
            transformed['mortgage_rate_monthly'] = df.resample('M').mean()
            transformed['mortgage_rate_quarterly'] = df.resample('Q').mean()
            logger.info("  - Transformed mortgage rates to multiple frequencies")
        
        # Economic indicators - add YoY changes
        for key in ['cpi', 'unemployment_rate', 'building_permits', 'housing_starts']:
            if key in fred_data:
                df = fred_data[key].set_index('date')
                transformed[key] = df
                transformed[f'{key}_yoy'] = df.pct_change(12) * 100
                transformed[f'{key}_ma3'] = df.rolling(3).mean()
                logger.info(f"  - Added YoY and MA3 for {key}")
        
        # Median sales price - quarterly to monthly interpolation
        if 'median_sales_price' in fred_data:
            df = fred_data['median_sales_price'].set_index('date')
            transformed['median_sales_price_quarterly'] = df
            transformed['median_sales_price_yoy'] = df.pct_change(4) * 100
            logger.info("  - Transformed median sales price")
        
        # Income median - annual handling
        if 'income_median' in fred_data:
            df = fred_data['income_median'].set_index('date')
            transformed['income_median'] = df
            transformed['income_median_yoy'] = df.pct_change(1) * 100
            logger.info("  - Transformed median income")
            
        return transformed
    
    def transform_zillow_data(self, zillow_data):
        """Transform Zillow data for regional analysis"""
        logger.info("Transforming Zillow data...")
        
        transformed = {}
        
        # Days on market transformations
        if 'days_on_market' in zillow_data:
            df = zillow_data['days_on_market']
            
            # State-level aggregations
            state_metrics = self._aggregate_by_region(df, 'State')
            transformed['days_on_market_state'] = state_metrics
            
            # National aggregations
            national_metrics = self._aggregate_national(df)
            transformed['days_on_market_national'] = national_metrics
            
            logger.info("  - Transformed days on market data")
        
        # For sale listings transformations
        if 'for_sale_listings' in zillow_data:
            df = zillow_data['for_sale_listings']
            
            # State-level inventory metrics
            state_inventory = self._aggregate_by_region(df, 'State')
            transformed['inventory_state'] = state_inventory
            
            # National inventory
            national_inventory = self._aggregate_national(df)
            transformed['inventory_national'] = national_inventory
            
            logger.info("  - Transformed for sale listings data")
        
        # Home values transformations
        if 'home_values' in zillow_data:
            df = zillow_data['home_values']
            
            # Extract different tiers
            value_cols = [col for col in df.columns if 'ZHVI' in col]
            for col in value_cols:
                tier_name = col.split(' ')[0].lower() + '_tier'
                state_values = self._aggregate_by_region(df, 'State', metric_col=col)
                transformed[f'home_values_{tier_name}'] = state_values
            
            logger.info("  - Transformed home values data")
        
        # Sales data transformations
        if 'sales' in zillow_data:
            df = zillow_data['sales']
            state_sales = self._aggregate_by_region(df, 'State')
            transformed['sales_state'] = state_sales
            logger.info("  - Transformed sales data")
        
        return transformed
    
    def _aggregate_by_region(self, df, region_col='State', metric_col=None):
        """Helper function to aggregate data by region"""
        if metric_col is None:
            # Auto-detect numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Date' in df.columns:
                numeric_cols = [col for col in numeric_cols if col != 'Date']
        else:
            numeric_cols = [metric_col]
        
        aggregated = {}
        for period_name, period_dates in self.time_periods.items():
            period_df = df[(df['Date'] >= period_dates['start']) & 
                          (df['Date'] <= period_dates['end'])]
            
            if len(period_df) > 0:
                for col in numeric_cols:
                    if col in period_df.columns:
                        agg_key = f'{col}_{period_name}'
                        aggregated[agg_key] = period_df.groupby(region_col)[col].mean()
        
        return aggregated
    
    def _aggregate_national(self, df, metric_col=None):
        """Helper function to aggregate data nationally"""
        if 'Date' not in df.columns:
            return {}
        
        if metric_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [metric_col]
        
        national = {}
        for col in numeric_cols:
            if col in df.columns:
                national[col] = df.groupby('Date')[col].mean()
        
        return national
    
    def create_derived_metrics(self, transformed_fred, transformed_zillow):
        """Create derived metrics from transformed data"""
        logger.info("Creating derived metrics...")
        
        metrics = {}
        
        # Period comparisons for state-level data
        if 'days_on_market_state' in transformed_zillow:
            dom_data = transformed_zillow['days_on_market_state']
            metrics['state_period_comparisons'] = dom_data
            logger.info("  - Created state period comparisons")
        
        # National aggregations
        if 'mortgage_rate_monthly' in transformed_fred:
            rates = transformed_fred['mortgage_rate_monthly']
            period_stats = {}
            for period_name, period_info in self.time_periods.items():
                period_data = rates[period_info['start']:period_info['end']]
                if len(period_data) > 0:
                    period_stats[f'{period_name}_avg'] = period_data.mean()
                    period_stats[f'{period_name}_std'] = period_data.std()
            metrics['rate_period_stats'] = period_stats
            logger.info("  - Created rate period statistics")
        
        return metrics


class DataLoader:
    """Handles data loading and storage"""
    
    def __init__(self, output_path='processed_data'):
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
    def save_transformed_data(self, data, category_name):
        """Save transformed data to files"""
        logger.info(f"Saving {category_name} data...")
        
        category_path = self.output_path / category_name
        category_path.mkdir(exist_ok=True)
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                # Save as parquet for DataFrames/Series
                filepath = category_path / f'{key}.parquet'
                if isinstance(value, pd.Series):
                    value = value.to_frame()
                value.to_parquet(filepath)
                logger.info(f"  - Saved {key} to {filepath}")
            elif isinstance(value, dict):
                # Check if dict contains pandas objects - save as parquet if so
                has_pandas = any(isinstance(v, (pd.Series, pd.DataFrame)) for v in value.values())
                
                if has_pandas:
                    # Convert dict of Series/DataFrames to a DataFrame and save as parquet
                    df_dict = {}
                    for k, v in value.items():
                        if isinstance(v, pd.Series):
                            df_dict[k] = v
                        elif isinstance(v, pd.DataFrame):
                            # If it's a DataFrame, we'll save each column with a prefix
                            for col in v.columns:
                                df_dict[f"{k}_{col}"] = v[col]
                    
                    if df_dict:
                        combined_df = pd.DataFrame(df_dict)
                        filepath = category_path / f'{key}.parquet'
                        combined_df.to_parquet(filepath)
                        logger.info(f"  - Saved {key} to {filepath}")
                else:
                    # Save as JSON for simple dict objects
                    filepath = category_path / f'{key}.json'
                    with open(filepath, 'w') as f:
                        json.dump(value, f, default=str, indent=2)
                    logger.info(f"  - Saved {key} to {filepath}")
                
    def load_processed_data(self, category_name, dataset_name):
        """Load specific processed dataset"""
        filepath = self.output_path / category_name / f'{dataset_name}.parquet'
        if filepath.exists():
            return pd.read_parquet(filepath)
        
        filepath = self.output_path / category_name / f'{dataset_name}.json'
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        
        logger.warning(f"Dataset not found: {category_name}/{dataset_name}")
        return None
    


class ETLPipeline:
    """Main ETL Pipeline orchestrator"""
    
    def __init__(self):
        self.extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.loader = DataLoader()
        
    def run_full_pipeline(self):
        """Run the complete ETL pipeline"""
        logger.info("="*60)
        logger.info("Starting ETL Pipeline")
        logger.info("="*60)
        
        # Extract
        fred_data = self.extractor.extract_fred_data()
        zillow_data = self.extractor.extract_zillow_data()
        census_data = self.extractor.extract_census_data()
        
        # Transform
        fred_transformed = self.transformer.transform_fred_data(fred_data)
        zillow_transformed = self.transformer.transform_zillow_data(zillow_data)
        
        # Create derived metrics
        metrics = self.transformer.create_derived_metrics(
            fred_transformed, zillow_transformed
        )
        
        # Load
        self.loader.save_transformed_data(fred_transformed, 'fred')
        self.loader.save_transformed_data(zillow_transformed, 'zillow')
        self.loader.save_transformed_data(metrics, 'metrics')
        
        logger.info("="*60)
        logger.info("ETL Pipeline Complete")
        logger.info("="*60)
        
        return {
            'fred': fred_transformed,
            'zillow': zillow_transformed,
            'metrics': metrics
        }
    
    def run_incremental_update(self, data_source='all'):
        """Run incremental update for specific data source"""
        logger.info(f"Running incremental update for: {data_source}")
        # Implementation for incremental updates
        pass


if __name__ == "__main__":
    # Run the ETL pipeline
    pipeline = ETLPipeline()
    results = pipeline.run_full_pipeline()
    
    # Print summary
    print("\nETL Pipeline Summary:")
    print(f"- FRED datasets processed: {len(results.get('fred', {}))}")
    print(f"- Zillow datasets processed: {len(results.get('zillow', {}))}")
    print(f"- Derived metrics created: {len(results.get('metrics', {}))}")
    print("\nData successfully transformed and saved to 'processed_data' directory")