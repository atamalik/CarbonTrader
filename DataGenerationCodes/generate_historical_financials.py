import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import lognorm, gamma, norm, weibull_min, truncnorm, expon
import logging
import json
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/historical_financials.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Connect to the database
conn = sqlite3.connect('sec_financials.db')
cursor = conn.cursor()

# Economic factors and sector trends for 2020-2024
# These represent realistic growth patterns and economic conditions
economic_factors = {
    2020: {
        'overall_growth': -0.05,  # COVID-19 impact
        'sector_multipliers': {
            'Manufacturing': 0.85,  # Supply chain disruptions
            'Finance, Insurance & Real Estate': 0.90,  # Market volatility
            'Services': 0.80,  # Service sector hit hard
            'Transportation and Public Utilities': 0.75,  # Travel restrictions
            'Mining': 0.70,  # Reduced demand
            'Retail Trade': 0.85,  # Store closures
            'Wholesale Trade': 0.80,  # Supply chain issues
            'Construction': 0.75,  # Project delays
            'Agriculture': 0.95,  # Essential services
            'Other': 0.85
        }
    },
    2021: {
        'overall_growth': 0.15,  # Recovery year
        'sector_multipliers': {
            'Manufacturing': 1.20,  # Supply chain recovery
            'Finance, Insurance & Real Estate': 1.15,  # Market recovery
            'Services': 1.10,  # Gradual recovery
            'Transportation and Public Utilities': 1.25,  # Pent-up demand
            'Mining': 1.30,  # Commodity boom
            'Retail Trade': 1.20,  # E-commerce growth
            'Wholesale Trade': 1.15,  # Supply chain normalization
            'Construction': 1.10,  # Infrastructure spending
            'Agriculture': 1.05,  # Stable growth
            'Other': 1.10
        }
    },
    2022: {
        'overall_growth': 0.08,  # Moderate growth
        'sector_multipliers': {
            'Manufacturing': 1.05,  # Normalization
            'Finance, Insurance & Real Estate': 0.95,  # Interest rate impact
            'Services': 1.08,  # Continued recovery
            'Transportation and Public Utilities': 1.10,  # Steady growth
            'Mining': 1.15,  # Continued commodity strength
            'Retail Trade': 1.05,  # Normalization
            'Wholesale Trade': 1.08,  # Steady growth
            'Construction': 1.12,  # Infrastructure projects
            'Agriculture': 1.08,  # Food security focus
            'Other': 1.05
        }
    },
    2023: {
        'overall_growth': 0.12,  # Strong growth
        'sector_multipliers': {
            'Manufacturing': 1.15,  # Reshoring trends
            'Finance, Insurance & Real Estate': 1.10,  # Market stability
            'Services': 1.12,  # Digital transformation
            'Transportation and Public Utilities': 1.08,  # Steady growth
            'Mining': 1.20,  # Energy transition
            'Retail Trade': 1.15,  # Omnichannel growth
            'Wholesale Trade': 1.12,  # Supply chain optimization
            'Construction': 1.18,  # Green building
            'Agriculture': 1.10,  # Sustainability focus
            'Other': 1.12
        }
    },
    2024: {
        'overall_growth': 0.10,  # Current year baseline
        'sector_multipliers': {
            'Manufacturing': 1.00,  # Baseline
            'Finance, Insurance & Real Estate': 1.00,
            'Services': 1.00,
            'Transportation and Public Utilities': 1.00,
            'Mining': 1.00,
            'Retail Trade': 1.00,
            'Wholesale Trade': 1.00,
            'Construction': 1.00,
            'Agriculture': 1.00,
            'Other': 1.00
        }
    }
}

def get_company_sector(adsh):
    """Get the sector for a company."""
    cursor.execute("SELECT sector FROM balance_sheet WHERE adsh = ? LIMIT 1", (adsh,))
    result = cursor.fetchone()
    return result[0] if result else 'Other'

def generate_historical_value(base_value, year, sector, volatility=0.15):
    """Generate historical value based on economic factors and trends."""
    if year == 2024:
        return base_value
    
    # Get economic factors for the year
    factors = economic_factors[year]
    sector_multiplier = factors['sector_multipliers'].get(sector, 1.0)
    overall_growth = factors['overall_growth']
    
    # Calculate compound growth from 2024 backwards
    years_from_2024 = 2024 - year
    growth_factor = (1 + overall_growth) ** years_from_2024
    sector_factor = sector_multiplier ** (1/years_from_2024) if years_from_2024 > 0 else 1.0
    
    # Add some randomness for realism
    random_factor = np.random.normal(1.0, volatility)
    
    historical_value = base_value * growth_factor * sector_factor * random_factor
    
    # Ensure non-negative values
    return max(0, historical_value)

def generate_historical_financials():
    """Generate historical financial data for 2020-2024."""
    logger.info("Starting historical financial data generation")
    
    # Get all companies with their 2024 data
    companies_query = """
    SELECT DISTINCT adsh FROM balance_sheet WHERE year = 2024
    """
    companies_df = pd.read_sql(companies_query, conn)
    logger.info(f"Processing {len(companies_df)} companies")
    
    # Generate historical data for each table
    tables_to_process = [
        ('balance_sheet', ['assets', 'liabilities', 'equity']),
        ('income_statement', ['revenue', 'expenses', 'net_income']),
        ('cash_flow', ['operating_cash', 'investing_cash', 'financing_cash'])
    ]
    
    for table_name, columns in tables_to_process:
        logger.info(f"Processing {table_name}")
        
        # Get 2024 data for this table
        current_data_query = f"""
        SELECT adsh, {', '.join(columns)} FROM {table_name} WHERE year = 2024
        """
        current_data = pd.read_sql(current_data_query, conn)
        
        historical_data = []
        
        for _, company in tqdm(current_data.iterrows(), total=len(current_data), desc=f"Generating {table_name}"):
            adsh = company['adsh']
            sector = get_company_sector(adsh)
            
            # Generate data for each year 2020-2023
            for year in range(2020, 2024):
                row_data = {'adsh': adsh, 'year': year}
                
                for col in columns:
                    base_value = company[col]
                    historical_value = generate_historical_value(base_value, year, sector)
                    row_data[col] = historical_value
                
                # Add sector (same for all years)
                row_data['sector'] = sector
                historical_data.append(row_data)
        
        # Insert historical data
        if historical_data:
            df = pd.DataFrame(historical_data)
            df.to_sql(table_name, conn, if_exists='append', index=False)
            conn.commit()
            logger.info(f"Inserted {len(df)} historical records for {table_name}")
    
    # Generate historical business activities
    logger.info("Processing business_activity table")
    
    # Get 2024 business activity data
    ba_query = "SELECT adsh, tag, unit, value FROM business_activity WHERE year = 2024"
    ba_data = pd.read_sql(ba_query, conn)
    
    historical_ba_data = []
    
    for _, activity in tqdm(ba_data.iterrows(), total=len(ba_data), desc="Generating business activities"):
        adsh = activity['adsh']
        tag = activity['tag']
        unit = activity['unit']
        base_value = activity['value']
        sector = get_company_sector(adsh)
        
        # Generate data for each year 2020-2023
        for year in range(2020, 2024):
            historical_value = generate_historical_value(base_value, year, sector, volatility=0.20)
            
            historical_ba_data.append({
                'adsh': adsh,
                'year': year,
                'tag': tag,
                'unit': unit,
                'value': historical_value
            })
    
    # Insert historical business activity data
    if historical_ba_data:
        df_ba = pd.DataFrame(historical_ba_data)
        df_ba.to_sql('business_activity', conn, if_exists='append', index=False)
        conn.commit()
        logger.info(f"Inserted {len(df_ba)} historical business activity records")
    
    # Validation
    logger.info("Validation results:")
    for table_name, _ in tables_to_process:
        result = pd.read_sql(f"SELECT year, COUNT(*) as count FROM {table_name} GROUP BY year ORDER BY year", conn)
        logger.info(f"{table_name}:")
        logger.info(result.to_string(index=False))
    
    # Business activity validation
    ba_result = pd.read_sql("SELECT year, COUNT(*) as count FROM business_activity GROUP BY year ORDER BY year", conn)
    logger.info("business_activity:")
    logger.info(ba_result.to_string(index=False))

if __name__ == '__main__':
    try:
        generate_historical_financials()
    except Exception as e:
        logger.error(f"Error generating historical financials: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed")
