import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import lognorm, gamma, norm, weibull_min, truncnorm, expon
import requests
from bs4 import BeautifulSoup
import logging
import json
import os
import time
from tqdm import tqdm
import uuid

# Configuration
SECTOR = 'Other'
USE_WEB_SEARCH = False  # Toggle for live benchmarks (slower) vs pre-populated (faster)
BENCHMARK_CACHE_FILE = 'other_benchmark_cache.json'
LOG_FILE = 'other_financials.log'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Connect to the database
conn = sqlite3.connect('sec_financials.db')
cursor = conn.cursor()

# Pre-populated benchmark means (in USD) for Other sector
# Other sector characteristics:
# - Diverse mix of companies that don't fit standard classifications
# - Generally smaller to medium-sized companies
# - Mixed business models and operational characteristics
# - Moderate capital requirements
# - Variable profitability patterns
# - Lower R&D investment compared to specialized sectors
pre_populated_benchmarks = {
    f"average revenue for other sector companies 2024": 8e8,      # Moderate revenue for diverse companies
    f"average expenses for other sector companies 2024": 6e8,     # Moderate operational costs
    f"average net_income for other sector companies 2024": 2e8,   # Moderate margins
    f"average assets for other sector companies 2024": 1.5e9,     # Moderate asset requirements
    f"average liabilities for other sector companies 2024": 8e8,  # Moderate debt levels
    f"average equity for other sector companies 2024": 7e8,       # Moderate equity levels
    f"average operating_cash for other sector companies 2024": 2.5e8, # Moderate cash flow
    f"average investing_cash for other sector companies 2024": -1.5e8, # Moderate investment needs
    f"average financing_cash for other sector companies 2024": 5e7,  # Lower financing needs
    f"average r_and_d_expense for other sector companies 2024": 3e7,     # Lower R&D investment
    f"average operating_income for other sector companies 2024": 2.5e8  # Operating income before interest/taxes
}

# Distribution types for Other sector components
# Other sector has diverse characteristics:
# - Mixed business models create varied financial patterns
# - Generally smaller scale operations
# - Moderate capital requirements
# - Variable profitability patterns
# - Lower technology investment
sector_distributions = {
    'Other': {
        'revenue': 'gamma',            # Right-skewed for varying business sizes
        'expenses': 'gamma',           # Right-skewed for operational cost variability
        'net_income': 'truncnorm',     # Truncated to reflect moderate margins
        'assets': 'gamma',             # Right-skewed for asset-light businesses
        'liabilities': 'gamma',        # Moderate debt levels create gamma distribution
        'equity': 'norm',              # More stable equity structure
        'operating_cash': 'truncnorm', # Truncated for moderate cash flow patterns
        'investing_cash': 'expon',     # Exponential for moderate investment needs
        'financing_cash': 'norm',      # Normal for lower financing needs
        'r_and_d_expense': 'weibull',      # Skewed for occasional innovation investments
        'operating_income': 'gamma'    # Gamma for operating income variability
    }
}

# Load or initialize benchmark cache
benchmark_cache = {}
if os.path.exists(BENCHMARK_CACHE_FILE):
    with open(BENCHMARK_CACHE_FILE, 'r') as f:
        benchmark_cache = json.load(f)

def fetch_benchmark(query, max_retries=3):
    """Fetch benchmark data from cache or web (if enabled)."""
    if query in benchmark_cache:
        logger.info(f"Retrieved cached benchmark for {query}: {benchmark_cache[query]}")
        return benchmark_cache[query]
    
    if not USE_WEB_SEARCH:
        return pre_populated_benchmarks.get(query, None)
    
    for attempt in range(max_retries):
        try:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('div', class_='BNeawe')
            for result in results:
                text = result.text.lower()
                if 'billion' in text or 'million' in text or 'trillion' in text:
                    try:
                        value_str = ''.join([c for c in text if c.isdigit() or c == '.'])
                        value = float(value_str)
                        if 'trillion' in text:
                            value *= 1e12
                        elif 'billion' in text:
                            value *= 1e9
                        elif 'million' in text:
                            value *= 1e6
                        benchmark_cache[query] = value
                        with open(BENCHMARK_CACHE_FILE, 'w') as f:
                            json.dump(benchmark_cache, f)
                        logger.info(f"Fetched benchmark for {query}: {value}")
                        return value
                    except ValueError:
                        continue
            logger.warning(f"Failed to parse benchmark for {query} on attempt {attempt + 1}")
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Error fetching benchmark for {query} on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return pre_populated_benchmarks.get(query, None)  # Fallback to pre-populated

def get_distribution_params(sector, component):
    """Get mean and SD for a financial component in the sector."""
    # Create normalized query that matches the dictionary keys
    normalized_sector = sector.lower().replace(',', '').replace('&', '').replace('  ', ' ').strip()
    query = f"average {component} for {normalized_sector} sector companies 2024"
    mean = fetch_benchmark(query)
    if mean is None:
        # Try to find a matching key in pre_populated_benchmarks
        for key in pre_populated_benchmarks.keys():
            if component in key and normalized_sector in key:
                mean = pre_populated_benchmarks[key]
                break
        if mean is None:
            logger.warning(f"No benchmark for {query}, using default mean")
            mean = 1e6  # Default fallback
    sd = mean * 0.30  # Other sector has moderate variability (30%) due to diversity
    return mean, sd

def generate_value(sector, component, mean, sd):
    """Generate a financial value using sector-specific distribution."""
    dist_type = sector_distributions.get(sector, {}).get(component, 'norm')
    mean = max(mean, 1e-6)
    sd = max(sd, 1e-6)
    
    try:
        if dist_type == 'lognorm':
            sigma = np.sqrt(np.log(1 + (sd / mean)**2))
            mu = np.log(mean) - 0.5 * sigma**2
            value = lognorm(s=sigma, scale=np.exp(mu)).rvs()
        elif dist_type == 'gamma':
            shape = (mean / sd)**2
            scale = sd**2 / mean
            value = gamma(shape, scale=scale).rvs()
        elif dist_type == 'norm':
            value = norm(loc=mean, scale=sd).rvs()
        elif dist_type == 'weibull':
            shape = 1.5
            scale = mean / np.gamma(1 + 1/shape)
            value = weibull_min(shape, scale=scale).rvs()
        elif dist_type == 'truncnorm':
            a = -mean / (2 * sd)
            b = np.inf
            value = truncnorm(a=a, b=b, loc=mean, scale=sd).rvs()
        elif dist_type == 'expon':
            # Handle negative means for outflows (e.g., investing_cash)
            if mean < 0:
                value = -expon(loc=0, scale=abs(mean)).rvs()
            else:
                value = expon(loc=0, scale=mean).rvs()
        else:
            value = norm(loc=mean, scale=sd).rvs()
        return max(0, value)  # Ensure non-negative
    except Exception as e:
        logger.error(f"Error generating value for {component} in {sector}: {e}")
        return mean  # Fallback to mean

def check_table_schema(table_name, expected_columns):
    """Ensure table exists with correct schema, create if missing."""
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [info[1] for info in cursor.fetchall()]
        if set(expected_columns).issubset(set(existing_columns)):
            logger.info(f"Table {table_name} has correct schema")
            return True
        logger.warning(f"Table {table_name} missing columns, recreating")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        if table_name == 'balance_sheet':
            cursor.execute("""
                CREATE TABLE balance_sheet (
                    adsh TEXT, year INTEGER, assets REAL, liabilities REAL, equity REAL, sector TEXT
                )
            """)
        elif table_name == 'income_statement':
            cursor.execute("""
                CREATE TABLE income_statement (
                    adsh TEXT, year INTEGER, revenue REAL, expenses REAL, net_income REAL, sector TEXT
                )
            """)
        elif table_name == 'cash_flow':
            cursor.execute("""
                CREATE TABLE cash_flow (
                    adsh TEXT, year INTEGER, operating_cash REAL, investing_cash REAL, financing_cash REAL, sector TEXT
                )
            """)
        elif table_name == 'business_activity':
            cursor.execute("""
                CREATE TABLE business_activity (
                    adsh TEXT, year INTEGER, r_and_d_expense REAL, operating_income REAL, sector TEXT
                )
            """)
        conn.commit()
        logger.info(f"Created table {table_name} with correct schema")
        return True
    except Exception as e:
        logger.error(f"Error checking/creating schema for {table_name}: {e}")
        return False

def is_sector_processed(table_name, sector):
    """Check if sector data already exists in the table."""
    try:
        count = pd.read_sql(f"SELECT COUNT(*) FROM {table_name} WHERE sector = '{sector}'", conn).iloc[0, 0]
        return count > 0
    except Exception as e:
        logger.error(f"Error checking if {sector} is processed in {table_name}: {e}")
        return False

def update_table(sector, table_name, columns):
    """Update a single table for the given sector."""
    if is_sector_processed(table_name, sector):
        logger.info(f"Skipping {table_name} for {sector} (already populated)")
        return
    
    # Ensure table schema
    if not check_table_schema(table_name, ['adsh', 'year'] + columns + ['sector']):
        logger.error(f"Failed to validate schema for {table_name}, skipping")
        return
    
    # Fetch Other sector companies (companies already marked as Other in balance_sheet)
    # This includes companies that don't fit into the standard sector classifications
    df_sub = pd.read_sql("""
        SELECT adsh, sector FROM balance_sheet 
        WHERE sector = 'Other'
    """, conn)
    
    if df_sub.empty:
        logger.warning(f"No companies found for {sector} in submissions")
        return
    
    logger.info(f"Updating {table_name} for {sector} with {len(df_sub)} companies")
    data = []
    for adsh in tqdm(df_sub['adsh'], desc=f"Processing {table_name} for {sector}"):
        values = {}
        for col in columns:
            mean, sd = get_distribution_params(sector, col)
            if mean is None:
                logger.warning(f"Using fallback mean for {col} in {sector}")
                mean = 1e6
                sd = mean * 0.30
            values[col] = generate_value(sector, col, mean, sd)
        data.append([adsh, 2024] + [values[col] for col in columns] + [sector])
    
    df = pd.DataFrame(data, columns=['adsh', 'year'] + columns + ['sector'])
    try:
        df.to_sql(table_name, conn, if_exists='append', index=False)
        conn.commit()
        logger.info(f"Successfully updated {table_name} for {sector}")
        
        # Validate data
        result = pd.read_sql(f"SELECT * FROM {table_name} WHERE sector = '{sector}'", conn)
        logger.info(f"Validation: {len(result)} rows inserted for {sector} in {table_name}")
        for col in columns:
            avg = result[col].mean()
            logger.info(f"Average {col} for {sector}: {avg:.2e}")
    except Exception as e:
        logger.error(f"Error updating {table_name} for {sector}: {e}")

def main():
    """Generate financial data for Other sector."""
    logger.info(f"Starting financial data generation for {SECTOR}")
    
    # Ensure submissions table has sector column
    try:
        cursor.execute("ALTER TABLE submissions ADD COLUMN sector TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        logger.info("Submissions table already has sector column")
    
    # Update submissions with sector for Other (companies not in balance_sheet)
    cursor.execute("""
        UPDATE submissions
        SET sector = 'Other'
        WHERE adsh NOT IN (SELECT DISTINCT adsh FROM balance_sheet)
    """)
    conn.commit()
    
    # Process each table
    update_table(SECTOR, 'balance_sheet', ['assets', 'liabilities', 'equity'])
    update_table(SECTOR, 'income_statement', ['revenue', 'expenses', 'net_income'])
    update_table(SECTOR, 'cash_flow', ['operating_cash', 'investing_cash', 'financing_cash'])
    update_table(SECTOR, 'business_activity', ['r_and_d_expense', 'operating_income'])
    
    logger.info(f"Completed financial data generation for {SECTOR}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed")
