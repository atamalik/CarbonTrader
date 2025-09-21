import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, beta, gamma
import logging
import json
import os
from tqdm import tqdm
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/sophisticated_lending.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Connect to the database
conn = sqlite3.connect('sec_financials.db')
cursor = conn.cursor()

# Industry Risk Assessment (based on Basel III and regional banking standards)
# Risk scores: 1 (Lowest Risk) to 10 (Highest Risk)
industry_risk_assessment = {
    # Low Risk Industries (Score 1-3)
    4911: {'risk_score': 2, 'name': 'Electric Utilities', 'volatility': 0.15, 'cyclical': False, 'regulatory': True},
    4922: {'risk_score': 2, 'name': 'Natural Gas Transmission', 'volatility': 0.18, 'cyclical': False, 'regulatory': True},
    6021: {'risk_score': 3, 'name': 'National Commercial Banks', 'volatility': 0.20, 'cyclical': True, 'regulatory': True},
    6022: {'risk_score': 3, 'name': 'State Commercial Banks', 'volatility': 0.22, 'cyclical': True, 'regulatory': True},
    
    # Medium Risk Industries (Score 4-6)
    2834: {'risk_score': 4, 'name': 'Pharmaceutical Preparations', 'volatility': 0.25, 'cyclical': False, 'regulatory': True},
    2836: {'risk_score': 4, 'name': 'Biological Products', 'volatility': 0.28, 'cyclical': False, 'regulatory': True},
    7372: {'risk_score': 5, 'name': 'Prepackaged Software', 'volatility': 0.30, 'cyclical': True, 'regulatory': False},
    5411: {'risk_score': 5, 'name': 'Grocery Stores', 'volatility': 0.20, 'cyclical': False, 'regulatory': False},
    5812: {'risk_score': 5, 'name': 'Eating Places', 'volatility': 0.25, 'cyclical': True, 'regulatory': False},
    
    # High Risk Industries (Score 7-10)
    4512: {'risk_score': 7, 'name': 'Air Transportation', 'volatility': 0.40, 'cyclical': True, 'regulatory': True},
    4213: {'risk_score': 7, 'name': 'Trucking', 'volatility': 0.35, 'cyclical': True, 'regulatory': False},
    1311: {'risk_score': 8, 'name': 'Crude Petroleum and Natural Gas', 'volatility': 0.45, 'cyclical': True, 'regulatory': True},
    1000: {'risk_score': 8, 'name': 'Metal Mining', 'volatility': 0.50, 'cyclical': True, 'regulatory': True},
    1531: {'risk_score': 9, 'name': 'Operative Builders', 'volatility': 0.35, 'cyclical': True, 'regulatory': False},
}

# Default risk assessment for unmapped industries
default_industry_risk = {'risk_score': 6, 'name': 'Other (Unmapped SIC)', 'volatility': 0.30, 'cyclical': True, 'regulatory': False}

# Credit Rating Mapping (based on risk scores and financial ratios)
def get_credit_rating(risk_score, financial_ratios):
    """Determine credit rating based on risk score and financial ratios."""
    # Base rating from risk score
    if risk_score <= 2:
        base_rating = 'AAA'
    elif risk_score <= 3:
        base_rating = 'AA'
    elif risk_score <= 4:
        base_rating = 'A'
    elif risk_score <= 5:
        base_rating = 'BBB'
    elif risk_score <= 6:
        base_rating = 'BB'
    elif risk_score <= 7:
        base_rating = 'B'
    elif risk_score <= 8:
        base_rating = 'CCC'
    else:
        base_rating = 'CC'
    
    # Adjust based on financial ratios
    current_ratio = financial_ratios.get('current_ratio', 1.0)
    debt_to_equity = financial_ratios.get('debt_to_equity', 1.0)
    interest_coverage = financial_ratios.get('interest_coverage', 1.0)
    
    # Rating adjustments
    if current_ratio > 2.0 and debt_to_equity < 0.5 and interest_coverage > 5.0:
        if base_rating in ['BBB', 'BB', 'B']:
            base_rating = chr(ord(base_rating[0]) - 1) + base_rating[1:]
    elif current_ratio < 1.0 or debt_to_equity > 2.0 or interest_coverage < 2.0:
        if base_rating in ['AA', 'A', 'BBB', 'BB']:
            base_rating = chr(ord(base_rating[0]) + 1) + base_rating[1:]
    
    return base_rating

def calculate_financial_ratios(income_data, balance_data):
    """Calculate key financial ratios for credit assessment."""
    ratios = {}
    
    try:
        # Liquidity Ratios
        current_assets = balance_data.get('assets', 0) * 0.4  # Assume 40% current assets
        current_liabilities = balance_data.get('liabilities', 0) * 0.3  # Assume 30% current liabilities
        ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities > 0 else 1.0
        
        # Leverage Ratios
        total_debt = balance_data.get('liabilities', 0)
        total_equity = balance_data.get('equity', 0)
        ratios['debt_to_equity'] = total_debt / total_equity if total_equity > 0 else 1.0
        ratios['debt_to_assets'] = total_debt / balance_data.get('assets', 1)
        
        # Profitability Ratios
        revenue = income_data.get('revenue', 0)
        net_income = income_data.get('net_income', 0)
        ratios['net_profit_margin'] = net_income / revenue if revenue > 0 else 0.0
        ratios['roa'] = net_income / balance_data.get('assets', 1)
        ratios['roe'] = net_income / total_equity if total_equity > 0 else 0.0
        
        # Coverage Ratios
        operating_income = income_data.get('revenue', 0) - income_data.get('expenses', 0)
        interest_expense = total_debt * 0.05  # Assume 5% interest rate
        ratios['interest_coverage'] = operating_income / interest_expense if interest_expense > 0 else 1.0
        
        # Efficiency Ratios
        ratios['asset_turnover'] = revenue / balance_data.get('assets', 1)
        
    except (ZeroDivisionError, TypeError, KeyError) as e:
        # Return default ratios if calculation fails
        ratios = {
            'current_ratio': 1.0,
            'debt_to_equity': 1.0,
            'debt_to_assets': 0.5,
            'net_profit_margin': 0.0,
            'roa': 0.0,
            'roe': 0.0,
            'interest_coverage': 1.0,
            'asset_turnover': 0.5
        }
    
    return ratios

def calculate_credit_limit(company_data, industry_risk, financial_ratios, credit_rating):
    """Calculate credit limit based on comprehensive risk assessment."""
    try:
        base_revenue = company_data.get('revenue', 0)
        base_assets = company_data.get('assets', 0)
    except (KeyError, TypeError):
        base_revenue = 0
        base_assets = 0
    
    # Base credit limit as percentage of revenue (industry standard: 10-30%)
    base_limit_pct = {
        'AAA': 0.30, 'AA': 0.25, 'A': 0.20, 'BBB': 0.15, 'BB': 0.10,
        'B': 0.08, 'CCC': 0.05, 'CC': 0.03, 'C': 0.02, 'D': 0.01
    }.get(credit_rating, 0.10)
    
    # Adjust for industry risk
    industry_multiplier = 1.0 - (industry_risk['risk_score'] - 1) * 0.05
    industry_multiplier = max(0.3, industry_multiplier)  # Minimum 30% of base limit
    
    # Adjust for financial ratios
    ratio_multiplier = 1.0
    
    # Current ratio adjustment
    if financial_ratios['current_ratio'] > 2.0:
        ratio_multiplier *= 1.2
    elif financial_ratios['current_ratio'] < 1.0:
        ratio_multiplier *= 0.7
    
    # Debt-to-equity adjustment
    if financial_ratios['debt_to_equity'] < 0.5:
        ratio_multiplier *= 1.15
    elif financial_ratios['debt_to_equity'] > 2.0:
        ratio_multiplier *= 0.8
    
    # Interest coverage adjustment
    if financial_ratios['interest_coverage'] > 5.0:
        ratio_multiplier *= 1.1
    elif financial_ratios['interest_coverage'] < 2.0:
        ratio_multiplier *= 0.9
    
    # Calculate final credit limit
    try:
        credit_limit = base_revenue * base_limit_pct * industry_multiplier * ratio_multiplier
        
        # Cap at reasonable maximum (50% of assets)
        max_limit = base_assets * 0.5
        credit_limit = min(credit_limit, max_limit)
        
        return max(credit_limit, base_revenue * 0.01)  # Minimum 1% of revenue
    except (TypeError, ValueError):
        return base_revenue * 0.1  # Default 10% of revenue

def generate_loan_portfolio(credit_limit, industry_risk, company_size):
    """Generate realistic loan portfolio based on credit limit and risk profile."""
    try:
        # Number of loans (more loans for larger companies and higher risk)
        base_loans = max(1, int(np.log10(max(company_size, 1000)) * 2))  # Ensure minimum company size
        risk_adjustment = industry_risk.get('risk_score', 6) * 0.5
        num_loans = int(base_loans + risk_adjustment + np.random.poisson(2))
        num_loans = min(num_loans, 50)  # Cap at 50 loans
    except (ValueError, TypeError):
        num_loans = 5  # Default number of loans
    
    # Loan sizes (Pareto distribution - few large, many small)
    loan_sizes = []
    try:
        # Use 60-80% of credit limit as actual loans (realistic utilization)
        utilization_rate = np.random.uniform(0.6, 0.8)
        total_loan_amount = credit_limit * utilization_rate
        remaining_amount = total_loan_amount
        
        for i in range(num_loans):
            if remaining_amount <= 0:
                break
                
            # Pareto distribution for loan sizes
            alpha = 1.5
            min_size = total_loan_amount * 0.01  # Minimum 1% of total loan amount
            max_size = min(remaining_amount, total_loan_amount * 0.3)  # Maximum 30% of total loan amount
            
            if max_size <= min_size:
                loan_size = remaining_amount
            else:
                # Generate Pareto-distributed loan size
                u = np.random.uniform(0, 1)
                loan_size = min_size * (1 - u) ** (-1/alpha)
                loan_size = min(loan_size, max_size)
            
            loan_sizes.append(loan_size)
            remaining_amount -= loan_size
    except (ValueError, TypeError):
        # Default loan sizes if generation fails
        loan_sizes = [credit_limit * 0.7 / num_loans] * num_loans
    
    return loan_sizes

def calculate_interest_rates(credit_rating, industry_risk, loan_sizes):
    """Calculate realistic interest rates based on risk profile."""
    # Base rates by credit rating (risk-free rate + spread)
    base_rates = {
        'AAA': 0.02, 'AA': 0.025, 'A': 0.03, 'BBB': 0.04, 'BB': 0.06,
        'B': 0.08, 'CCC': 0.12, 'CC': 0.15, 'C': 0.20, 'D': 0.25
    }
    
    base_rate = base_rates.get(credit_rating, 0.08)
    
    # Industry risk adjustment
    industry_spread = (industry_risk['risk_score'] - 1) * 0.005
    
    # Loan size adjustment (larger loans get better rates)
    rates = []
    for loan_size in loan_sizes:
        size_adjustment = -0.001 * np.log10(loan_size / 1000000)  # Better rates for larger loans
        final_rate = base_rate + industry_spread + size_adjustment + np.random.normal(0, 0.002)
        final_rate = max(0.01, min(final_rate, 0.30))  # Cap between 1% and 30%
        rates.append(final_rate)
    
    return rates

def calculate_maturities(industry_risk, loan_sizes):
    """Calculate realistic loan maturities."""
    maturities = []
    
    for loan_size in loan_sizes:
        # Base maturity by industry
        if industry_risk['cyclical']:
            base_maturity = np.random.uniform(2, 7)  # Shorter for cyclical industries
        else:
            base_maturity = np.random.uniform(3, 10)  # Longer for stable industries
        
        # Size adjustment (larger loans typically longer maturities)
        size_adjustment = np.log10(loan_size / 1000000) * 0.5
        final_maturity = base_maturity + size_adjustment + np.random.normal(0, 1)
        final_maturity = max(1, min(final_maturity, 15))  # Cap between 1 and 15 years
        
        maturities.append(final_maturity)
    
    return maturities

def calculate_risk_metrics(credit_rating, industry_risk, financial_ratios, loan_sizes):
    """Calculate additional risk metrics."""
    # Default probability based on credit rating and industry
    base_pd = {
        'AAA': 0.001, 'AA': 0.002, 'A': 0.005, 'BBB': 0.01, 'BB': 0.02,
        'B': 0.05, 'CCC': 0.10, 'CC': 0.15, 'C': 0.25, 'D': 0.50
    }.get(credit_rating, 0.05)
    
    # Industry adjustment
    industry_pd_multiplier = 1.0 + (industry_risk['risk_score'] - 1) * 0.1
    default_probability = base_pd * industry_pd_multiplier
    
    # Non-performing loans (typically 1-5% of portfolio)
    npl_rate = min(0.05, default_probability * 2 + np.random.uniform(0, 0.02))
    
    # Collateral coverage (higher for riskier loans)
    base_coverage = 1.0 + (industry_risk['risk_score'] - 1) * 0.1
    collateral_coverage = base_coverage + np.random.normal(0, 0.1)
    collateral_coverage = max(0.5, min(collateral_coverage, 2.0))
    
    return {
        'default_probability': default_probability,
        'npl_rate': npl_rate,
        'collateral_coverage': collateral_coverage
    }

def generate_sophisticated_lending_data():
    """Generate sophisticated lending data using credit risk modeling."""
    logger.info("Starting sophisticated lending data generation")
    
    # Clear existing data
    cursor.execute("DELETE FROM lending_data")
    conn.commit()
    logger.info("Cleared existing lending data")
    
    # Get all companies with their financial data (both with and without SIC codes)
    companies_query = """
    SELECT DISTINCT s.adsh, s.sic
    FROM submissions s
    WHERE s.adsh IN (SELECT adsh FROM income_statement WHERE year = 2024)
    AND s.adsh IN (SELECT adsh FROM balance_sheet WHERE year = 2024)
    """
    companies_df = pd.read_sql(companies_query, conn)
    logger.info(f"Processing {len(companies_df)} companies (with and without SIC codes)")
    
    # Count companies with and without SIC codes
    companies_with_sic = companies_df[companies_df['sic'].notna()].shape[0]
    companies_without_sic = companies_df[companies_df['sic'].isna()].shape[0]
    logger.info(f"Companies with SIC codes: {companies_with_sic}")
    logger.info(f"Companies without SIC codes (will be categorized as 'Other'): {companies_without_sic}")
    
    lending_data = []
    
    for _, company in tqdm(companies_df.iterrows(), total=len(companies_df), desc="Generating lending data"):
        try:
            adsh = company['adsh']
            sic = company['sic']
            
            # Get industry risk assessment (handle companies without SIC codes)
            if pd.isna(sic) or sic is None:
                # Companies without SIC codes get "Other" sector with medium risk
                industry_risk = {'risk_score': 6, 'name': 'Other (No SIC)', 'volatility': 0.30, 'cyclical': True, 'regulatory': False}
            else:
                industry_risk = industry_risk_assessment.get(sic, default_industry_risk)
            
            # Get latest financial data (2024)
            financial_query = """
            SELECT i.revenue, i.expenses, i.net_income, b.assets, b.liabilities, b.equity
            FROM income_statement i
            LEFT JOIN balance_sheet b ON i.adsh = b.adsh AND i.year = b.year
            WHERE i.adsh = ? AND i.year = 2024
            """
            financial_data = pd.read_sql(financial_query, conn, params=(adsh,))
            
            if financial_data.empty:
                continue
            
            # Extract data safely with better error handling
            try:
                revenue = float(financial_data.iloc[0]['revenue']) if not pd.isna(financial_data.iloc[0]['revenue']) else 0
                expenses = float(financial_data.iloc[0]['expenses']) if not pd.isna(financial_data.iloc[0]['expenses']) else 0
                net_income = float(financial_data.iloc[0]['net_income']) if not pd.isna(financial_data.iloc[0]['net_income']) else 0
                assets = float(financial_data.iloc[0]['assets']) if not pd.isna(financial_data.iloc[0]['assets']) else 0
                liabilities = float(financial_data.iloc[0]['liabilities']) if not pd.isna(financial_data.iloc[0]['liabilities']) else 0
                equity = float(financial_data.iloc[0]['equity']) if not pd.isna(financial_data.iloc[0]['equity']) else 0
            except (ValueError, TypeError) as e:
                logger.error(f"Error extracting financial data for {adsh}: {e}")
                continue
            
            # Skip if no valid revenue data
            if revenue <= 0:
                continue
            
            # Ensure we have valid balance sheet data
            if assets <= 0:
                # Use income data to estimate balance sheet
                assets = revenue * 2.0  # Assume 2x revenue
                liabilities = assets * 0.6  # Assume 60% debt
                equity = assets - liabilities
            
            # Create data dictionaries
            income_data = {'revenue': revenue, 'expenses': expenses, 'net_income': net_income}
            balance_data = {'assets': assets, 'liabilities': liabilities, 'equity': equity}
            
            # Calculate financial ratios
            financial_ratios = calculate_financial_ratios(income_data, balance_data)
            
            # Determine credit rating
            credit_rating = get_credit_rating(industry_risk['risk_score'], financial_ratios)
            
            # Calculate credit limit
            credit_limit = calculate_credit_limit(income_data, industry_risk, financial_ratios, credit_rating)
            
            # Generate loan portfolio
            loan_sizes = generate_loan_portfolio(credit_limit, industry_risk, income_data['revenue'])
            
            if not loan_sizes:
                continue
            
            # Calculate loan terms
            interest_rates = calculate_interest_rates(credit_rating, industry_risk, loan_sizes)
            maturities = calculate_maturities(industry_risk, loan_sizes)
            
            # Calculate risk metrics
            risk_metrics = calculate_risk_metrics(credit_rating, industry_risk, financial_ratios, loan_sizes)
            
            # Calculate portfolio metrics
            total_loans = sum(loan_sizes)
            num_loans = len(loan_sizes)
            avg_interest_rate = np.average(interest_rates, weights=loan_sizes)
            weighted_avg_maturity = np.average(maturities, weights=loan_sizes)
            
            # Calculate total exposure (actual loans outstanding)
            total_exposure = total_loans  # This is the actual amount lent out
            
            # Get company name (simplified)
            company_name = f"COMPANY_{adsh[:8]}"
            
            lending_data.append({
                'adsh': adsh,
                'company_name': company_name,
                'total_loans_outstanding_usd': total_loans,
                'number_of_loans': num_loans,
                'avg_interest_rate_pct': avg_interest_rate * 100,
                'weighted_avg_maturity_years': weighted_avg_maturity,
                'non_performing_loans_pct': risk_metrics['npl_rate'] * 100,
                'total_exposure_usd': total_exposure,
                'collateral_coverage_ratio': risk_metrics['collateral_coverage'],
                'credit_rating': credit_rating,
                'default_probability_pct': risk_metrics['default_probability'] * 100,
                # Additional financial ratios
                'current_ratio': financial_ratios['current_ratio'],
                'debt_to_equity_ratio': financial_ratios['debt_to_equity'],
                'interest_coverage_ratio': financial_ratios['interest_coverage'],
                'net_profit_margin_pct': financial_ratios['net_profit_margin'] * 100,
                'roa_pct': financial_ratios['roa'] * 100,
                'roe_pct': financial_ratios['roe'] * 100,
                'asset_turnover_ratio': financial_ratios['asset_turnover'],
                'industry_risk_score': industry_risk['risk_score'],
                'industry_name': industry_risk['name']
            })
            
        except Exception as e:
            logger.error(f"Error processing company {adsh}: {e}")
            continue
    
    # Create enhanced lending_data table with additional columns
    cursor.execute("DROP TABLE IF EXISTS lending_data")
    cursor.execute("""
        CREATE TABLE lending_data (
            adsh TEXT,
            company_name TEXT,
            total_loans_outstanding_usd REAL,
            number_of_loans INTEGER,
            avg_interest_rate_pct REAL,
            weighted_avg_maturity_years REAL,
            non_performing_loans_pct REAL,
            total_exposure_usd REAL,
            collateral_coverage_ratio REAL,
            credit_rating TEXT,
            default_probability_pct REAL,
            current_ratio REAL,
            debt_to_equity_ratio REAL,
            interest_coverage_ratio REAL,
            net_profit_margin_pct REAL,
            roa_pct REAL,
            roe_pct REAL,
            asset_turnover_ratio REAL,
            industry_risk_score INTEGER,
            industry_name TEXT
        )
    """)
    conn.commit()
    
    # Insert data
    df = pd.DataFrame(lending_data)
    df.to_sql('lending_data', conn, if_exists='append', index=False)
    conn.commit()
    
    logger.info(f"Successfully inserted {len(df)} lending records")
    
    # Validation and summary
    result = pd.read_sql("SELECT COUNT(*) as total_records FROM lending_data", conn)
    logger.info(f"Total lending records: {result.iloc[0, 0]}")
    
    # Summary statistics
    summary = pd.read_sql("""
        SELECT 
            AVG(total_loans_outstanding_usd) as avg_loans,
            AVG(number_of_loans) as avg_num_loans,
            AVG(avg_interest_rate_pct) as avg_interest_rate,
            AVG(weighted_avg_maturity_years) as avg_maturity,
            AVG(non_performing_loans_pct) as avg_npl,
            AVG(default_probability_pct) as avg_default_prob,
            COUNT(CASE WHEN credit_rating IN ('AAA', 'AA', 'A') THEN 1 END) as investment_grade_count,
            COUNT(CASE WHEN credit_rating IN ('BBB', 'BB', 'B') THEN 1 END) as speculative_grade_count,
            COUNT(CASE WHEN credit_rating IN ('CCC', 'CC', 'C', 'D') THEN 1 END) as junk_grade_count
        FROM lending_data
    """, conn)
    logger.info("Summary statistics:")
    logger.info(summary.to_string(index=False))
    
    # Credit rating distribution
    rating_dist = pd.read_sql("""
        SELECT credit_rating, COUNT(*) as count, AVG(total_loans_outstanding_usd) as avg_loans
        FROM lending_data
        GROUP BY credit_rating
        ORDER BY count DESC
    """, conn)
    logger.info("Credit rating distribution:")
    logger.info(rating_dist.to_string(index=False))
    
    # Industry risk distribution
    industry_dist = pd.read_sql("""
        SELECT industry_name, industry_risk_score, COUNT(*) as count, AVG(total_loans_outstanding_usd) as avg_loans
        FROM lending_data
        GROUP BY industry_name, industry_risk_score
        ORDER BY industry_risk_score, count DESC
    """, conn)
    logger.info("Industry risk distribution:")
    logger.info(industry_dist.to_string(index=False))

if __name__ == '__main__':
    try:
        generate_sophisticated_lending_data()
    except Exception as e:
        logger.error(f"Error generating sophisticated lending data: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed")
