import sqlite3
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Connect to the database
conn = sqlite3.connect('sec_financials.db')
cursor = conn.cursor()

def generate_simple_lending_data():
    """Generate simplified lending data with enhanced schema."""
    logger.info("Starting simplified lending data generation")
    
    # Clear existing data
    cursor.execute("DELETE FROM lending_data")
    conn.commit()
    
    # Create enhanced lending_data table
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
    
    # Get companies with financial data
    companies_query = """
    SELECT DISTINCT s.adsh, s.sic
    FROM submissions s
    WHERE s.sic IS NOT NULL
    """
    companies_df = pd.read_sql(companies_query, conn)
    logger.info(f"Processing {len(companies_df)} companies")
    
    lending_data = []
    
    for _, company in companies_df.iterrows():
        adsh = company['adsh']
        sic = company['sic']
        
        try:
            # Get financial data
            financial_query = """
            SELECT i.revenue, i.expenses, i.net_income, b.assets, b.liabilities, b.equity
            FROM income_statement i
            LEFT JOIN balance_sheet b ON i.adsh = b.adsh AND i.year = b.year
            WHERE i.adsh = ? AND i.year = 2024
            """
            financial_data = pd.read_sql(financial_query, conn, params=(adsh,))
            
            if financial_data.empty:
                continue
            
            # Extract data
            revenue = financial_data.iloc[0]['revenue']
            expenses = financial_data.iloc[0]['expenses']
            net_income = financial_data.iloc[0]['net_income']
            assets = financial_data.iloc[0]['assets']
            liabilities = financial_data.iloc[0]['liabilities']
            equity = financial_data.iloc[0]['equity']
            
            # Skip if no valid data
            if pd.isna(revenue) or revenue <= 0:
                continue
            
            # Calculate financial ratios
            current_assets = assets * 0.4
            current_liabilities = liabilities * 0.3
            current_ratio = current_assets / current_liabilities if current_liabilities > 0 else 1.0
            
            debt_to_equity = liabilities / equity if equity > 0 else 1.0
            
            operating_income = revenue - expenses
            interest_expense = liabilities * 0.05
            interest_coverage = operating_income / interest_expense if interest_expense > 0 else 1.0
            
            net_profit_margin = net_income / revenue if revenue > 0 else 0.0
            roa = net_income / assets if assets > 0 else 0.0
            roe = net_income / equity if equity > 0 else 0.0
            asset_turnover = revenue / assets if assets > 0 else 0.0
            
            # Industry risk assessment (simplified)
            industry_risk_scores = {
                4911: 2, 4922: 2, 6021: 3, 6022: 3, 2834: 4, 2836: 4,
                7372: 5, 5411: 5, 5812: 5, 4512: 7, 4213: 7, 1311: 8,
                1000: 8, 1531: 9
            }
            industry_risk_score = industry_risk_scores.get(sic, 6)
            
            industry_names = {
                4911: 'Electric Utilities', 4922: 'Natural Gas', 6021: 'National Banks',
                6022: 'State Banks', 2834: 'Pharmaceuticals', 2836: 'Biotech',
                7372: 'Software', 5411: 'Grocery Stores', 5812: 'Restaurants',
                4512: 'Airlines', 4213: 'Trucking', 1311: 'Oil & Gas',
                1000: 'Metal Mining', 1531: 'Construction'
            }
            industry_name = industry_names.get(sic, 'Other')
            
            # Credit rating based on ratios
            if current_ratio > 2.0 and debt_to_equity < 0.5 and interest_coverage > 5.0:
                credit_rating = 'AAA'
            elif current_ratio > 1.5 and debt_to_equity < 1.0 and interest_coverage > 3.0:
                credit_rating = 'AA'
            elif current_ratio > 1.2 and debt_to_equity < 1.5 and interest_coverage > 2.0:
                credit_rating = 'A'
            elif current_ratio > 1.0 and debt_to_equity < 2.0 and interest_coverage > 1.5:
                credit_rating = 'BBB'
            elif current_ratio > 0.8 and debt_to_equity < 3.0 and interest_coverage > 1.0:
                credit_rating = 'BB'
            else:
                credit_rating = 'B'
            
            # Adjust for industry risk
            if industry_risk_score >= 8:
                if credit_rating in ['AAA', 'AA', 'A']:
                    credit_rating = 'BBB'
                elif credit_rating == 'BBB':
                    credit_rating = 'BB'
            
            # Calculate credit limit (simplified)
            base_limit = revenue * 0.15  # 15% of revenue
            risk_adjustment = 1.0 - (industry_risk_score - 1) * 0.05
            credit_limit = base_limit * risk_adjustment
            
            # Generate loan portfolio
            num_loans = max(1, int(np.log10(revenue) * 2) + np.random.poisson(2))
            num_loans = min(num_loans, 20)
            
            # Loan sizes (simplified)
            loan_sizes = []
            remaining = credit_limit
            for i in range(num_loans):
                if remaining <= 0:
                    break
                size = remaining / (num_loans - i) * np.random.uniform(0.5, 1.5)
                size = min(size, remaining)
                loan_sizes.append(size)
                remaining -= size
            
            total_loans = sum(loan_sizes)
            
            # Interest rates based on credit rating
            base_rates = {'AAA': 0.02, 'AA': 0.025, 'A': 0.03, 'BBB': 0.04, 'BB': 0.06, 'B': 0.08}
            avg_interest_rate = base_rates.get(credit_rating, 0.08)
            
            # Maturities
            weighted_avg_maturity = np.random.uniform(3, 8)
            
            # Risk metrics
            default_probability = {'AAA': 0.001, 'AA': 0.002, 'A': 0.005, 'BBB': 0.01, 'BB': 0.02, 'B': 0.05}.get(credit_rating, 0.05)
            npl_rate = default_probability * 2 + np.random.uniform(0, 0.02)
            collateral_coverage = 1.0 + (industry_risk_score - 1) * 0.1
            
            # Total exposure
            total_exposure = credit_limit * 1.2
            
            lending_data.append({
                'adsh': adsh,
                'company_name': f"COMPANY_{adsh[:8]}",
                'total_loans_outstanding_usd': total_loans,
                'number_of_loans': num_loans,
                'avg_interest_rate_pct': avg_interest_rate * 100,
                'weighted_avg_maturity_years': weighted_avg_maturity,
                'non_performing_loans_pct': npl_rate * 100,
                'total_exposure_usd': total_exposure,
                'collateral_coverage_ratio': collateral_coverage,
                'credit_rating': credit_rating,
                'default_probability_pct': default_probability * 100,
                'current_ratio': current_ratio,
                'debt_to_equity_ratio': debt_to_equity,
                'interest_coverage_ratio': interest_coverage,
                'net_profit_margin_pct': net_profit_margin * 100,
                'roa_pct': roa * 100,
                'roe_pct': roe * 100,
                'asset_turnover_ratio': asset_turnover,
                'industry_risk_score': industry_risk_score,
                'industry_name': industry_name
            })
            
        except Exception as e:
            logger.error(f"Error processing company {adsh}: {e}")
            continue
    
    # Insert data
    df = pd.DataFrame(lending_data)
    df.to_sql('lending_data', conn, if_exists='append', index=False)
    conn.commit()
    
    logger.info(f"Successfully inserted {len(df)} lending records")
    
    # Summary
    result = pd.read_sql("SELECT COUNT(*) as total_records FROM lending_data", conn)
    logger.info(f"Total lending records: {result.iloc[0, 0]}")
    
    # Credit rating distribution
    rating_dist = pd.read_sql("""
        SELECT credit_rating, COUNT(*) as count, AVG(total_loans_outstanding_usd) as avg_loans
        FROM lending_data
        GROUP BY credit_rating
        ORDER BY count DESC
    """, conn)
    logger.info("Credit rating distribution:")
    logger.info(rating_dist.to_string(index=False))

if __name__ == '__main__':
    try:
        generate_simple_lending_data()
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed")
