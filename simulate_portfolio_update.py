
import pandas as pd
import sqlite3
import numpy as np
import time
from datetime import datetime
import logging

# Setup logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Connect to database
conn = sqlite3.connect('sec_financials.db')
logger.info("Database connection opened")

# Create financed_emissions table if not exists
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS financed_emissions (
        adsh TEXT,
        company_name TEXT,
        sector TEXT,
        baseline_finance_total_co2e FLOAT,
        current_finance_total_co2e FLOAT,
        bank_share_pct FLOAT,
        total_debt_usd FLOAT,
        delta_co2e FLOAT,
        update_timestamp TEXT,
        PRIMARY KEY (adsh, update_timestamp)
    )
''')
conn.commit()

# Sector-specific emissions factors (tonnes CO2e per $ exposure)
sector_factors = {
    'Energy': 0.002, 'Construction': 0.0015, 'Food': 0.001, 'Beverages': 0.0012,
    'Machinery': 0.0011, 'Tech': 0.0008, 'Transportation': 0.0018, 'Other': 0.001
}

def update_portfolio():
    logger.info("Starting portfolio update cycle")
    try:
        submissions = pd.read_sql("SELECT adsh, name, sic FROM submissions", conn)
        lending = pd.read_sql("SELECT adsh, total_exposure_usd, credit_rating FROM lending_data", conn)
        emissions = pd.read_sql("SELECT adsh, scope1_co2e, scope2_co2e, scope3_co2e, credits_purchased_tonnes FROM emissions_estimates", conn)
        balance_sheet = pd.read_sql("SELECT adsh, Liabilities FROM balance_sheet", conn)  # Use Liabilities as debt metric
        
        sic_to_sector = {1311: 'Energy', 1623: 'Construction', 2000: 'Food', 2080: 'Beverages', 3531: 'Machinery', 3826: 'Tech', 7370: 'Tech', 4011: 'Transportation'}
        submissions['sector'] = submissions['sic'].map(sic_to_sector).fillna('Other')
        
        num_updates = max(1, int(len(lending) * 0.1))
        update_indices = np.random.choice(len(lending), num_updates, replace=False)
        total_sold_tonnes = 0
        
        for idx in update_indices:
            adsh = lending.iloc[idx]['adsh']
            company_name = submissions[submissions['adsh'] == adsh]['name'].iloc[0]
            old_exposure = lending.iloc[idx]['total_exposure_usd']
            change_pct = np.random.uniform(-0.1, 0.1)
            new_exposure = max(0, old_exposure * (1 + change_pct))
            logger.info(f"Processing {company_name} ({adsh}): Exposure {old_exposure:,.2f} -> {new_exposure:,.2f} ({change_pct:.2%})")
            
            # Update lending_data
            cursor = conn.cursor()
            cursor.execute("UPDATE lending_data SET total_exposure_usd = ? WHERE adsh = ?", (new_exposure, adsh))
            
            # Update emissions
            sector = submissions[submissions['adsh'] == adsh]['sector'].iloc[0]
            factor = sector_factors.get(sector, 0.001)
            new_total_co2e = max(0, new_exposure * factor)
            cursor.execute("UPDATE emissions_estimates SET total_co2e = ? WHERE adsh = ?", (new_total_co2e, adsh))
            logger.info(f"Updated emissions for {company_name}: {new_total_co2e:,.2f} tonnes")
            
            # PCAF Financed Emissions Calculation
            company_debt = balance_sheet[balance_sheet['adsh'] == adsh]['Liabilities'].iloc[0] if not balance_sheet[balance_sheet['adsh'] == adsh].empty and 'Liabilities' in balance_sheet.columns else new_exposure * 2  # Fallback dummy
            bank_share = new_exposure / company_debt if company_debt > 0 else 0
            financed_scope1 = bank_share * emissions[emissions['adsh'] == adsh]['scope1_co2e'].iloc[0] if not emissions[emissions['adsh'] == adsh].empty else 0
            financed_scope2 = bank_share * emissions[emissions['adsh'] == adsh]['scope2_co2e'].iloc[0] if not emissions[emissions['adsh'] == adsh].empty else 0
            financed_scope3 = bank_share * emissions[emissions['adsh'] == adsh]['scope3_co2e'].iloc[0] if not emissions[emissions['adsh'] == adsh].empty else 0
            financed_total = financed_scope1 + financed_scope2 + financed_scope3
            
            # Get baseline for comparison (previous value or 0)
            baseline_total = 0  # For now, we'll use 0 as baseline
            delta_co2e = financed_total - baseline_total
            
            update_timestamp = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO financed_emissions (adsh, company_name, sector, baseline_finance_total_co2e, current_finance_total_co2e, bank_share_pct, total_debt_usd, delta_co2e, update_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (adsh, company_name, sector, baseline_total, financed_total, bank_share * 100, company_debt, delta_co2e, update_timestamp))
            logger.info(f"Financed emissions for {company_name}: {financed_total:,.2f} tonnes (bank share {bank_share*100:.2f}%)")
            
            # Check for surplus credits
            credits_purchased = emissions[emissions['adsh'] == adsh]['credits_purchased_tonnes'].iloc[0] if not emissions[emissions['adsh'] == adsh].empty else 0
            target_offset = new_total_co2e * 0.8
            surplus = credits_purchased - target_offset
            logger.info(f"Surplus check for {company_name}: credits_purchased={credits_purchased:,.2f}, target_offset={target_offset:,.2f}, surplus={surplus:,.2f}")
            if surplus > 0:
                market_price = 25.0
                surplus_trade_price = surplus * market_price
                trade_timestamp = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO carbon_trades (adsh, company_name, trade_timestamp, credits_traded_tonnes, trade_price_usd)
                    VALUES (?, ?, ?, ?, ?)
                """, (adsh, company_name, trade_timestamp, -surplus, surplus_trade_price))
                total_sold_tonnes += surplus
                logger.info(f"Surplus sell for {company_name}: {-surplus:,.2f} tonnes at ${surplus_trade_price:,.2f}")
            
            conn.commit()
        
        logger.info(f"Portfolio updated: {num_updates} companies changed, {total_sold_tonnes:,.2f} tonnes sold")
    
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")

# Run simulation loop
try:
    while True:
        update_portfolio()
        time.sleep(5)
except KeyboardInterrupt:
    logger.info("Simulation stopped")
finally:
    conn.close()
    logger.info("Database connection closed")
