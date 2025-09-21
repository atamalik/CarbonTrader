
import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect('sec_financials.db')

# List all tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables in database:", tables['name'].tolist())

# Check table structures
for table in ['submissions', 'lending_data', 'emissions_estimates', 'carbon_trades', 'financed_emissions', 'balance_sheet']:
    try:
        structure = pd.read_sql(f"PRAGMA table_info({table});", conn)
        print(f"\nStructure of {table}:")
        print(structure[['name', 'type', 'notnull', 'pk']].to_string(index=False))
    except Exception as e:
        print(f"\nTable {table} not found or error: {e}")

# Fix emissions_estimates: Add total_co2e if missing
try:
    emissions = pd.read_sql("SELECT adsh, scope1_co2e, scope2_co2e, scope3_co2e, credits_purchased_tonnes FROM emissions_estimates", conn)
    if 'total_co2e' not in emissions.columns:
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE emissions_estimates ADD COLUMN total_co2e REAL")
        emissions['total_co2e'] = emissions['scope1_co2e'].fillna(0) + emissions['scope2_co2e'].fillna(0) + emissions['scope3_co2e'].fillna(0)
        emissions[['adsh', 'total_co2e']].to_sql('emissions_estimates', conn, if_exists='replace', index=False)
        print("Added and populated total_co2e in emissions_estimates")
except Exception as e:
    print(f"Error fixing emissions_estimates: {e}")

# Create/verify financed_emissions table
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS financed_emissions (
        adsh TEXT,
        company_name TEXT,
        sector TEXT,
        financed_scope1_co2e REAL,
        financed_scope2_co2e REAL,
        financed_scope3_co2e REAL,
        financed_total_co2e REAL,
        bank_share_pct REAL,
        update_timestamp TEXT,
        PRIMARY KEY (adsh, update_timestamp)
    )
''')
conn.commit()
print("Verified/created financed_emissions table")

# Populate dummy company_total_debt_usd in balance_sheet if missing
try:
    balance_structure = pd.read_sql("PRAGMA table_info(balance_sheet);", conn)
    if 'company_total_debt_usd' not in balance_structure['name'].values:
        cursor.execute("ALTER TABLE balance_sheet ADD COLUMN company_total_debt_usd REAL")
        lending = pd.read_sql("SELECT adsh, total_exposure_usd FROM lending_data", conn)
        balance = pd.read_sql("SELECT adsh, year FROM balance_sheet", conn)
        balance = balance.merge(lending, on='adsh', how='left')
        balance['company_total_debt_usd'] = balance['total_exposure_usd'] * np.random.uniform(2.0, 5.0)
        balance[['adsh', 'year', 'company_total_debt_usd']].to_sql('balance_sheet', conn, if_exists='append', index=False)
        print("Populated company_total_debt_usd in balance_sheet")
except Exception as e:
    print(f"Error populating balance_sheet: {e}")

conn.close()
