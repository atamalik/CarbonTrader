
import pandas as pd
import numpy as np
import sqlite3
from langgraph.graph import StateGraph, END
from typing import TypedDict
from crewai import Agent, Task, Crew
from datetime import datetime
import os
import re

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Note: Set LANGCHAIN_API_KEY environment variable if needed

# Define the state structure
class State(TypedDict):
    adsh: str
    company_name: str
    sector: str
    financial_data: pd.DataFrame
    activity_data: pd.DataFrame
    emissions_data: pd.DataFrame
    lending_data: pd.DataFrame
    carbon_adjusted_exposure: float
    credits_traded_tonnes: float
    trade_price_usd: float
    status: str

# Helper function to parse Markdown table
def parse_markdown_table(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # Find table rows (skip header and separator)
        for line in lines[2:]:  # Skip "| Field | Value |" and "|-------|-------|"
            if line.strip().startswith('|'):
                fields = [f.strip() for f in line.strip().split('|')[1:-1]]  # Skip leading/trailing |
                if len(fields) != 3:
                    raise ValueError(f"Invalid number of fields in Markdown table: {fields}")
                # Clean and convert fields
                credits_traded = float(fields[0].replace(',', ''))
                price_per_tonne = float(fields[1].replace('$', '').replace(',', ''))
                total_price = float(fields[2].replace('$', '').replace(',', ''))
                parsed_data = {
                    'credits_traded_tonnes': credits_traded,
                    'price_per_tonne_usd': price_per_tonne,
                    'total_price_usd': total_price
                }
                print(f"Parsed Markdown table from {file_path}: {parsed_data}")
                return parsed_data
        raise ValueError("No valid table row found in Markdown file")
    except Exception as e:
        print(f"Error parsing Markdown file {file_path}: {e}")
        return None

# Node functions
def fetch_data(state):
    conn = sqlite3.connect('sec_financials.db')
    try:
        state['sector'] = pd.read_sql(f"SELECT sic FROM submissions WHERE adsh = '{state['adsh']}'", conn)['sic'].iloc[0]
        sic_to_sector = {1311: 'Energy', 1623: 'Construction', 2000: 'Food', 2080: 'Beverages', 3531: 'Machinery', 3826: 'Tech', 7370: 'Tech', 4011: 'Transportation'}
        state['sector'] = sic_to_sector.get(state['sector'], 'Other')
        
        state['financial_data'] = pd.read_sql(f"""
            SELECT year, expenses, revenue
            FROM income_statement
            WHERE adsh = '{state['adsh']}'
        """, conn)
        
        state['activity_data'] = pd.read_sql(f"""
            SELECT year, tag, value
            FROM business_activity
            WHERE adsh = '{state['adsh']}'
        """, conn)
        
        state['emissions_data'] = pd.read_sql(f"""
            SELECT year, total_co2e, scope1_co2e, scope2_co2e, scope3_co2e, credits_purchased_tonnes, uncertainty, proxy_co2e
            FROM emissions_estimates
            WHERE adsh = '{state['adsh']}'
        """, conn)
        
        state['lending_data'] = pd.read_sql(f"""
            SELECT total_loans_outstanding_usd, number_of_loans, avg_interest_rate_pct,
                   weighted_avg_maturity_years, non_performing_loans_pct, total_exposure_usd,
                   collateral_coverage_ratio, credit_rating, default_probability_pct
            FROM lending_data
            WHERE adsh = '{state['adsh']}'
        """, conn)
        
        state['status'] = 'data_fetched'
        print(f"Fetched data for {state['company_name']}: emissions_data rows={len(state['emissions_data'])}, lending_data rows={len(state['lending_data'])}")
    except Exception as e:
        state['status'] = f'fetch_failed: {e}'
        print(f"Fetch failed for {state['company_name']}: {e}")
    finally:
        conn.close()
    return state

def validate_emissions(state):
    if state['emissions_data'].empty:
        state['status'] = 'validation_failed: no_emissions_data'
        print(f"Validation failed for {state['company_name']}: No emissions data")
        return state
    state['emissions_data']['valid'] = (abs(state['emissions_data']['total_co2e'] - state['emissions_data']['proxy_co2e']) / state['emissions_data']['proxy_co2e']) <= 0.2
    state['status'] = 'emissions_validated' if state['emissions_data']['valid'].any() else 'validation_failed: outside_tolerance'
    print(f"Emissions validation for {state['company_name']}: {'Valid' if state['emissions_data']['valid'].any() else 'Outside tolerance'}")
    return state

def calculate_carbon_adjusted_exposure(state):
    if state['status'] != 'emissions_validated' or state['lending_data'].empty:
        state['status'] = 'adjustment_failed: invalid_emissions_or_lending'
        state['carbon_adjusted_exposure'] = 0
        print(f"Exposure adjustment skipped for {state['company_name']}: Invalid emissions or no lending data")
        return state
    
    exposure = state['lending_data']['total_exposure_usd'].iloc[0] if not state['lending_data'].empty else 0
    total_co2e = state['emissions_data']['total_co2e'].iloc[0] if not state['emissions_data'].empty else 0
    credits_purchased = state['emissions_data'].get('credits_purchased_tonnes', pd.Series([0])).iloc[0]
    
    threshold = 1_000_000
    offset_ratio = credits_purchased / total_co2e if total_co2e > 0 else 0
    adjustment_factor = 1 + min(0.05, 0.05 * (total_co2e / threshold)) * (1 - offset_ratio)
    state['carbon_adjusted_exposure'] = exposure * adjustment_factor
    
    conn = sqlite3.connect('sec_financials.db')
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO carbon_adjusted_exposures (adsh, company_name, carbon_adjusted_exposure)
            VALUES (?, ?, ?)
        ''', (state['adsh'], state['company_name'], state['carbon_adjusted_exposure']))
        conn.commit()
        state['status'] = 'exposure_calculated'
        print(f"Exposure calculated for {state['company_name']}: ${state['carbon_adjusted_exposure']:,.2f}")
    except Exception as e:
        state['status'] = f'exposure_calc_failed: {e}'
        print(f"Exposure calculation failed for {state['company_name']}: {e}")
    finally:
        conn.close()
    return state

def trade_credits(state):
    if state['status'] != 'exposure_calculated':
        state['status'] = 'trade_skipped: calculation_failed'
        state['credits_traded_tonnes'] = 0
        state['trade_price_usd'] = 0
        print(f"Trade skipped for {state['company_name']}: Previous step failed")
        return state

    total_co2e = state['emissions_data']['total_co2e'].iloc[0] if not state['emissions_data'].empty else 0
    credits_purchased = state['emissions_data'].get('credits_purchased_tonnes', pd.Series([0])).iloc[0]
    exposure = state['lending_data']['total_exposure_usd'].iloc[0] if not state['lending_data'].empty else 0
    default_prob = state['lending_data']['default_probability_pct'].iloc[0] if not state['lending_data'].empty else 0
    
    print(f"Starting trade for {state['company_name']} (sector: {state['sector']}, exposure: ${exposure:,.2f}, total_co2e: {total_co2e:,.2f})")
    
    buyer = Agent(
        role='Carbon Credit Buyer',
        goal='Purchase credits to offset emissions and reduce portfolio risk',
        backstory=f'Represents a bank managing {state["company_name"]} ({state["sector"]} sector) with exposure ${exposure:,.2f} and default probability {default_prob:.2f}%.',
        llm="ollama/llama2:latest",
        verbose=True
    )
    seller = Agent(
        role='Carbon Credit Seller',
        goal='Sell credits at optimal price based on market demand and risk',
        backstory=f'Represents a renewable energy project offering credits for {state["company_name"]}.',
        llm="ollama/llama2:latest",
        verbose=True
    )
    coordinator = Agent(
        role='Market Coordinator',
        goal='Match buyer and seller offers to finalize trades',
        backstory='Facilitates carbon credit trades, ensuring fair pricing and compliance.',
        llm="ollama/llama2:latest",
        verbose=True
    )
    
    target_offset = 0.8
    credits_needed = max(0, total_co2e * target_offset - credits_purchased)
    budget = exposure * 0.01
    base_price = 30
    price_adjustment = 1 + (default_prob / 100) if state['sector'] == 'Energy' else 1
    
    buyer_task = Task(
        description=f'''
        For {state["company_name"]} ({state["sector"]}), buy up to {credits_needed:,.2f} tonnes of carbon credits to offset {total_co2e:,.2f} tonnes CO2e, within budget ${budget:,.2f}. Propose a price between $20 and $50 per tonne, considering default risk {default_prob:.2f}%. Output your proposal in a Markdown table with columns: Credits Traded (Tonnes), Price Per Tonne (USD), Total Price (USD).
        Example:
        | Credits Traded (Tonnes) | Price Per Tonne (USD) | Total Price (USD) |
        |-------------------------|-----------------------|-------------------|
        | 50000                   | 30.00                 | 1500000.00        |
        ''',
        agent=buyer,
        expected_output='Markdown table with Credits Traded (Tonnes), Price Per Tonne (USD), Total Price (USD).'
    )
    seller_task = Task(
        description=f'''
        For {state["company_name"]} ({state["sector"]}), offer up to {credits_needed:,.2f} tonnes of credits at a price between ${base_price * price_adjustment:,.2f} and $50/tonne, based on market conditions and risk. Read the buyer's proposal from buyer_output.md to align your offer. Output your proposal in a Markdown table with columns: Credits Traded (Tonnes), Price Per Tonne (USD), Total Price (USD).
        Example:
        | Credits Traded (Tonnes) | Price Per Tonne (USD) | Total Price (USD) |
        |-------------------------|-----------------------|-------------------|
        | 50000                   | 32.00                 | 1600000.00        |
        ''',
        agent=seller,
        expected_output='Markdown table with Credits Traded (Tonnes), Price Per Tonne (USD), Total Price (USD).'
    )
    coordinator_task = Task(
        description=f'''
        Match the buyer and seller offers for {state["company_name"]}. Buyer needs {credits_needed:,.2f} tonnes within ${budget:,.2f}. Seller offers at ${base_price * price_adjustment:,.2f} to $50/tonne. Read buyer_output.md and seller_output.md to finalize the trade quantity and price, ensuring budget compliance. Output the final trade in a Markdown table with columns: Credits Traded (Tonnes), Price Per Tonne (USD), Total Price (USD).
        Example:
        | Credits Traded (Tonnes) | Price Per Tonne (USD) | Total Price (USD) |
        |-------------------------|-----------------------|-------------------|
        | 50000                   | 31.00                 | 1550000.00        |
        ''',
        agent=coordinator,
        expected_output='Markdown table with Credits Traded (Tonnes), Price Per Tonne (USD), Total Price (USD).'
    )
    
    try:
        # Clear previous output files
        for file in ['buyer_output.md', 'seller_output.md', 'coordinator_output.md']:
            if os.path.exists(file):
                os.remove(file)
        
        # Run Buyer task
        buyer_crew = Crew(agents=[buyer], tasks=[buyer_task], verbose=True)
        buyer_result = buyer_crew.kickoff()
        with open('buyer_output.md', 'w') as f:
            f.write(str(buyer_result))
        print(f"Buyer output for {state['company_name']}: {buyer_result}")
        
        # Run Seller task
        seller_crew = Crew(agents=[seller], tasks=[seller_task], verbose=True)
        seller_result = seller_crew.kickoff()
        with open('seller_output.md', 'w') as f:
            f.write(str(seller_result))
        print(f"Seller output for {state['company_name']}: {seller_result}")
        
        # Run Coordinator task
        coordinator_crew = Crew(agents=[coordinator], tasks=[coordinator_task], verbose=True)
        coordinator_result = coordinator_crew.kickoff()
        with open('coordinator_output.md', 'w') as f:
            f.write(str(coordinator_result))
        print(f"Coordinator output for {state['company_name']}: {coordinator_result}")
        
        # Parse final trade from coordinator_output.md
        trade_data = parse_markdown_table('coordinator_output.md')
        if trade_data:
            credits_traded = trade_data['credits_traded_tonnes']
            price_per_tonne = trade_data['price_per_tonne_usd']
            total_price = trade_data['total_price_usd']
            print(f"Parsed coordinator output for {state['company_name']}: {credits_traded} tonnes at ${price_per_tonne:.2f}/tonne (total ${total_price:.2f})")
        else:
            print(f"Failed to parse coordinator output for {state['company_name']}, using fallback")
            credits_traded = min(credits_needed, budget / (base_price * price_adjustment))
            total_price = credits_traded * base_price * price_adjustment
            price_per_tonne = base_price * price_adjustment
            print(f"Using fallback for {state['company_name']}: {credits_traded} tonnes at ${price_per_tonne:.2f}/tonne (total ${total_price:.2f})")
    
    except Exception as e:
        print(f"Trade failed for {state['company_name']}: {e}")
        credits_traded = min(credits_needed, budget / (base_price * price_adjustment))
        total_price = credits_traded * base_price * price_adjustment
        price_per_tonne = base_price * price_adjustment
        print(f"Using fallback for {state['company_name']}: {credits_traded} tonnes at ${price_per_tonne:.2f}/tonne (total ${total_price:.2f})")
    
    state['credits_traded_tonnes'] = credits_traded
    state['trade_price_usd'] = total_price
    
    conn = sqlite3.connect('sec_financials.db')
    try:
        trade_timestamp = datetime.now().isoformat()
        df_trade = pd.DataFrame({
            'adsh': [state['adsh']],
            'company_name': [state['company_name']],
            'trade_timestamp': [trade_timestamp],
            'credits_traded_tonnes': [credits_traded],
            'trade_price_usd': [total_price]
        })
        df_trade.to_sql('carbon_trades', conn, if_exists='append', index=False)
        conn.commit()
        print(f"Trade inserted for {state['company_name']}: {credits_traded} tonnes at ${total_price:.2f}")
        state['status'] = 'trade_completed'
    except Exception as e:
        print(f"Trade insertion failed for {state['company_name']}: {e}")
        state['status'] = f'trade_failed: {e}'
    finally:
        conn.close()
    
    return state

# Build the graph
graph = StateGraph(State)
graph.add_node("fetch", fetch_data)
graph.add_node("validate", validate_emissions)
graph.add_node("adjust_exposure", calculate_carbon_adjusted_exposure)
graph.add_node("trade", trade_credits)
graph.add_edge("fetch", "validate")
graph.add_edge("validate", "adjust_exposure")
graph.add_edge("adjust_exposure", "trade")
graph.add_edge("trade", END)
graph.set_entry_point("fetch")

app = graph.compile()

conn = sqlite3.connect('sec_financials.db')
conn.execute('CREATE INDEX IF NOT EXISTS idx_adsh_lending ON lending_data(adsh)')
conn.execute('CREATE INDEX IF NOT EXISTS idx_adsh_emissions ON emissions_estimates(adsh)')
conn.execute('CREATE INDEX IF NOT EXISTS idx_adsh_submissions ON submissions(adsh)')
conn.execute('''
    CREATE TABLE IF NOT EXISTS carbon_adjusted_exposures (
        adsh TEXT,
        company_name TEXT,
        carbon_adjusted_exposure FLOAT,
        PRIMARY KEY (adsh)
    )
''')
conn.execute('''
    CREATE TABLE IF NOT EXISTS carbon_trades (
        adsh TEXT,
        company_name TEXT,
        trade_timestamp TEXT,
        credits_traded_tonnes FLOAT,
        trade_price_usd FLOAT,
        PRIMARY KEY (adsh, trade_timestamp)
    )
''')
conn.close()

conn = sqlite3.connect('sec_financials.db')
try:
    df_sub = pd.read_sql("SELECT adsh, name AS company_name FROM submissions", conn)
except Exception as e:
    print(f"Error: Failed to read submissions table: {e}")
    conn.close()
    exit(1)
conn.close()

df_sub = df_sub.head(5)
print(f"Processing {len(df_sub)} companies for testing.")

results = []
for _, row in df_sub.iterrows():
    print(f"Processing {row['company_name']}...")
    result = app.invoke({"adsh": row['adsh'], "company_name": row['company_name'], "sector": ""})
    results.append({
        'adsh': row['adsh'],
        'company_name': row['company_name'],
        'status': result['status'],
        'carbon_adjusted_exposure': result.get('carbon_adjusted_exposure', 0),
        'credits_traded_tonnes': result.get('credits_traded_tonnes', 0),
        'trade_price_usd': result.get('trade_price_usd', 0)
    })

conn = sqlite3.connect('sec_financials.db')
pd.DataFrame(results).to_sql('workflow_results', conn, if_exists='replace', index=False)
conn.close()
print("Test batch processing completed. Check 'carbon_trades' table and console logs.")