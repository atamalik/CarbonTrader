
import threading
import pandas as pd
import numpy as np
import sqlite3
from langgraph.graph import StateGraph, END
from typing import TypedDict
from crewai import Agent, Task, Crew
from datetime import datetime
import os
import time
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Note: Set LANGCHAIN_API_KEY environment variable if needed

# --- app_mcp.py Logic ---
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

def parse_markdown_table(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines[2:]:  # Skip header and separator
            if line.strip().startswith('|') and '---' not in line:
                fields = [f.strip() for f in line.strip().split('|')[1:-1]]
                if len(fields) != 3:
                    raise ValueError(f"Invalid number of fields: {fields}")
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
        raise ValueError("No valid table row found")
    except Exception as e:
        print(f"Error parsing Markdown file {file_path}: {e}")
        return None

def fetch_data(state, conn):
    try:
        state['sector'] = pd.read_sql(f"SELECT sic FROM submissions WHERE adsh = '{state['adsh']}'", conn)['sic'].iloc[0]
        sic_to_sector = {1311: 'Energy', 1623: 'Construction', 2000: 'Food', 2080: 'Beverages', 3531: 'Machinery', 3826: 'Tech', 7370: 'Tech', 4011: 'Transportation'}
        state['sector'] = sic_to_sector.get(state['sector'], 'Other')
        
        state['financial_data'] = pd.read_sql(f"SELECT year, expenses, revenue FROM income_statement WHERE adsh = '{state['adsh']}'", conn)
        state['activity_data'] = pd.read_sql(f"SELECT year, tag, value FROM business_activity WHERE adsh = '{state['adsh']}'", conn)
        state['emissions_data'] = pd.read_sql(f"SELECT year, total_co2e, scope1_co2e, scope2_co2e, scope3_co2e, credits_purchased_tonnes, uncertainty, proxy_co2e FROM emissions_estimates WHERE adsh = '{state['adsh']}'", conn)
        state['lending_data'] = pd.read_sql(f"SELECT total_loans_outstanding_usd, number_of_loans, avg_interest_rate_pct, weighted_avg_maturity_years, non_performing_loans_pct, total_exposure_usd, collateral_coverage_ratio, credit_rating, default_probability_pct FROM lending_data WHERE adsh = '{state['adsh']}'", conn)
        
        state['status'] = 'data_fetched'
        print(f"Fetched data for {state['company_name']}: emissions_data rows={len(state['emissions_data'])}, lending_data rows={len(state['lending_data'])}")
    except Exception as e:
        state['status'] = f'fetch_failed: {e}'
        print(f"Fetch failed for {state['company_name']}: {e}")
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

def calculate_carbon_adjusted_exposure(state, conn):
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
    
    try:
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO carbon_adjusted_exposures (adsh, company_name, carbon_adjusted_exposure) VALUES (?, ?, ?)', 
                      (state['adsh'], state['company_name'], state['carbon_adjusted_exposure']))
        conn.commit()
        state['status'] = 'exposure_calculated'
        print(f"Exposure calculated for {state['company_name']}: ${state['carbon_adjusted_exposure']:,.2f}")
    except Exception as e:
        state['status'] = f'exposure_calc_failed: {e}'
        print(f"Exposure calculation failed for {state['company_name']}: {e}")
    return state

def trade_credits(state, conn):
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
        for file in ['buyer_output.md', 'seller_output.md', 'coordinator_output.md']:
            if os.path.exists(file):
                os.remove(file)
        
        buyer_crew = Crew(agents=[buyer], tasks=[buyer_task], verbose=True)
        buyer_result = buyer_crew.kickoff()
        with open('buyer_output.md', 'w') as f:
            f.write(str(buyer_result))
        print(f"Buyer output for {state['company_name']}: {buyer_result}")
        
        seller_crew = Crew(agents=[seller], tasks=[seller_task], verbose=True)
        seller_result = seller_crew.kickoff()
        with open('seller_output.md', 'w') as f:
            f.write(str(seller_result))
        print(f"Seller output for {state['company_name']}: {seller_result}")
        
        coordinator_crew = Crew(agents=[coordinator], tasks=[coordinator_task], verbose=True)
        coordinator_result = coordinator_crew.kickoff()
        with open('coordinator_output.md', 'w') as f:
            f.write(str(coordinator_result))
        print(f"Coordinator output for {state['company_name']}: {coordinator_result}")
        
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
    
    return state

def run_mcp():
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

    try:
        df_sub = pd.read_sql("SELECT adsh, name AS company_name FROM submissions", conn)
    except Exception as e:
        print(f"Error: Failed to read submissions table: {e}")
        conn.close()
        return

    graph = StateGraph(State)
    graph.add_node("fetch", lambda state: fetch_data(state, conn))
    graph.add_node("validate", validate_emissions)
    graph.add_node("adjust_exposure", lambda state: calculate_carbon_adjusted_exposure(state, conn))
    graph.add_node("trade", lambda state: trade_credits(state, conn))
    graph.add_edge("fetch", "validate")
    graph.add_edge("validate", "adjust_exposure")
    graph.add_edge("adjust_exposure", "trade")
    graph.add_edge("trade", END)
    graph.set_entry_point("fetch")
    app = graph.compile()

    df_sub = df_sub.head(5)  # Remove for full run
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

    pd.DataFrame(results).to_sql('workflow_results', conn, if_exists='replace', index=False)
    conn.close()
    print("Initial trade processing completed.")

# --- simulate_portfolio_update.py Logic ---
sector_factors = {
    'Energy': 0.002, 'Construction': 0.0015, 'Food': 0.001, 'Beverages': 0.0012,
    'Machinery': 0.0011, 'Tech': 0.0008, 'Transportation': 0.0018, 'Other': 0.001
}

def update_portfolio():
    conn = sqlite3.connect('sec_financials.db')
    try:
        submissions = pd.read_sql("SELECT adsh, name, sic FROM submissions", conn)
        lending = pd.read_sql("SELECT adsh, total_exposure_usd, credit_rating FROM lending_data", conn)
        emissions = pd.read_sql("SELECT adsh, total_co2e, credits_purchased_tonnes FROM emissions_estimates", conn)
        
        sic_to_sector = {1311: 'Energy', 1623: 'Construction', 2000: 'Food', 2080: 'Beverages', 3531: 'Machinery', 3826: 'Tech', 7370: 'Tech', 4011: 'Transportation'}
        submissions['sector'] = submissions['sic'].map(sic_to_sector).fillna('Other')
        
        num_updates = max(1, int(len(lending) * 0.1))
        update_indices = np.random.choice(len(lending), num_updates, replace=False)
        total_sold_tonnes = 0
        
        for idx in update_indices:
            adsh = lending.iloc[idx]['adsh']
            old_exposure = lending.iloc[idx]['total_exposure_usd']
            change_pct = np.random.uniform(-0.1, 0.1)
            new_exposure = max(0, old_exposure * (1 + change_pct))
            
            cursor = conn.cursor()
            cursor.execute("UPDATE lending_data SET total_exposure_usd = ? WHERE adsh = ?", (new_exposure, adsh))
            
            sector = submissions[submissions['adsh'] == adsh]['sector'].iloc[0]
            factor = sector_factors.get(sector, 0.001)
            new_total_co2e = max(0, new_exposure * factor)
            
            cursor.execute("UPDATE emissions_estimates SET total_co2e = ? WHERE adsh = ?", (new_total_co2e, adsh))
            
            credits_purchased = emissions[emissions['adsh'] == adsh]['credits_purchased_tonnes'].iloc[0]
            target_offset = new_total_co2e * 0.8
            surplus = credits_purchased - target_offset
            if surplus > 0:
                market_price = 25.0
                surplus_trade_price = surplus * market_price
                trade_timestamp = datetime.now().isoformat()
                company_name = submissions[submissions['adsh'] == adsh]['name'].iloc[0]
                cursor.execute("""
                    INSERT INTO carbon_trades (adsh, company_name, trade_timestamp, credits_traded_tonnes, trade_price_usd)
                    VALUES (?, ?, ?, ?, ?)
                """, (adsh, company_name, trade_timestamp, -surplus, surplus_trade_price))
                total_sold_tonnes += surplus
                print(f"Surplus sell for {company_name}: {-surplus:.2f} tonnes at ${surplus_trade_price:.2f}")
            
            conn.commit()
        
        print(f"Portfolio updated: {num_updates} companies changed, {total_sold_tonnes:.2f} tonnes sold.")
    
    except Exception as e:
        print(f"Error updating portfolio: {e}")
    finally:
        conn.close()

# --- app_ui.py Logic ---
def run_ui():
    st.set_page_config(page_title="ACCMN: Sustainable Portfolio Management", layout="wide")
    st.title("ACCMN: Sustainable Portfolio Management")
    st.markdown("**Powered by Autonomous AI Agents** | Optimize your lending portfolio with ESG insights")
    
    conn = sqlite3.connect('sec_financials.db')
    submissions = pd.read_sql("SELECT adsh, name AS company_name, sic FROM submissions", conn)
    lending = pd.read_sql("SELECT adsh, total_loans_outstanding_usd, total_exposure_usd, non_performing_loans_pct, credit_rating, default_probability_pct FROM lending_data", conn)
    emissions = pd.read_sql("SELECT adsh, total_co2e, scope1_co2e, scope2_co2e, scope3_co2e, credits_purchased_tonnes, uncertainty FROM emissions_estimates", conn)
    exposures = pd.read_sql("SELECT adsh, carbon_adjusted_exposure FROM carbon_adjusted_exposures", conn)
    trades = pd.read_sql("SELECT adsh, company_name, trade_timestamp, credits_traded_tonnes, trade_price_usd FROM carbon_trades ORDER BY trade_timestamp DESC", conn)
    conn.close()
    
    emissions['credits_purchased_tonnes'] = emissions.get('credits_purchased_tonnes', pd.Series([0] * len(emissions)))
    emissions['total_co2e'] = emissions.get('total_co2e', pd.Series([0] * len(emissions)))
    emissions['scope1_co2e'] = emissions.get('scope1_co2e', pd.Series([0] * len(emissions)))
    emissions['scope2_co2e'] = emissions.get('scope2_co2e', pd.Series([0] * len(emissions)))
    emissions['scope3_co2e'] = emissions.get('scope3_co2e', pd.Series([0] * len(emissions)))
    emissions['uncertainty'] = emissions.get('uncertainty', pd.Series([0] * len(emissions)))
    
    df = submissions.merge(lending, on='adsh', how='left').merge(emissions, on='adsh', how='left').merge(exposures, on='adsh', how='left')
    sic_to_sector = {1311: 'Energy', 1623: 'Construction', 2000: 'Food', 2080: 'Beverages', 3531: 'Machinery', 3826: 'Tech', 7370: 'Tech', 4011: 'Transportation'}
    df['sector'] = df['sic'].map(sic_to_sector).fillna('Other')
    
    emissions_count = df['total_co2e'].notnull().sum()
    if emissions_count < len(df) * 0.1:
        st.warning(f"Only {emissions_count} of {len(df)} companies have emissions data. Graphs may be limited.")
    
    tabs = st.tabs(["Portfolio Overview", "Climate Risk", "Carbon Trading", "Trading Desk", "Company Details"])
    
    with tabs[0]:
        st.header("Portfolio Overview")
        col1, col2 = st.columns(2)
        with col1:
            sector_exposure = df.groupby('sector')['carbon_adjusted_exposure'].sum().reset_index()
            if not sector_exposure['carbon_adjusted_exposure'].sum() == 0:
                fig1 = px.bar(sector_exposure, x='sector', y='carbon_adjusted_exposure', title="Total Exposure by Sector ($)",
                              color='sector', color_discrete_sequence=px.colors.sequential.Blues)
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.write("No exposure data available.")
        with col2:
            rating_dist = df.groupby('credit_rating')['total_loans_outstanding_usd'].sum().reset_index()
            if not rating_dist['total_loans_outstanding_usd'].sum() == 0:
                fig2 = px.pie(rating_dist, values='total_loans_outstanding_usd', names='credit_rating', title="Loans by Credit Rating")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write("No loan data available.")
        fig3 = px.scatter(df, x='total_exposure_usd', y='non_performing_loans_pct', color='sector',
                          size='default_probability_pct', hover_data=['company_name'], title="Exposure vs. NPLs")
        if fig3.data:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.write("No data for Exposure vs. NPLs.")
    
    with tabs[1]:
        st.header("Climate Risk")
        col1, col2 = st.columns(2)
        with col1:
            df_emissions = df[df['total_co2e'].notnull() & df['carbon_adjusted_exposure'].notnull()]
            if not df_emissions.empty:
                fig4 = px.scatter(df_emissions, x='total_co2e', y='carbon_adjusted_exposure', color='sector',
                                  hover_data=['company_name'], title="Emissions vs. Adjusted Exposure")
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.write("No emissions or exposure data available.")
        with col2:
            scope_data = df.groupby('sector')[['scope1_co2e', 'scope2_co2e', 'scope3_co2e']].sum().reset_index()
            if scope_data[['scope1_co2e', 'scope2_co2e', 'scope3_co2e']].sum().sum() > 0:
                fig5 = go.Figure(data=[
                    go.Bar(name='Scope 1', x=scope_data['sector'], y=scope_data['scope1_co2e']),
                    go.Bar(name='Scope 2', x=scope_data['sector'], y=scope_data['scope2_co2e']),
                    go.Bar(name='Scope 3', x=scope_data['sector'], y=scope_data['scope3_co2e'])
                ])
                fig5.update_layout(barmode='stack', title="Emissions by Scope and Sector")
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.write("No scope emissions data available.")
    
    with tabs[2]:
        st.header("Carbon Trading")
        col1, col2 = st.columns(2)
        with col1:
            top10 = df_emissions.nlargest(10, 'carbon_adjusted_exposure')
            if not top10.empty:
                fig6 = go.Figure(data=[
                    go.Bar(name='Emissions', x=top10['company_name'], y=top10['total_co2e']),
                    go.Bar(name='Credits Purchased', x=top10['company_name'], y=top10['credits_purchased_tonnes'])
                ])
                fig6.update_layout(barmode='group', title="Emissions vs. Credits Purchased (Top 10)")
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.write("No emissions or credits data available.")
        with col2:
            df_emissions['offset_ratio'] = df_emissions['credits_purchased_tonnes'] / df_emissions['total_co2e'].replace(0, 1)
            df_emissions['exposure_reduction_pct'] = (df_emissions['total_exposure_usd'] - df_emissions['carbon_adjusted_exposure']) / df_emissions['total_exposure_usd'] * 100
            if not df_emissions.empty:
                fig7 = px.scatter(df_emissions, x='offset_ratio', y='exposure_reduction_pct', color='sector',
                                  hover_data=['company_name'], title="Exposure Reduction vs. Offset Ratio")
                st.plotly_chart(fig7, use_container_width=True)
            else:
                st.write("No data for Exposure Reduction vs. Offset Ratio.")
        st.subheader("Carbon Trades")
        if not trades.empty:
            gb = GridOptionsBuilder.from_dataframe(trades)
            gb.configure_gridOptions(autoSizeColumns='allColIds')
            grid_options = gb.build()
            AgGrid(trades, gridOptions=grid_options, height=300)
        else:
            st.write("No carbon trade data available.")
    
    with tabs[3]:
        st.header("Company Details")
        company = st.selectbox("Select Company", df['company_name'])
        company_data = df[df['company_name'] == company]
        
        st.write(f"**{company}**")
        st.metric("Total Exposure ($)", f"{company_data['carbon_adjusted_exposure'].iloc[0]:,.2f}" if pd.notnull(company_data['carbon_adjusted_exposure'].iloc[0]) else "N/A")
        st.metric("Default Probability (%)", f"{company_data['default_probability_pct'].iloc[0]:.2f}" if pd.notnull(company_data['default_probability_pct'].iloc[0]) else "N/A")
        
        if pd.notnull(company_data['total_exposure_usd'].iloc[0]) and pd.notnull(company_data['carbon_adjusted_exposure'].iloc[0]):
            fig8 = go.Figure(data=[
                go.Bar(name='Base Exposure', x=[company], y=company_data['total_exposure_usd']),
                go.Bar(name='Adjusted Exposure', x=[company], y=company_data['carbon_adjusted_exposure'])
            ])
            fig8.update_layout(barmode='group', title="Base vs. Adjusted Exposure")
            st.plotly_chart(fig8, use_container_width=True)
        else:
            st.write("No exposure data available.")
    
    with tabs[4]:
        st.header("Trading Desk")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_traded = trades['credits_traded_tonnes'].sum()
            st.metric("Total Traded Tonnes", f"{total_traded:,.2f}")
        with col2:
            total_revenue = trades['trade_price_usd'].sum()
            st.metric("Total Revenue ($)", f"${total_revenue:,.2f}")
        with col3:
            avg_price = trades['trade_price_usd'].sum() / total_traded if total_traded != 0 else 0
            st.metric("Avg Price per Tonne ($)", f"${avg_price:.2f}")
        
        if not trades.empty:
            gb = GridOptionsBuilder.from_dataframe(trades)
            gb.configure_gridOptions(autoSizeColumns='allColIds', enableFilter=True, enableSorting=True)
            grid_options = gb.build()
            AgGrid(trades, gridOptions=grid_options, height=400)
        else:
            st.write("No trades to display.")
        
        if not trades.empty:
            trades['trade_timestamp'] = pd.to_datetime(trades['trade_timestamp'], format='ISO8601')
            trades_sorted = trades.sort_values('trade_timestamp')
            fig_trade_volume = px.line(trades_sorted, x='trade_timestamp', y='credits_traded_tonnes', title="Trade Volume Over Time")
            st.plotly_chart(fig_trade_volume, use_container_width=True)
            
            trades_with_sector = trades.merge(df[['adsh', 'sector']], on='adsh', how='left')
            sector_trades = trades_with_sector.groupby('sector')['credits_traded_tonnes'].sum().reset_index()
            fig_sector_pie = px.pie(sector_trades, values='credits_traded_tonnes', names='sector', title="Trades by Sector")
            st.plotly_chart(fig_sector_pie, use_container_width=True)
        else:
            st.write("No trades for visualization.")
        
        if st.button("Refresh Data (Manual)"):
            st.rerun()
    
    time.sleep(5)
    st.rerun()

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting ACCMN system...")
    run_mcp()  # Run initial trade processing
    simulation_thread = threading.Thread(target=lambda: [update_portfolio() or time.sleep(5) for _ in iter(int, 1)], daemon=True)
    simulation_thread.start()
    run_ui()  # Launch Streamlit UI
