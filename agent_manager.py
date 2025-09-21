"""
Agent Manager Module for ACCMN
Handles AI agent orchestration, LangGraph workflows, and CrewAI multi-agent coordination
"""

import pandas as pd
import numpy as np
import sqlite3
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List, Optional, Any
from crewai import Agent, Task, Crew
from datetime import datetime
import os
import logging
import threading
import time

# Import configuration
from config import SECTOR_MAPPING, SESSION_KEYS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# Note: Set LANGCHAIN_API_KEY environment variable if needed

class AgentState(TypedDict):
    """State structure for LangGraph workflow"""
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

class AgentManager:
    """Manages AI agent workflows and orchestration"""
    
    def __init__(self, db_path: str = 'sec_financials.db'):
        self.db_path = db_path
        self.graph = self._build_workflow_graph()
        self.simulation_thread = None
        self.is_running = False
        
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow for agent orchestration"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("fetch", self._fetch_data)
        graph.add_node("validate", self._validate_emissions)
        graph.add_node("adjust_exposure", self._calculate_carbon_adjusted_exposure)
        graph.add_node("trade", self._trade_credits)
        
        # Add edges
        graph.add_edge("fetch", "validate")
        graph.add_edge("validate", "adjust_exposure")
        graph.add_edge("adjust_exposure", "trade")
        graph.add_edge("trade", END)
        graph.set_entry_point("fetch")
        
        return graph.compile()
    
    def _fetch_data(self, state: AgentState) -> AgentState:
        """Fetch company data from database"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Get sector from submissions
            sector_result = pd.read_sql(f"SELECT sic FROM submissions WHERE adsh = '{state['adsh']}'", conn)
            if not sector_result.empty:
                sic = sector_result['sic'].iloc[0]
                state['sector'] = SECTOR_MAPPING.get(sic, 'Other')
            else:
                state['sector'] = 'Other'
            
            # Fetch financial data
            state['financial_data'] = pd.read_sql(f"""
                SELECT year, expenses, revenue
                FROM income_statement
                WHERE adsh = '{state['adsh']}'
            """, conn)
            
            # Fetch activity data
            state['activity_data'] = pd.read_sql(f"""
                SELECT year, tag, value
                FROM business_activity
                WHERE adsh = '{state['adsh']}'
            """, conn)
            
            # Fetch emissions data
            state['emissions_data'] = pd.read_sql(f"""
                SELECT year, total_co2e, scope1_co2e, scope2_co2e, scope3_co2e, 
                       credits_purchased_tonnes, uncertainty, proxy_co2e
                FROM emissions_estimates
                WHERE adsh = '{state['adsh']}'
            """, conn)
            
            # Fetch lending data
            state['lending_data'] = pd.read_sql(f"""
                SELECT total_loans_outstanding_usd, number_of_loans, avg_interest_rate_pct,
                       weighted_avg_maturity_years, non_performing_loans_pct, total_exposure_usd,
                       collateral_coverage_ratio, credit_rating, default_probability_pct
                FROM lending_data
                WHERE adsh = '{state['adsh']}'
            """, conn)
            
            state['status'] = 'data_fetched'
            logger.info(f"Fetched data for {state['company_name']}: emissions_data rows={len(state['emissions_data'])}, lending_data rows={len(state['lending_data'])}")
            
        except Exception as e:
            state['status'] = f'fetch_failed: {e}'
            logger.error(f"Fetch failed for {state['company_name']}: {e}")
        finally:
            conn.close()
        
        return state
    
    def _validate_emissions(self, state: AgentState) -> AgentState:
        """Validate emissions data against proxy calculations"""
        if state['emissions_data'].empty:
            state['status'] = 'validation_failed: no_emissions_data'
            logger.warning(f"Validation failed for {state['company_name']}: No emissions data")
            return state
        
        # Check if proxy_co2e column exists
        if 'proxy_co2e' in state['emissions_data'].columns:
            state['emissions_data']['valid'] = (
                abs(state['emissions_data']['total_co2e'] - state['emissions_data']['proxy_co2e']) / 
                state['emissions_data']['proxy_co2e'].replace(0, 1)
            ) <= 0.2
        else:
            # If no proxy data, assume valid
            state['emissions_data']['valid'] = True
        
        state['status'] = 'emissions_validated' if state['emissions_data']['valid'].any() else 'validation_failed: outside_tolerance'
        logger.info(f"Emissions validation for {state['company_name']}: {'Valid' if state['emissions_data']['valid'].any() else 'Outside tolerance'}")
        
        return state
    
    def _calculate_carbon_adjusted_exposure(self, state: AgentState) -> AgentState:
        """Calculate carbon-adjusted exposure using AI logic"""
        if state['status'] != 'emissions_validated' or state['lending_data'].empty:
            state['status'] = 'adjustment_failed: invalid_emissions_or_lending'
            state['carbon_adjusted_exposure'] = 0
            logger.warning(f"Exposure adjustment skipped for {state['company_name']}: Invalid emissions or no lending data")
            return state
        
        exposure = state['lending_data']['total_exposure_usd'].iloc[0] if not state['lending_data'].empty else 0
        total_co2e = state['emissions_data']['total_co2e'].iloc[0] if not state['emissions_data'].empty else 0
        credits_purchased = state['emissions_data'].get('credits_purchased_tonnes', pd.Series([0])).iloc[0]
        
        # AI-powered carbon adjustment logic
        threshold = 1_000_000
        offset_ratio = credits_purchased / total_co2e if total_co2e > 0 else 0
        adjustment_factor = 1 + min(0.05, 0.05 * (total_co2e / threshold)) * (1 - offset_ratio)
        state['carbon_adjusted_exposure'] = exposure * adjustment_factor
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO carbon_adjusted_exposures (adsh, company_name, carbon_adjusted_exposure)
                VALUES (?, ?, ?)
            ''', (state['adsh'], state['company_name'], state['carbon_adjusted_exposure']))
            conn.commit()
            state['status'] = 'exposure_calculated'
            logger.info(f"Exposure calculated for {state['company_name']}: ${state['carbon_adjusted_exposure']:,.2f}")
        except Exception as e:
            state['status'] = f'exposure_calc_failed: {e}'
            logger.error(f"Exposure calculation failed for {state['company_name']}: {e}")
        finally:
            conn.close()
        
        return state
    
    def _trade_credits(self, state: AgentState) -> AgentState:
        """Execute AI-powered carbon credit trading using CrewAI agents"""
        if state['status'] != 'exposure_calculated':
            state['status'] = 'trade_skipped: calculation_failed'
            state['credits_traded_tonnes'] = 0
            state['trade_price_usd'] = 0
            logger.warning(f"Trade skipped for {state['company_name']}: Previous step failed")
            return state
        
        total_co2e = state['emissions_data']['total_co2e'].iloc[0] if not state['emissions_data'].empty else 0
        credits_purchased = state['emissions_data'].get('credits_purchased_tonnes', pd.Series([0])).iloc[0]
        exposure = state['lending_data']['total_exposure_usd'].iloc[0] if not state['lending_data'].empty else 0
        default_prob = state['lending_data']['default_probability_pct'].iloc[0] if not state['lending_data'].empty else 0
        
        logger.info(f"Starting AI trade for {state['company_name']} (sector: {state['sector']}, exposure: ${exposure:,.2f}, total_co2e: {total_co2e:,.2f})")
        
        # Create AI agents
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
        
        # Calculate trading parameters
        target_offset = 0.8
        credits_needed = max(0, total_co2e * target_offset - credits_purchased)
        budget = exposure * 0.01
        base_price = 30
        price_adjustment = 1 + (default_prob / 100) if state['sector'] == 'Energy' else 1
        
        # Create tasks
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
        
        # Execute AI agent workflow
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
            logger.info(f"Buyer output for {state['company_name']}: {buyer_result}")
            
            # Run Seller task
            seller_crew = Crew(agents=[seller], tasks=[seller_task], verbose=True)
            seller_result = seller_crew.kickoff()
            with open('seller_output.md', 'w') as f:
                f.write(str(seller_result))
            logger.info(f"Seller output for {state['company_name']}: {seller_result}")
            
            # Run Coordinator task
            coordinator_crew = Crew(agents=[coordinator], tasks=[coordinator_task], verbose=True)
            coordinator_result = coordinator_crew.kickoff()
            with open('coordinator_output.md', 'w') as f:
                f.write(str(coordinator_result))
            logger.info(f"Coordinator output for {state['company_name']}: {coordinator_result}")
            
            # Parse final trade
            trade_data = self._parse_markdown_table('coordinator_output.md')
            if trade_data:
                credits_traded = trade_data['credits_traded_tonnes']
                price_per_tonne = trade_data['price_per_tonne_usd']
                total_price = trade_data['total_price_usd']
                logger.info(f"Parsed coordinator output for {state['company_name']}: {credits_traded} tonnes at ${price_per_tonne:.2f}/tonne (total ${total_price:.2f})")
            else:
                logger.warning(f"Failed to parse coordinator output for {state['company_name']}, using fallback")
                credits_traded = min(credits_needed, budget / (base_price * price_adjustment))
                total_price = credits_traded * base_price * price_adjustment
                price_per_tonne = base_price * price_adjustment
                logger.info(f"Using fallback for {state['company_name']}: {credits_traded} tonnes at ${price_per_tonne:.2f}/tonne (total ${total_price:.2f})")
        
        except Exception as e:
            logger.error(f"AI trade failed for {state['company_name']}: {e}")
            credits_traded = min(credits_needed, budget / (base_price * price_adjustment))
            total_price = credits_traded * base_price * price_adjustment
            price_per_tonne = base_price * price_adjustment
            logger.info(f"Using fallback for {state['company_name']}: {credits_traded} tonnes at ${price_per_tonne:.2f}/tonne (total ${total_price:.2f})")
        
        state['credits_traded_tonnes'] = credits_traded
        state['trade_price_usd'] = total_price
        
        # Save trade to database
        conn = sqlite3.connect(self.db_path)
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
            logger.info(f"Trade inserted for {state['company_name']}: {credits_traded} tonnes at ${total_price:.2f}")
            state['status'] = 'trade_completed'
        except Exception as e:
            logger.error(f"Trade insertion failed for {state['company_name']}: {e}")
            state['status'] = f'trade_failed: {e}'
        finally:
            conn.close()
        
        return state
    
    def _parse_markdown_table(self, file_path: str) -> Optional[Dict]:
        """Parse markdown table from agent output"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines[2:]:  # Skip header and separator
                if line.strip().startswith('|'):
                    fields = [f.strip() for f in line.strip().split('|')[1:-1]]
                    if len(fields) != 3:
                        continue
                    
                    credits_traded = float(fields[0].replace(',', ''))
                    price_per_tonne = float(fields[1].replace('$', '').replace(',', ''))
                    total_price = float(fields[2].replace('$', '').replace(',', ''))
                    
                    return {
                        'credits_traded_tonnes': credits_traded,
                        'price_per_tonne_usd': price_per_tonne,
                        'total_price_usd': total_price
                    }
            
            return None
        except Exception as e:
            logger.error(f"Error parsing Markdown file {file_path}: {e}")
            return None
    
    def process_company(self, adsh: str, company_name: str) -> Dict:
        """Process a single company through the AI workflow"""
        try:
            result = self.graph.invoke({
                "adsh": adsh,
                "company_name": company_name,
                "sector": "",
                "financial_data": pd.DataFrame(),
                "activity_data": pd.DataFrame(),
                "emissions_data": pd.DataFrame(),
                "lending_data": pd.DataFrame(),
                "carbon_adjusted_exposure": 0,
                "credits_traded_tonnes": 0,
                "trade_price_usd": 0,
                "status": ""
            })
            
            return {
                'adsh': adsh,
                'company_name': company_name,
                'status': result['status'],
                'carbon_adjusted_exposure': result.get('carbon_adjusted_exposure', 0),
                'credits_traded_tonnes': result.get('credits_traded_tonnes', 0),
                'trade_price_usd': result.get('trade_price_usd', 0)
            }
        except Exception as e:
            logger.error(f"Error processing company {company_name}: {e}")
            return {
                'adsh': adsh,
                'company_name': company_name,
                'status': f'processing_failed: {e}',
                'carbon_adjusted_exposure': 0,
                'credits_traded_tonnes': 0,
                'trade_price_usd': 0
            }
    
    def process_batch(self, companies: List[Dict], max_companies: int = 5) -> List[Dict]:
        """Process a batch of companies through AI workflows"""
        results = []
        companies_to_process = companies[:max_companies]
        
        logger.info(f"Processing {len(companies_to_process)} companies through AI workflows")
        
        for company in companies_to_process:
            result = self.process_company(company['adsh'], company['company_name'])
            results.append(result)
        
        return results
    
    def start_portfolio_simulation(self):
        """Start the portfolio simulation in a background thread"""
        if self.is_running:
            logger.warning("Portfolio simulation is already running")
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._portfolio_simulation_loop, daemon=True)
        self.simulation_thread.start()
        logger.info("Portfolio simulation started")
    
    def stop_portfolio_simulation(self):
        """Stop the portfolio simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        logger.info("Portfolio simulation stopped")
    
    def _portfolio_simulation_loop(self):
        """Portfolio simulation loop (from simulate_portfolio_update.py)"""
        conn = sqlite3.connect(self.db_path)
        
        # Sector-specific emissions factors
        sector_factors = {
            'Energy': 0.002, 'Construction': 0.0015, 'Food': 0.001, 'Beverages': 0.0012,
            'Machinery': 0.0011, 'Tech': 0.0008, 'Transportation': 0.0018, 'Other': 0.001
        }
        
        try:
            while self.is_running:
                logger.info("Starting portfolio update cycle")
                
                # Load data
                submissions = pd.read_sql("SELECT adsh, sector AS company_name, sic FROM submissions", conn)
                lending = pd.read_sql("SELECT adsh, total_exposure_usd, credit_rating FROM lending_data", conn)
                emissions = pd.read_sql("SELECT adsh, scope1_co2e, scope2_co2e, scope3_co2e, credits_purchased_tonnes FROM emissions_estimates", conn)
                balance_sheet = pd.read_sql("SELECT adsh, Liabilities FROM balance_sheet", conn)
                
                # Map SIC to sector
                submissions['sector'] = submissions['sic'].map(SECTOR_MAPPING).fillna('Other')
                
                # Update random subset of companies
                num_updates = max(1, int(len(lending) * 0.1))
                update_indices = np.random.choice(len(lending), num_updates, replace=False)
                total_sold_tonnes = 0
                
                for idx in update_indices:
                    adsh = lending.iloc[idx]['adsh']
                    company_name = submissions[submissions['adsh'] == adsh]['company_name'].iloc[0]
                    old_exposure = lending.iloc[idx]['total_exposure_usd']
                    change_pct = np.random.uniform(-0.1, 0.1)
                    new_exposure = max(0, old_exposure * (1 + change_pct))
                    
                    logger.info(f"Processing {company_name} ({adsh}): Exposure {old_exposure:,.2f} -> {new_exposure:,.2f} ({change_pct:.2%})")
                    
                    # Update lending data
                    cursor = conn.cursor()
                    cursor.execute("UPDATE lending_data SET total_exposure_usd = ? WHERE adsh = ?", (new_exposure, adsh))
                    
                    # Update emissions
                    sector = submissions[submissions['adsh'] == adsh]['sector'].iloc[0]
                    factor = sector_factors.get(sector, 0.001)
                    new_total_co2e = max(0, new_exposure * factor)
                    cursor.execute("UPDATE emissions_estimates SET total_co2e = ? WHERE adsh = ?", (new_total_co2e, adsh))
                    
                    # PCAF Financed Emissions Calculation
                    company_debt = balance_sheet[balance_sheet['adsh'] == adsh]['Liabilities'].iloc[0] if not balance_sheet[balance_sheet['adsh'] == adsh].empty and 'Liabilities' in balance_sheet.columns else new_exposure * 2
                    bank_share = new_exposure / company_debt if company_debt > 0 else 0
                    financed_scope1 = bank_share * emissions[emissions['adsh'] == adsh]['scope1_co2e'].iloc[0] if not emissions[emissions['adsh'] == adsh].empty else 0
                    financed_scope2 = bank_share * emissions[emissions['adsh'] == adsh]['scope2_co2e'].iloc[0] if not emissions[emissions['adsh'] == adsh].empty else 0
                    financed_scope3 = bank_share * emissions[emissions['adsh'] == adsh]['scope3_co2e'].iloc[0] if not emissions[emissions['adsh'] == adsh].empty else 0
                    financed_total = financed_scope1 + financed_scope2 + financed_scope3
                    
                    # Update financed emissions
                    baseline_total = 0
                    delta_co2e = financed_total - baseline_total
                    update_timestamp = datetime.now().isoformat()
                    cursor.execute("""
                        INSERT OR REPLACE INTO financed_emissions (adsh, company_name, sector, baseline_finance_total_co2e, current_finance_total_co2e, bank_share_pct, total_debt_usd, delta_co2e, update_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (adsh, company_name, sector, baseline_total, financed_total, bank_share * 100, company_debt, delta_co2e, update_timestamp))
                    
                    # Check for surplus credits
                    credits_purchased = emissions[emissions['adsh'] == adsh]['credits_purchased_tonnes'].iloc[0] if not emissions[emissions['adsh'] == adsh].empty else 0
                    target_offset = new_total_co2e * 0.8
                    surplus = credits_purchased - target_offset
                    
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
                time.sleep(5)  # Wait 5 seconds before next update
                
        except Exception as e:
            logger.error(f"Error in portfolio simulation: {e}")
        finally:
            conn.close()
            self.is_running = False

# Global instance
agent_manager = AgentManager()

def process_company_with_ai(adsh: str, company_name: str) -> Dict:
    """Process a company through AI workflow"""
    return agent_manager.process_company(adsh, company_name)

def process_batch_with_ai(companies: List[Dict], max_companies: int = 5) -> List[Dict]:
    """Process a batch of companies through AI workflows"""
    return agent_manager.process_batch(companies, max_companies)

def start_ai_simulation():
    """Start AI-powered portfolio simulation"""
    agent_manager.start_portfolio_simulation()

def stop_ai_simulation():
    """Stop AI-powered portfolio simulation"""
    agent_manager.stop_portfolio_simulation()
