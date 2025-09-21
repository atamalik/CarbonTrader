"""
Data Manager Module for ACCMN
Handles data pipeline, caching, and integration between internal and external data sources
"""

import streamlit as st
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import configuration and market data
from config import (
    DATABASE_PATH, 
    SESSION_KEYS, 
    REFRESH_INTERVALS, 
    SECTOR_MAPPING, 
    DEFAULT_VALUES,
    COMPLIANCE_CONFIG,
    RISK_CONFIG
)
from market_data import get_market_data, get_carbon_prices, get_regulatory_news
from agent_manager import agent_manager, process_company_with_ai, process_batch_with_ai, start_ai_simulation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages data pipeline, caching, and integration"""
    
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.cache = {}
        
    def load_portfolio_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load portfolio data from database with enhanced error handling
        Returns: (main_df, trades_df, financed_df)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load core data
            submissions = pd.read_sql("SELECT adsh, sector AS company_name, sic FROM submissions", conn)
            lending = pd.read_sql("SELECT adsh, total_loans_outstanding_usd, total_exposure_usd, non_performing_loans_pct, credit_rating, default_probability_pct FROM lending_data", conn)
            emissions = pd.read_sql("SELECT adsh, total_co2e, scope1_co2e, scope2_co2e, scope3_co2e, credits_purchased_tonnes, uncertainty FROM emissions_estimates", conn)
            exposures = pd.read_sql("SELECT adsh, carbon_adjusted_exposure FROM carbon_adjusted_exposures", conn)
            trades = pd.read_sql("SELECT adsh, company_name, trade_timestamp, credits_traded_tonnes, trade_price_usd FROM carbon_trades ORDER BY trade_timestamp DESC", conn)
            
            # Load financed emissions with fallback
            try:
                financed = pd.read_sql("SELECT adsh, company_name, sector, baseline_finance_total_co2e, current_finance_total_co2e, bank_share_pct, total_debt_usd, delta_co2e, update_timestamp FROM financed_emissions ORDER BY update_timestamp DESC", conn)
            except Exception as e:
                logger.warning(f"financed_emissions table missing or incorrect schema: {e}")
                financed = pd.DataFrame(columns=['adsh', 'company_name', 'sector', 'baseline_finance_total_co2e', 'current_finance_total_co2e', 'bank_share_pct', 'total_debt_usd', 'delta_co2e', 'update_timestamp'])
            
            # Process and merge data
            main_df = self._process_portfolio_data(submissions, lending, emissions, exposures)
            trades_df = self._process_trades_data(trades)
            financed_df = self._process_financed_data(financed)
            
            conn.close()
            
            logger.info(f"Loaded {len(main_df)} companies, {len(trades_df)} trades, {len(financed_df)} financed emissions")
            return main_df, trades_df, financed_df
            
        except Exception as e:
            logger.error(f"Error loading portfolio data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def _process_portfolio_data(self, submissions: pd.DataFrame, lending: pd.DataFrame, 
                               emissions: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
        """Process and merge portfolio data"""
        try:
            # Add default values for emissions
            for col in ['credits_purchased_tonnes', 'scope1_co2e', 'scope2_co2e', 'scope3_co2e', 'uncertainty']:
                if col not in emissions.columns:
                    emissions[col] = DEFAULT_VALUES['emissions'].get(col, 0)
                else:
                    emissions[col] = emissions[col].fillna(DEFAULT_VALUES['emissions'].get(col, 0))
            
            # Calculate total CO2e if missing
            if 'total_co2e' not in emissions.columns:
                emissions['total_co2e'] = emissions['scope1_co2e'] + emissions['scope2_co2e'] + emissions['scope3_co2e']
                logger.info("Calculated total_co2e from scope emissions")
            
            # Merge data
            df = submissions.merge(lending, on='adsh', how='left').merge(emissions, on='adsh', how='left').merge(exposures, on='adsh', how='left')
            
            # Add sector mapping
            df['sector'] = df['sic'].map(SECTOR_MAPPING).fillna('Other')
            
            # Add calculated fields
            df = self._add_calculated_fields(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing portfolio data: {e}")
            return pd.DataFrame()
    
    def _process_trades_data(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Process trades data"""
        try:
            if trades.empty:
                return trades
            
            # Convert timestamp
            trades['trade_timestamp'] = pd.to_datetime(trades['trade_timestamp'], errors='coerce')
            
            # Add calculated fields
            trades['price_per_tonne'] = trades['trade_price_usd'] / trades['credits_traded_tonnes'].replace(0, np.nan)
            trades['trade_date'] = trades['trade_timestamp'].dt.date
            trades['trade_hour'] = trades['trade_timestamp'].dt.hour
            
            return trades
            
        except Exception as e:
            logger.error(f"Error processing trades data: {e}")
            return trades
    
    def _process_financed_data(self, financed: pd.DataFrame) -> pd.DataFrame:
        """Process financed emissions data"""
        try:
            if financed.empty:
                return financed
            
            # Convert timestamp
            financed['update_timestamp'] = pd.to_datetime(financed['update_timestamp'], errors='coerce')
            
            # Add calculated fields
            financed['offset_ratio'] = financed['current_finance_total_co2e'] / financed['baseline_finance_total_co2e'].replace(0, np.nan)
            financed['compliance_status'] = financed['offset_ratio'].apply(self._get_compliance_status)
            
            return financed
            
        except Exception as e:
            logger.error(f"Error processing financed data: {e}")
            return financed
    
    def _add_calculated_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated fields to portfolio data"""
        try:
            # Risk metrics
            df['risk_score'] = self._calculate_risk_score(df)
            df['carbon_intensity'] = df['total_co2e'] / df['total_exposure_usd'].replace(0, np.nan)
            
            # Compliance metrics
            df['offset_ratio'] = df['credits_purchased_tonnes'] / df['total_co2e'].replace(0, np.nan)
            df['compliance_status'] = df['offset_ratio'].apply(self._get_compliance_status)
            
            # Performance metrics
            df['exposure_reduction_pct'] = ((df['total_exposure_usd'] - df['carbon_adjusted_exposure']) / df['total_exposure_usd'] * 100).fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding calculated fields: {e}")
            return df
    
    def _calculate_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite risk score"""
        try:
            # Normalize risk factors (0-1 scale)
            default_risk = df['default_probability_pct'].fillna(0) / 100
            npl_risk = df['non_performing_loans_pct'].fillna(0) / 100
            
            # Calculate carbon intensity if not present
            if 'carbon_intensity' not in df.columns:
                df['carbon_intensity'] = df['total_co2e'] / df['total_exposure_usd'].replace(0, np.nan)
            
            carbon_risk = df['carbon_intensity'].fillna(0)
            if carbon_risk.max() > 0:
                carbon_risk = carbon_risk / carbon_risk.quantile(0.95)
            
            # Weighted composite score
            risk_score = (default_risk * 0.4 + npl_risk * 0.3 + carbon_risk * 0.3)
            return risk_score.fillna(0)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return pd.Series([0] * len(df))
    
    def _get_compliance_status(self, offset_ratio: float) -> str:
        """Get compliance status based on offset ratio"""
        if pd.isna(offset_ratio):
            return 'Unknown'
        elif offset_ratio >= COMPLIANCE_CONFIG['offset_thresholds']['target_offset_ratio']:
            return 'Compliant'
        elif offset_ratio >= COMPLIANCE_CONFIG['offset_thresholds']['minimum_offset_ratio']:
            return 'At Risk'
        elif offset_ratio >= COMPLIANCE_CONFIG['offset_thresholds']['warning_threshold']:
            return 'Warning'
        else:
            return 'Non-Compliant'
    
    def get_market_overview(self) -> Dict:
        """Get comprehensive market overview with external data"""
        try:
            market_data = get_market_data()
            return market_data
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}
    
    def get_portfolio_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate portfolio-level metrics"""
        try:
            if df.empty:
                return {}
            
            metrics = {
                'total_exposure': df['total_exposure_usd'].sum(),
                'carbon_adjusted_exposure': df['carbon_adjusted_exposure'].sum(),
                'total_emissions': df['total_co2e'].sum(),
                'total_credits_purchased': df['credits_purchased_tonnes'].sum(),
                'avg_risk_score': df['risk_score'].mean(),
                'compliance_rate': (df['compliance_status'] == 'Compliant').mean(),
                'sector_diversification': df['sector'].nunique(),
                'companies_count': len(df)
            }
            
            # Add percentage changes if we have historical data
            if 'exposure_reduction_pct' in df.columns:
                metrics['avg_exposure_reduction'] = df['exposure_reduction_pct'].mean()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def get_risk_metrics(self, df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """Calculate risk metrics including VaR"""
        try:
            risk_metrics = {}
            
            if not df.empty:
                # Portfolio risk metrics
                risk_metrics['portfolio_var_95'] = self._calculate_var(df, confidence=0.95)
                risk_metrics['portfolio_var_99'] = self._calculate_var(df, confidence=0.99)
                risk_metrics['max_sector_exposure'] = df.groupby('sector')['total_exposure_usd'].sum().max()
                risk_metrics['concentration_risk'] = self._calculate_concentration_risk(df)
            
            if not trades_df.empty:
                # Trading risk metrics
                risk_metrics['trading_var'] = self._calculate_trading_var(trades_df)
                risk_metrics['price_volatility'] = trades_df['price_per_tonne'].std()
                risk_metrics['volume_volatility'] = trades_df['credits_traded_tonnes'].std()
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_var(self, df: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            # Simple VaR calculation based on exposure and default probability
            exposure = df['total_exposure_usd'].fillna(0)
            default_prob = df['default_probability_pct'].fillna(0) / 100
            
            # Simulate potential losses
            potential_losses = exposure * default_prob
            
            # Calculate VaR
            var = np.percentile(potential_losses, (1 - confidence) * 100)
            return var
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_concentration_risk(self, df: pd.DataFrame) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        try:
            sector_exposures = df.groupby('sector')['total_exposure_usd'].sum()
            total_exposure = sector_exposures.sum()
            
            if total_exposure == 0:
                return 0.0
            
            # Calculate Herfindahl index
            market_shares = sector_exposures / total_exposure
            hhi = (market_shares ** 2).sum()
            
            return hhi
            
        except Exception as e:
            logger.error(f"Error calculating concentration risk: {e}")
            return 0.0
    
    def _calculate_trading_var(self, trades_df: pd.DataFrame) -> float:
        """Calculate trading VaR"""
        try:
            if trades_df.empty:
                return 0.0
            
            # Calculate daily P&L
            daily_pnl = trades_df.groupby('trade_date')['trade_price_usd'].sum()
            
            if len(daily_pnl) < 2:
                return 0.0
            
            # Calculate VaR
            var = np.percentile(daily_pnl, 5)  # 5th percentile
            return abs(var)
            
        except Exception as e:
            logger.error(f"Error calculating trading VaR: {e}")
            return 0.0
    
    def get_compliance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate compliance metrics"""
        try:
            if df.empty:
                return {}
            
            compliance_metrics = {
                'total_companies': len(df),
                'compliant_companies': (df['compliance_status'] == 'Compliant').sum(),
                'at_risk_companies': (df['compliance_status'] == 'At Risk').sum(),
                'warning_companies': (df['compliance_status'] == 'Warning').sum(),
                'non_compliant_companies': (df['compliance_status'] == 'Non-Compliant').sum(),
                'compliance_rate': (df['compliance_status'] == 'Compliant').mean(),
                'avg_offset_ratio': df['offset_ratio'].mean(),
                'total_offset_shortfall': df[df['offset_ratio'] < 0.8]['total_co2e'].sum()
            }
            
            return compliance_metrics
            
        except Exception as e:
            logger.error(f"Error calculating compliance metrics: {e}")
            return {}
    
    def trigger_ai_workflow(self, companies: List[Dict] = None, max_companies: int = 5) -> List[Dict]:
        """Trigger AI agent workflow for companies"""
        try:
            if companies is None:
                # Get companies that need AI processing
                conn = sqlite3.connect(self.db_path)
                companies_df = pd.read_sql("SELECT adsh, sector AS company_name FROM submissions LIMIT ?", conn, params=[max_companies])
                conn.close()
                companies = companies_df.to_dict('records')
            
            logger.info(f"Triggering AI workflow for {len(companies)} companies")
            results = process_batch_with_ai(companies, max_companies)
            
            # Save results to database
            if results:
                conn = sqlite3.connect(self.db_path)
                pd.DataFrame(results).to_sql('ai_workflow_results', conn, if_exists='append', index=False)
                conn.close()
                logger.info(f"AI workflow results saved for {len(results)} companies")
            
            return results
            
        except Exception as e:
            logger.error(f"Error triggering AI workflow: {e}")
            return []
    
    def start_ai_simulation(self):
        """Start AI-powered portfolio simulation"""
        try:
            start_ai_simulation()
            logger.info("AI simulation started successfully")
        except Exception as e:
            logger.error(f"Error starting AI simulation: {e}")
    
    def get_ai_workflow_status(self) -> Dict:
        """Get status of AI workflows"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create ai_workflow_results table if it doesn't exist
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ai_workflow_results (
                    adsh TEXT,
                    company_name TEXT,
                    status TEXT,
                    carbon_adjusted_exposure FLOAT,
                    credits_traded_tonnes FLOAT,
                    trade_price_usd FLOAT,
                    update_timestamp TEXT,
                    PRIMARY KEY (adsh, update_timestamp)
                )
            ''')
            conn.commit()
            
            # Get recent AI workflow results
            recent_results = pd.read_sql("""
                SELECT status, COUNT(*) as count 
                FROM ai_workflow_results 
                WHERE datetime(update_timestamp) > datetime('now', '-1 hour')
                GROUP BY status
            """, conn)
            
            # Get simulation status
            simulation_status = "running" if agent_manager.is_running else "stopped"
            
            conn.close()
            
            return {
                'simulation_status': simulation_status,
                'recent_workflows': recent_results.to_dict('records') if not recent_results.empty else [],
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting AI workflow status: {e}")
            return {'simulation_status': 'unknown', 'error': str(e)}
    
    def should_refresh_data(self, data_type: str) -> bool:
        """Check if data should be refreshed"""
        if SESSION_KEYS['last_refresh'] not in st.session_state:
            return True
        
        last_refresh = datetime.fromisoformat(st.session_state[SESSION_KEYS['last_refresh']])
        refresh_interval = REFRESH_INTERVALS.get(data_type, 60)
        
        return (datetime.now() - last_refresh).total_seconds() > refresh_interval
    
    def cache_data(self, data_type: str, data: Any):
        """Cache data in session state"""
        if data_type not in st.session_state:
            st.session_state[data_type] = {}
        
        st.session_state[data_type] = data
        st.session_state[SESSION_KEYS['last_refresh']] = datetime.now().isoformat()
    
    def get_cached_data(self, data_type: str) -> Any:
        """Get cached data from session state"""
        return st.session_state.get(data_type, None)

# Global instance
data_manager = DataManager()

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data with caching"""
    if data_manager.should_refresh_data('portfolio_data'):
        main_df, trades_df, financed_df = data_manager.load_portfolio_data()
        data_manager.cache_data('portfolio_data', (main_df, trades_df, financed_df))
        return main_df, trades_df, financed_df
    else:
        cached_data = data_manager.get_cached_data('portfolio_data')
        if cached_data:
            return cached_data
        else:
            return data_manager.load_portfolio_data()

def get_market_overview(force_refresh: bool = True) -> Dict:
    """Get market overview with caching"""
    if force_refresh or data_manager.should_refresh_data('market_data'):
        market_data = data_manager.get_market_overview()
        data_manager.cache_data('market_data', market_data)
        return market_data
    else:
        cached_data = data_manager.get_cached_data('market_data')
        if cached_data:
            return cached_data
        else:
            return data_manager.get_market_overview()

def get_portfolio_metrics(df: pd.DataFrame) -> Dict:
    """Get portfolio metrics"""
    return data_manager.get_portfolio_metrics(df)

def get_risk_metrics(df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
    """Get risk metrics"""
    return data_manager.get_risk_metrics(df, trades_df)

def get_compliance_metrics(df: pd.DataFrame) -> Dict:
    """Get compliance metrics"""
    return data_manager.get_compliance_metrics(df)

def trigger_ai_workflow(companies: List[Dict] = None, max_companies: int = 5) -> List[Dict]:
    """Trigger AI agent workflow for companies"""
    return data_manager.trigger_ai_workflow(companies, max_companies)

def start_ai_simulation():
    """Start AI-powered portfolio simulation"""
    data_manager.start_ai_simulation()

def get_ai_workflow_status() -> Dict:
    """Get status of AI workflows"""
    return data_manager.get_ai_workflow_status()
