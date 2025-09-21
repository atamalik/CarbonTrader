
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
import time
import logging
from datetime import datetime

# Import our new modular components
from config import UI_CONFIG, SESSION_KEYS
from data_manager import load_data, get_market_overview, get_portfolio_metrics, get_risk_metrics, get_compliance_metrics
from ui_components import (
    create_sidebar_navigation, 
    create_header, 
    create_market_overview_panel,
    create_portfolio_panel,
    create_trading_panel,
    create_risk_panel,
    create_research_panel,
    create_ai_agents_panel
)

# Setup logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('frontend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(
    page_title=UI_CONFIG['page_title'], 
    layout=UI_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# Initialize session state
if SESSION_KEYS['last_refresh'] not in st.session_state:
    st.session_state[SESSION_KEYS['last_refresh']] = datetime.now().isoformat()

# Load data using new modular system
logger.info("Loading data using modular data manager")
df, trades_df, financed_df = load_data()
market_data = get_market_overview(force_refresh=True)

# Main Application Logic
def main():
    """Main application function"""
    
    # Create sidebar navigation
    selected_tab = create_sidebar_navigation()
    
    # Update navigation to include AI Agents
    if selected_tab == "AI Agents":
        selected_tab = "AI Agents"
    
    # Create main header
    create_header()
    
    # Check emissions data availability
    emissions_count = df['total_co2e'].notnull().sum() if 'total_co2e' in df.columns else 0
    if emissions_count < len(df) * 0.1:
        st.warning(f"Only {emissions_count} of {len(df)} companies have emissions data. Some graphs may be limited.")
        logger.warning(f"Low emissions data: {emissions_count}/{len(df)} companies")
    
    # Main content area based on selected tab
    if selected_tab == "Markets":
        st.header("📊 Market Overview")
        create_market_overview_panel(market_data)
        
    elif selected_tab == "Portfolio":
        st.header("💼 Portfolio Management")
        create_portfolio_panel(df, trades_df)
        
    elif selected_tab == "Trading":
        st.header("⚡ Trading Desk")
        create_trading_panel(trades_df)
        
    elif selected_tab == "Risk & Compliance":
        st.header("⚠️ Risk & Compliance")
        create_risk_panel(df, trades_df)
        
    elif selected_tab == "Research & Insights":
        st.header("🔬 Research & Insights")
        create_research_panel(market_data)
        
    elif selected_tab == "AI Agents":
        st.header("🤖 AI Agents & Workflows")
        create_ai_agents_panel()
        
    elif selected_tab == "Settings":
        st.header("⚙️ Settings")
        create_settings_panel()
    
    # Auto-refresh logic for real-time updates
    auto_refresh_enabled = st.session_state.get('auto_refresh', True)
    auto_refresh_prices = st.session_state.get('auto_refresh_prices', True)
    refresh_interval = st.session_state.get('price_refresh_interval', 10)
    
    # Only auto-refresh if we're on the Markets tab and auto-refresh is enabled
    if selected_tab == "Markets" and auto_refresh_enabled and auto_refresh_prices:
        # Show refresh indicator
        st.markdown(f"🔄 **Auto-refreshing every {refresh_interval} seconds**")
        time.sleep(refresh_interval)
        st.rerun()
    elif auto_refresh_enabled:
        # General auto-refresh for other tabs
        time.sleep(5)
        st.rerun()

def create_settings_panel():
    """Create settings panel"""
    st.subheader("System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Sources")
        st.checkbox("Enable External Data", value=True, key="enable_external_data")
        st.checkbox("Enable Real-time Updates", value=True, key="enable_realtime")
        st.slider("Refresh Interval (seconds)", 10, 300, 60, key="refresh_interval")
    
    with col2:
        st.subheader("Display Options")
        st.selectbox("Theme", ["Light", "Dark"], key="theme_select")
        st.selectbox("Default View", ["Markets", "Portfolio", "Trading"], key="default_view")
        st.checkbox("Show Debug Info", value=False, key="show_debug")
    
    st.subheader("System Information")
    st.info(f"Last Data Refresh: {st.session_state.get(SESSION_KEYS['last_refresh'], 'Never')}")
    st.info(f"Database Records: {len(df)} companies, {len(trades_df)} trades")
    
    if st.button("🔄 Force Refresh All Data"):
        # Clear cache and refresh
        for key in [SESSION_KEYS['market_data'], SESSION_KEYS['portfolio_data']]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Run the main application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")