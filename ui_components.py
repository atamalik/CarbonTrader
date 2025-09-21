"""
UI Components Module for ACCMN
Reusable UI components for the carbon trading desk
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

# Import configuration
from config import (
    UI_CONFIG, 
    TRADING_CONFIG, 
    RISK_CONFIG, 
    COMPLIANCE_CONFIG,
    SESSION_KEYS
)
from data_manager import trigger_ai_workflow, start_ai_simulation, get_ai_workflow_status

class UIComponents:
    """Reusable UI components for the trading desk"""
    
    def __init__(self):
        self.color_sequence = UI_CONFIG['charts']['color_sequence']
        self.theme = UI_CONFIG['theme']
    
    def create_sidebar_navigation(self) -> str:
        """Create sidebar navigation and return selected tab"""
        with st.sidebar:
            st.title("ACCMN Trading Desk")
            st.markdown("---")
            
            # Navigation tabs
            tab = st.radio(
                "Navigation",
                ["Markets", "Portfolio", "Trading", "Risk & Compliance", "Research & Insights", "AI Agents", "Settings"],
                key="main_navigation"
            )
            
            st.markdown("---")
            
            # System status
            self._display_system_status()
            
            # Quick stats
            self._display_quick_stats()
            
            st.markdown("---")
            
            # User preferences
            self._display_user_preferences()
        
        return tab
    
    def _display_system_status(self):
        """Display system status indicators"""
        st.subheader("System Status")
        
        # Data freshness
        if SESSION_KEYS['last_refresh'] in st.session_state:
            last_refresh = datetime.fromisoformat(st.session_state[SESSION_KEYS['last_refresh']])
            time_diff = (datetime.now() - last_refresh).total_seconds()
            
            if time_diff < 60:
                st.success("🟢 Data Fresh")
            elif time_diff < 300:
                st.warning("🟡 Data Stale")
            else:
                st.error("🔴 Data Outdated")
        else:
            st.info("🟡 Initializing...")
    
    def _display_quick_stats(self):
        """Display quick portfolio stats"""
        st.subheader("Quick Stats")
        
        # These would be populated from actual data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Exposure", "$2.5B", "5.2%")
        with col2:
            st.metric("Carbon Credits", "1.2M tCO2e", "12.3%")
    
    def _display_user_preferences(self):
        """Display user preferences"""
        st.subheader("Preferences")
        
        # Theme selection
        theme = st.selectbox("Theme", ["Light", "Dark"], key="theme_select")
        
        # Refresh interval
        refresh_interval = st.slider("Refresh Interval (s)", 10, 300, 60, key="sidebar_refresh_interval")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh", value=True, key="auto_refresh")
        
        # AI Agent Controls
        st.subheader("AI Agent Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Start AI Simulation", key="start_ai_sim"):
                start_ai_simulation()
                st.success("AI simulation started!")
        
        with col2:
            if st.button("⚡ Trigger AI Workflow", key="trigger_ai_workflow"):
                with st.spinner("Running AI workflow..."):
                    results = trigger_ai_workflow(max_companies=3)
                    st.success(f"AI workflow completed for {len(results)} companies!")
        
        # AI Status
        ai_status = get_ai_workflow_status()
        if ai_status.get('simulation_status') == 'running':
            st.success("🟢 AI Simulation: Running")
        else:
            st.info("🟡 AI Simulation: Stopped")
    
    def create_header(self):
        """Create main header with alerts and system info"""
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.title("ACCMN: Sustainable Portfolio Management")
            st.markdown("**Powered by Autonomous AI Agents** | Optimize your lending portfolio with ESG insights")
        
        with col2:
            st.metric("System Time", datetime.now().strftime("%H:%M:%S"))
        
        with col3:
            st.metric("Market Status", "🟢 Open")
        
        with col4:
            if st.button("🔄 Refresh", key="header_refresh"):
                st.rerun()
        
        # Alerts panel
        self._display_alerts()
    
    def _display_alerts(self):
        """Display system alerts"""
        # Check for price alerts
        if SESSION_KEYS['market_data'] in st.session_state:
            market_data = st.session_state[SESSION_KEYS['market_data']]
            if 'prices' in market_data:
                for credit_type, price_data in market_data['prices'].items():
                    if 'change_pct' in price_data and abs(price_data['change_pct']) > 5:
                        if price_data['change_pct'] > 0:
                            st.success(f"🚀 {credit_type} up {price_data['change_pct']:.1f}%")
                        else:
                            st.error(f"📉 {credit_type} down {abs(price_data['change_pct']):.1f}%")
    
    def create_market_overview_panel(self, market_data: Dict) -> None:
        """Create market overview panel"""
        with st.expander("📊 Market Overview", expanded=True):
            if not market_data:
                st.warning("No market data available")
                return
            
            # Live prices
            if 'prices' in market_data:
                self._display_live_prices(market_data['prices'])
            
            # Market sentiment
            if 'sentiment' in market_data:
                self._display_market_sentiment(market_data['sentiment'])
            
            # Regulatory news
            if 'news' in market_data:
                self._display_regulatory_news(market_data['news'])
    
    def _display_live_prices(self, prices: Dict):
        """Display live carbon prices as a compact ticker with real-time updates"""
        st.subheader("📈 Live Carbon Prices Ticker")
        
        if not prices:
            st.info("No price data available")
            return
        
        # Add auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Real-time Market Data** - Prices update automatically")
        with col2:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=True, key="auto_refresh_prices")
        with col3:
            refresh_interval = st.selectbox("Update Interval", [5, 10, 30, 60], index=1, key="price_refresh_interval")
            st.markdown(f"*Every {refresh_interval}s*")
        
        # Create a real-time updating ticker
        ticker_container = st.container()
        
        with ticker_container:
            # Create a compact ticker using Streamlit columns
            st.markdown("""
            <div style="
                background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                animation: pulse 2s infinite;
            ">
            """, unsafe_allow_html=True)
            
            # Create columns for each price item
            cols = st.columns(len(prices))
            
            for i, (credit_type, data) in enumerate(prices.items()):
                price = data.get('price', 0)
                change = data.get('change', 0)
                change_pct = data.get('change_pct', 0)
                volume = data.get('volume', 0)
                source = data.get('source', 'Unknown')
                timestamp = data.get('timestamp', '')
                
                # Determine color and arrow based on change
                if change > 0:
                    color = "🟢"  # Green
                    arrow = "▲"
                    change_color = "#00ff88"
                    animation = "glow-green"
                elif change < 0:
                    color = "🔴"  # Red
                    arrow = "▼"
                    change_color = "#ff4444"
                    animation = "glow-red"
                else:
                    color = "⚪"  # White
                    arrow = "●"
                    change_color = "#ffffff"
                    animation = "glow-neutral"
                
                with cols[i]:
                    st.markdown(f"""
                    <div style="
                        background: rgba(255,255,255,0.1);
                        padding: 8px;
                        border-radius: 4px;
                        text-align: center;
                        margin: 2px;
                        border: 2px solid {change_color};
                        animation: {animation} 1s ease-in-out;
                    ">
                        <strong style="font-size: 0.9em;">{credit_type}</strong><br>
                        <span style="font-size: 1.1em; font-weight: bold; color: {change_color};">${price:.2f}</span><br>
                        <span style="font-size: 0.8em; color: {change_color};">{color} {arrow} {change:+.2f}</span><br>
                        <span style="font-size: 0.7em; color: {change_color};">({change_pct:+.2f}%)</span><br>
                        <span style="font-size: 0.6em; opacity: 0.8;">Vol: {volume:,}</span><br>
                        <span style="font-size: 0.5em; opacity: 0.6;">{source}</span><br>
                        <span style="font-size: 0.4em; opacity: 0.5;">{timestamp[11:19] if timestamp else ''}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add CSS animations for visual effects
            st.markdown("""
            <style>
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.8; }
                100% { opacity: 1; }
            }
            @keyframes glow-green {
                0% { box-shadow: 0 0 5px #00ff88; }
                50% { box-shadow: 0 0 20px #00ff88, 0 0 30px #00ff88; }
                100% { box-shadow: 0 0 5px #00ff88; }
            }
            @keyframes glow-red {
                0% { box-shadow: 0 0 5px #ff4444; }
                50% { box-shadow: 0 0 20px #ff4444, 0 0 30px #ff4444; }
                100% { box-shadow: 0 0 5px #ff4444; }
            }
            @keyframes glow-neutral {
                0% { box-shadow: 0 0 5px #ffffff; }
                50% { box-shadow: 0 0 15px #ffffff; }
                100% { box-shadow: 0 0 5px #ffffff; }
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Add real-time update indicator
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="
            background: rgba(0,255,136,0.1);
            border: 1px solid #00ff88;
            border-radius: 4px;
            padding: 8px;
            margin: 10px 0;
            text-align: center;
        ">
            <span style="color: #00ff88; font-weight: bold;">🔄 LIVE</span>
            <span style="color: white; margin-left: 10px;">Last Updated: {current_time}</span>
            <span style="color: #00ff88; margin-left: 10px;">●</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Add data source indicator
        data_types = [data.get('data_type', 'live') for data in prices.values()]
        if 'simulated' in data_types:
            st.info("ℹ️ **Data Status**: Currently showing simulated prices with realistic market movements. In production, this would connect to live APIs from ICE, ARB, and other exchanges.")
        else:
            st.success("✅ **Data Status**: Live prices from official exchanges")
        
        # Add a compact data table for detailed view
        with st.expander("📊 Detailed Price Data", expanded=False):
            price_data = []
            for credit_type, data in prices.items():
                price_data.append({
                    'Credit Type': credit_type,
                    'Price ($)': f"${data.get('price', 0):.2f}",
                    'Change ($)': f"${data.get('change', 0):.2f}",
                    'Change (%)': f"{data.get('change_pct', 0):.2f}%",
                    'Volume': f"{data.get('volume', 0):,}",
                    'Source': data.get('source', 'Unknown'),
                    'Exchange': data.get('exchange', 'N/A'),
                    'Contract': data.get('contract', 'N/A'),
                    'Last Update': data.get('timestamp', 'N/A')[:19] if data.get('timestamp') else 'N/A'
                })
            
            df_prices = pd.DataFrame(price_data)
            st.dataframe(df_prices, width='stretch', hide_index=True)
            
            # Add simulation details for simulated data
            simulated_prices = [k for k, v in prices.items() if v.get('data_type') == 'simulated']
            if simulated_prices:
                st.subheader("🎯 Dynamic Simulation Details")
                st.markdown("**Realistic market simulation with:**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **📈 Market Dynamics:**
                    - Long-term trends (±5%)
                    - Short-term momentum (±3%)
                    - Dynamic volatility (0.5x-2x)
                    - Mean reversion
                    """)
                
                with col2:
                    st.markdown("""
                    **📊 Volume Correlation:**
                    - Higher volume on big moves
                    - Realistic volume ranges
                    - Market condition effects
                    - Time-based updates
                    """)
                
                with col3:
                    st.markdown("""
                    **⏰ Time-Based Updates:**
                    - Continuous price evolution
                    - Realistic market movements
                    - Price history tracking
                    - Professional-grade simulation
                    """)
                
                # Show simulation metrics for each credit type
                for credit_type in simulated_prices:
                    data = prices[credit_type]
                    if 'trend' in data and 'momentum' in data:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(f"{credit_type} Trend", f"{data['trend']:+.2f}%")
                        with col2:
                            st.metric(f"{credit_type} Momentum", f"{data['momentum']:+.2f}%")
                        with col3:
                            st.metric(f"{credit_type} Volatility", f"{data['volatility_factor']:.2f}x")
                        with col4:
                            st.metric(f"{credit_type} Volume", f"{data['volume']:,}")
            
            # Add data source information
            st.markdown("""
            **Data Sources:**
            - **EUA**: ICE Futures Europe (European Union Allowances)
            - **CCA**: California Air Resources Board (California Carbon Allowances)  
            - **RGGI**: Regional Greenhouse Gas Initiative
            - **VCS**: Verified Carbon Standard (Voluntary Market)
            - **Gold Standard**: Gold Standard Foundation (Voluntary Market)
            
            *Note: Prices are updated every 60 seconds with dynamic simulation. In production, this would connect to real-time APIs.*
            """)
    
    def _display_market_sentiment(self, sentiment: Dict):
        """Display market sentiment indicators"""
        st.subheader("Market Sentiment")
        
        if not sentiment:
            st.info("No sentiment data available")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_score = sentiment.get('sentiment_score', 0)
            if sentiment_score > 0.3:
                st.success(f"😊 Bullish ({sentiment_score:.2f})")
            elif sentiment_score < -0.3:
                st.error(f"😞 Bearish ({sentiment_score:.2f})")
            else:
                st.info(f"😐 Neutral ({sentiment_score:.2f})")
        
        with col2:
            fear_greed = sentiment.get('fear_greed_index', 50)
            if fear_greed > 70:
                st.warning(f"😰 Greed ({fear_greed})")
            elif fear_greed < 30:
                st.info(f"😨 Fear ({fear_greed})")
            else:
                st.success(f"😌 Neutral ({fear_greed})")
        
        with col3:
            overall = sentiment.get('overall_sentiment', 'neutral')
            if overall == 'bullish':
                st.success("📈 Bullish")
            elif overall == 'bearish':
                st.error("📉 Bearish")
            else:
                st.info("📊 Neutral")
    
    def _display_regulatory_news(self, news: List[Dict]):
        """Display regulatory news ticker"""
        st.subheader("Regulatory News")
        
        if not news:
            st.info("No news available")
            return
        
        # Display top 5 news items
        for i, item in enumerate(news[:5]):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{item.get('title', 'No Title')}**")
                    st.write(item.get('summary', 'No summary available'))
                with col2:
                    impact = item.get('impact', 'low')
                    if impact == 'high':
                        st.error(f"🔴 {impact.title()}")
                    elif impact == 'medium':
                        st.warning(f"🟡 {impact.title()}")
                    else:
                        st.info(f"🟢 {impact.title()}")
                
                if i < len(news) - 1:
                    st.markdown("---")
    
    def create_portfolio_panel(self, df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
        """Create portfolio overview panel"""
        with st.expander("💼 Portfolio Overview", expanded=True):
            if df.empty:
                st.warning("No portfolio data available")
                return
            
            # Portfolio metrics
            self._display_portfolio_metrics(df)
            
            # Holdings dashboard
            self._display_holdings_dashboard(df, trades_df)
            
            # Performance charts
            self._display_performance_charts(df)
    
    def _display_portfolio_metrics(self, df: pd.DataFrame):
        """Display portfolio-level metrics"""
        st.subheader("Portfolio Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_exposure = df['total_exposure_usd'].sum()
            st.metric("Total Exposure", f"${total_exposure:,.0f}")
        
        with col2:
            carbon_adjusted = df['carbon_adjusted_exposure'].sum()
            st.metric("Carbon Adjusted", f"${carbon_adjusted:,.0f}")
        
        with col3:
            total_emissions = df['total_co2e'].sum()
            st.metric("Total Emissions", f"{total_emissions:,.0f} tCO2e")
        
        with col4:
            total_credits = df['credits_purchased_tonnes'].sum()
            st.metric("Credits Purchased", f"{total_credits:,.0f} tCO2e")
    
    def _display_holdings_dashboard(self, df: pd.DataFrame, trades_df: pd.DataFrame):
        """Display holdings dashboard"""
        st.subheader("Holdings Dashboard")
        
        # Sector exposure
        sector_exposure = df.groupby('sector')['carbon_adjusted_exposure'].sum().reset_index()
        if not sector_exposure.empty:
            fig = px.bar(sector_exposure, x='sector', y='carbon_adjusted_exposure', 
                        title="Exposure by Sector", color='sector',
                        color_discrete_sequence=self.color_sequence)
            st.plotly_chart(fig, width='stretch')
        
        # Credit rating distribution
        rating_dist = df.groupby('credit_rating')['total_loans_outstanding_usd'].sum().reset_index()
        if not rating_dist.empty:
            fig = px.pie(rating_dist, values='total_loans_outstanding_usd', names='credit_rating',
                        title="Loans by Credit Rating")
            st.plotly_chart(fig, width='stretch')
    
    def _display_performance_charts(self, df: pd.DataFrame):
        """Display performance charts"""
        st.subheader("Performance Analysis")
        
        # Risk vs Return scatter
        if 'risk_score' in df.columns and 'default_probability_pct' in df.columns:
            fig = px.scatter(df, x='risk_score', y='default_probability_pct', 
                           color='sector', size='total_exposure_usd',
                           hover_data=['company_name'], title="Risk vs Return Analysis")
            st.plotly_chart(fig, width='stretch')
        
        # Compliance status
        if 'compliance_status' in df.columns:
            compliance_counts = df['compliance_status'].value_counts()
            fig = px.bar(x=compliance_counts.index, y=compliance_counts.values,
                        title="Compliance Status Distribution")
            st.plotly_chart(fig, width='stretch')
    
    def create_trading_panel(self, trades_df: pd.DataFrame) -> None:
        """Create trading execution panel"""
        with st.expander("⚡ Trading Desk", expanded=True):
            # Trade execution form
            self._display_trade_execution_form()
            
            # Order book
            self._display_order_book(trades_df)
            
            # Trade history
            self._display_trade_history(trades_df)
    
    def _display_trade_execution_form(self):
        """Display trade execution form"""
        st.subheader("Trade Execution")
        
        with st.form("trade_execution"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                credit_type = st.selectbox("Credit Type", TRADING_CONFIG['default_credit_types'])
                quantity = st.number_input("Quantity (tonnes)", min_value=1, max_value=100000, value=1000)
            
            with col2:
                price = st.number_input("Price per tonne ($)", min_value=0.01, max_value=1000.0, value=30.0)
                order_type = st.selectbox("Order Type", ["Market", "Limit", "Stop"])
            
            with col3:
                side = st.selectbox("Side", ["Buy", "Sell"])
                validity = st.selectbox("Validity", ["Day", "GTC", "IOC", "FOK"])
            
            submitted = st.form_submit_button("Execute Trade", type="primary")
            
            if submitted:
                st.success(f"Trade executed: {side} {quantity} tonnes of {credit_type} at ${price:.2f}/tonne")
    
    def _display_order_book(self, trades_df: pd.DataFrame):
        """Display order book"""
        st.subheader("Order Book")
        
        if trades_df.empty:
            st.info("No trades available")
            return
        
        # Recent trades
        recent_trades = trades_df.head(10)
        
        gb = GridOptionsBuilder.from_dataframe(recent_trades)
        gb.configure_default_column(filter=True, sortable=True, resizable=True)
        gb.configure_pagination(paginationAutoPageSize=True)
        grid_options = gb.build()
        
        AgGrid(recent_trades, grid_options=grid_options, height=300, theme='streamlit')
    
    def _display_trade_history(self, trades_df: pd.DataFrame):
        """Display trade history charts"""
        st.subheader("Trade History")
        
        if trades_df.empty:
            st.info("No trade history available")
            return
        
        # Trade volume over time
        if 'trade_timestamp' in trades_df.columns:
            trades_df['trade_date'] = pd.to_datetime(trades_df['trade_timestamp']).dt.date
            daily_volume = trades_df.groupby('trade_date')['credits_traded_tonnes'].sum().reset_index()
            
            fig = px.line(daily_volume, x='trade_date', y='credits_traded_tonnes',
                         title="Daily Trade Volume")
            st.plotly_chart(fig, width='stretch')
    
    def create_risk_panel(self, df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
        """Create risk and compliance panel"""
        with st.expander("⚠️ Risk & Compliance", expanded=True):
            # Risk metrics
            self._display_risk_metrics(df, trades_df)
            
            # Compliance tracking
            self._display_compliance_tracking(df)
            
            # Exposure analysis
            self._display_exposure_analysis(df)
    
    def _display_risk_metrics(self, df: pd.DataFrame, trades_df: pd.DataFrame):
        """Display risk metrics"""
        st.subheader("Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'risk_score' in df.columns:
                avg_risk = df['risk_score'].mean()
                st.metric("Average Risk Score", f"{avg_risk:.3f}")
        
        with col2:
            if not trades_df.empty and 'price_per_tonne' in trades_df.columns:
                volatility = trades_df['price_per_tonne'].std()
                st.metric("Price Volatility", f"${volatility:.2f}")
        
        with col3:
            if 'default_probability_pct' in df.columns:
                max_default = df['default_probability_pct'].max()
                st.metric("Max Default Risk", f"{max_default:.1f}%")
    
    def _display_compliance_tracking(self, df: pd.DataFrame):
        """Display compliance tracking"""
        st.subheader("Compliance Tracking")
        
        if 'compliance_status' in df.columns:
            compliance_counts = df['compliance_status'].value_counts()
            
            # Compliance progress
            total_companies = len(df)
            compliant_companies = compliance_counts.get('Compliant', 0)
            compliance_rate = (compliant_companies / total_companies) * 100
            
            st.progress(compliance_rate / 100)
            st.write(f"Compliance Rate: {compliance_rate:.1f}% ({compliant_companies}/{total_companies})")
            
            # Compliance breakdown
            fig = px.pie(values=compliance_counts.values, names=compliance_counts.index,
                        title="Compliance Status Distribution")
            st.plotly_chart(fig, width='stretch')
    
    def _display_exposure_analysis(self, df: pd.DataFrame):
        """Display exposure analysis"""
        st.subheader("Exposure Analysis")
        
        # Sector exposure heatmap
        if 'sector' in df.columns and 'carbon_adjusted_exposure' in df.columns:
            sector_exposure = df.groupby('sector')['carbon_adjusted_exposure'].sum().reset_index()
            
            # Create heatmap data
            heatmap_data = sector_exposure.set_index('sector')['carbon_adjusted_exposure'].values.reshape(-1, 1)
            heatmap_labels = sector_exposure['sector'].tolist()
            
            fig = px.imshow(heatmap_data, 
                           labels=dict(x="Exposure", y="Sector", color="Amount"),
                           x=['Carbon Adjusted Exposure'],
                           y=heatmap_labels,
                           title="Sector Exposure Heatmap")
            st.plotly_chart(fig, width='stretch')
    
    def create_research_panel(self, market_data: Dict) -> None:
        """Create research and insights panel"""
        with st.expander("🔬 Research & Insights", expanded=True):
            # AI-powered signals
            self._display_ai_signals()
            
            # Market analysis
            self._display_market_analysis(market_data)
            
            # Benchmarking
            self._display_benchmarking()
    
    def _display_ai_signals(self):
        """Display AI-powered market signals"""
        st.subheader("AI-Powered Signals")
        
        # Simulated AI signals
        signals = [
            {"signal": "Carbon Credit Oversupply", "impact": "Bearish", "confidence": 0.85},
            {"signal": "EU CBAM Implementation", "impact": "Bullish", "confidence": 0.92},
            {"signal": "Nature-based Credits Volatility", "impact": "Neutral", "confidence": 0.67}
        ]
        
        for signal in signals:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{signal['signal']}**")
            with col2:
                impact = signal['impact']
                if impact == 'Bullish':
                    st.success(f"📈 {impact}")
                elif impact == 'Bearish':
                    st.error(f"📉 {impact}")
                else:
                    st.info(f"📊 {impact}")
            with col3:
                confidence = signal['confidence']
                st.write(f"Confidence: {confidence:.0%}")
    
    def _display_market_analysis(self, market_data: Dict):
        """Display market analysis"""
        st.subheader("Market Analysis")
        
        if 'sentiment' in market_data:
            sentiment = market_data['sentiment']
            
            # Sentiment breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Social Media Sentiment**")
                social_sentiment = sentiment.get('social_media_sentiment', 0.5)
                st.progress(social_sentiment)
                st.write(f"Score: {social_sentiment:.2f}")
            
            with col2:
                st.write("**News Sentiment**")
                news_sentiment = sentiment.get('news_sentiment', 0.5)
                st.progress(news_sentiment)
                st.write(f"Score: {news_sentiment:.2f}")
    
    def _display_benchmarking(self):
        """Display benchmarking analysis"""
        st.subheader("Benchmarking")
        
        # Simulated benchmarking data
        benchmark_data = {
            "Metric": ["Carbon Intensity", "Compliance Rate", "Risk Score", "Return on Investment"],
            "Your Portfolio": [0.45, 0.78, 0.32, 0.12],
            "Industry Average": [0.52, 0.65, 0.38, 0.09],
            "Best in Class": [0.28, 0.95, 0.15, 0.18]
        }
        
        df_benchmark = pd.DataFrame(benchmark_data)
        
        # Display comparison table
        st.dataframe(df_benchmark, use_container_width=True)
        
        # Performance vs benchmark
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Your Portfolio', x=df_benchmark['Metric'], y=df_benchmark['Your Portfolio']))
        fig.add_trace(go.Bar(name='Industry Average', x=df_benchmark['Metric'], y=df_benchmark['Industry Average']))
        fig.add_trace(go.Bar(name='Best in Class', x=df_benchmark['Metric'], y=df_benchmark['Best in Class']))
        
        fig.update_layout(title="Portfolio vs Benchmark", barmode='group')
        st.plotly_chart(fig, width='stretch')
    
    def create_ai_agents_panel(self):
        """Create AI agents control panel"""
        with st.expander("🤖 AI Agents & Workflows", expanded=True):
            st.subheader("Autonomous AI Agent System")
            st.markdown("**ACCMN's core AI agents that autonomously manage carbon credit trading and portfolio optimization.**")
            
            # AI Agent Status
            ai_status = get_ai_workflow_status()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("AI Simulation", "🟢 Running" if ai_status.get('simulation_status') == 'running' else "🟡 Stopped")
            
            with col2:
                recent_workflows = ai_status.get('recent_workflows', [])
                total_workflows = sum(wf.get('count', 0) for wf in recent_workflows)
                st.metric("Recent Workflows", f"{total_workflows}")
            
            with col3:
                completed_workflows = sum(wf.get('count', 0) for wf in recent_workflows if 'completed' in wf.get('status', '').lower())
                st.metric("Completed", f"{completed_workflows}")
            
            # Agent Controls
            st.subheader("Agent Controls")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Start AI Simulation", key="start_ai_simulation", type="primary"):
                    with st.spinner("Starting AI simulation..."):
                        start_ai_simulation()
                    st.success("AI simulation started!")
                    st.rerun()
            
            with col2:
                if st.button("⚡ Run AI Workflow", key="run_ai_workflow"):
                    with st.spinner("Running AI workflow..."):
                        results = trigger_ai_workflow(max_companies=5)
                    st.success(f"AI workflow completed for {len(results)} companies!")
                    if results:
                        st.json(results[:2])  # Show first 2 results
            
            with col3:
                if st.button("📊 View AI Status", key="view_ai_status"):
                    st.json(ai_status)
            
            # AI Agent Information
            st.subheader("AI Agent Architecture")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🤖 Buyer Agent**
                - Role: Carbon Credit Buyer
                - Goal: Purchase credits to offset emissions
                - Capability: Risk-aware pricing decisions
                
                **🤖 Seller Agent**
                - Role: Carbon Credit Seller  
                - Goal: Sell credits at optimal prices
                - Capability: Market demand analysis
                """)
            
            with col2:
                st.markdown("""
                **🤖 Coordinator Agent**
                - Role: Market Coordinator
                - Goal: Match buyer/seller offers
                - Capability: Fair pricing & compliance
                
                **🔄 LangGraph Workflow**
                - Data Fetching → Validation → Exposure Calculation → Trading
                - State management across agent interactions
                """)
            
            # Recent AI Activity
            if recent_workflows:
                st.subheader("Recent AI Activity")
                for workflow in recent_workflows:
                    status = workflow.get('status', 'Unknown')
                    count = workflow.get('count', 0)
                    if 'completed' in status.lower():
                        st.success(f"✅ {status}: {count} companies")
                    elif 'failed' in status.lower():
                        st.error(f"❌ {status}: {count} companies")
                    else:
                        st.info(f"🔄 {status}: {count} companies")
            
            # AI Performance Metrics
            st.subheader("AI Performance")
            
            # Simulated metrics - in production, these would come from actual AI performance data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Success Rate", "94.2%", "2.1%")
            
            with col2:
                st.metric("Avg Processing Time", "2.3s", "-0.5s")
            
            with col3:
                st.metric("Trades Executed", "1,247", "156")
            
            with col4:
                st.metric("Risk Reduction", "23.4%", "5.2%")

# Global instance
ui_components = UIComponents()

def create_sidebar_navigation() -> str:
    """Create sidebar navigation"""
    return ui_components.create_sidebar_navigation()

def create_header():
    """Create main header"""
    ui_components.create_header()

def create_market_overview_panel(market_data: Dict):
    """Create market overview panel"""
    ui_components.create_market_overview_panel(market_data)

def create_portfolio_panel(df: pd.DataFrame, trades_df: pd.DataFrame):
    """Create portfolio panel"""
    ui_components.create_portfolio_panel(df, trades_df)

def create_trading_panel(trades_df: pd.DataFrame):
    """Create trading panel"""
    ui_components.create_trading_panel(trades_df)

def create_risk_panel(df: pd.DataFrame, trades_df: pd.DataFrame):
    """Create risk panel"""
    ui_components.create_risk_panel(df, trades_df)

def create_research_panel(market_data: Dict):
    """Create research panel"""
    ui_components.create_research_panel(market_data)

def create_ai_agents_panel():
    """Create AI agents panel"""
    ui_components.create_ai_agents_panel()
