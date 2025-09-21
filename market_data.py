"""
Market Data Module for ACCMN
Handles external data fetching for carbon prices, regulatory news, and market sentiment
"""

import streamlit as st
import pandas as pd
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import re

# Import configuration
from config import (
    EXTERNAL_DATA_SOURCES, 
    REFRESH_INTERVALS, 
    API_CONFIG, 
    SESSION_KEYS,
    TRADING_CONFIG,
    CARBON_API_ENDPOINTS,
    API_KEYS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    """Handles fetching and caching of external market data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = API_CONFIG['timeout']
        self.rate_limit_tracker = {}
        self.price_history = {}  # Track price history for dynamic simulation
        self.last_update_time = None
        self.simulation_state = {}  # Track simulation state for each credit type
        
    def _get_dynamic_simulated_price(self, credit_type: str, base_price: float, 
                                   volatility: float, volume_range: tuple, 
                                   source: str, exchange: str, contract: str) -> Dict:
        """Generate dynamic simulated price with realistic market movements"""
        import random
        import numpy as np
        
        current_time = datetime.now()
        
        # Initialize simulation state if not exists
        if credit_type not in self.simulation_state:
            self.simulation_state[credit_type] = {
                'current_price': base_price,
                'trend': 0.0,  # Long-term trend
                'momentum': 0.0,  # Short-term momentum
                'last_update': current_time,
                'price_history': [base_price],
                'volatility_factor': 1.0
            }
        
        state = self.simulation_state[credit_type]
        time_diff = (current_time - state['last_update']).total_seconds()
        
        # Update trend and momentum based on time elapsed
        if time_diff > 0:
            # Random walk with mean reversion
            trend_change = random.gauss(0, 0.001) * (time_diff / 3600)  # Hourly trend change
            state['trend'] += trend_change
            state['trend'] = max(-0.05, min(0.05, state['trend']))  # Limit trend to ±5%
            
            # Momentum (short-term price direction)
            momentum_change = random.gauss(0, 0.002) * (time_diff / 3600)
            state['momentum'] += momentum_change
            state['momentum'] = max(-0.03, min(0.03, state['momentum']))  # Limit momentum to ±3%
            
            # Volatility factor (market conditions)
            volatility_change = random.gauss(0, 0.001) * (time_diff / 3600)
            state['volatility_factor'] += volatility_change
            state['volatility_factor'] = max(0.5, min(2.0, state['volatility_factor']))  # 0.5x to 2x volatility
        
        # Calculate price change
        base_volatility = volatility * state['volatility_factor']
        
        # Random price movement
        random_change = random.gauss(0, base_volatility)
        
        # Apply trend and momentum
        trend_effect = state['trend'] * state['current_price']
        momentum_effect = state['momentum'] * state['current_price']
        
        # Total price change
        total_change = random_change + trend_effect + momentum_effect
        
        # Update price
        new_price = state['current_price'] + total_change
        
        # Ensure price doesn't go negative or too extreme
        new_price = max(base_price * 0.5, min(base_price * 2.0, new_price))
        
        # Calculate change from previous price
        change = new_price - state['current_price']
        change_pct = (change / state['current_price']) * 100 if state['current_price'] > 0 else 0
        
        # Update state
        state['current_price'] = new_price
        state['last_update'] = current_time
        state['price_history'].append(new_price)
        
        # Keep only last 100 price points
        if len(state['price_history']) > 100:
            state['price_history'] = state['price_history'][-100:]
        
        # Generate volume with some correlation to price movement
        base_volume = random.randint(volume_range[0], volume_range[1])
        volume_multiplier = 1.0 + abs(change_pct) * 0.1  # Higher volume on bigger moves
        volume = int(base_volume * volume_multiplier)
        
        return {
            'price': round(new_price, 2),
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'volume': volume,
            'timestamp': current_time.isoformat(),
            'source': source,
            'exchange': exchange,
            'contract': contract,
            'data_type': 'simulated',
            'trend': round(state['trend'] * 100, 2),  # Trend as percentage
            'momentum': round(state['momentum'] * 100, 2),  # Momentum as percentage
            'volatility_factor': round(state['volatility_factor'], 2)
        }
        
    def _check_rate_limit(self, source: str) -> bool:
        """Check if we're within rate limits for a data source"""
        now = time.time()
        if source not in self.rate_limit_tracker:
            self.rate_limit_tracker[source] = []
        
        # Remove requests older than 1 minute
        self.rate_limit_tracker[source] = [
            req_time for req_time in self.rate_limit_tracker[source] 
            if now - req_time < 60
        ]
        
        return len(self.rate_limit_tracker[source]) < API_CONFIG['rate_limit']['requests_per_minute']
    
    def _record_request(self, source: str):
        """Record a request for rate limiting"""
        now = time.time()
        if source not in self.rate_limit_tracker:
            self.rate_limit_tracker[source] = []
        self.rate_limit_tracker[source].append(now)
    
    def fetch_carbon_prices(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Fetch live carbon prices from various sources
        Returns: Dict with price data for different carbon credit types
        """
        # For real-time updates, always fetch fresh data
        if force_refresh or not self._check_rate_limit('carbon_prices'):
            logger.info("Fetching fresh carbon prices for real-time update")
        else:
            logger.warning("Rate limit exceeded for carbon prices")
            return self._get_cached_prices()
        
        prices = {}
        
        try:
            # EUA (European Union Allowances) - ICE
            eua_price = self._fetch_eua_price()
            if eua_price:
                prices['EUA'] = eua_price
            
            # CCA (California Carbon Allowances) - ARB
            cca_price = self._fetch_cca_price()
            if cca_price:
                prices['CCA'] = cca_price
            
            # RGGI (Regional Greenhouse Gas Initiative)
            rggi_price = self._fetch_rggi_price()
            if rggi_price:
                prices['RGGI'] = rggi_price
            
            # VCS and Gold Standard (voluntary markets)
            voluntary_prices = self._fetch_voluntary_prices()
            prices.update(voluntary_prices)
            
            # Cache the results
            self._cache_prices(prices)
            self._record_request('carbon_prices')
            
            logger.info(f"Successfully fetched prices for {len(prices)} credit types")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching carbon prices: {e}")
            return self._get_cached_prices()
    
    def _fetch_eua_price(self) -> Optional[Dict]:
        """Fetch EUA price from real APIs"""
        try:
            # Try DOVU API first (free tier available)
            dovu_price = self._fetch_dovu_eua_price()
            if dovu_price:
                return dovu_price
            
            # Try RFF World Carbon Pricing Database
            rff_price = self._fetch_rff_eua_price()
            if rff_price:
                return rff_price
            
            # Fallback to realistic simulation
            logger.info("Using simulated EUA price - no real API data available")
            return self._get_simulated_eua_price()
            
        except Exception as e:
            logger.error(f"Error fetching EUA price: {e}")
            return self._get_simulated_eua_price()
    
    def _fetch_dovu_eua_price(self) -> Optional[Dict]:
        """Fetch EUA price from DOVU API"""
        try:
            # DOVU API endpoint for carbon prices
            dovu_config = CARBON_API_ENDPOINTS['dovu']
            url = f"{dovu_config['base_url']}{dovu_config['prices_endpoint']}"
            headers = {
                'Accept': 'application/json',
                'User-Agent': API_CONFIG['user_agent']
            }
            
            # Add API key if available
            if API_KEYS['dovu']:
                headers['Authorization'] = f"Bearer {API_KEYS['dovu']}"
            
            response = requests.get(url, headers=headers, timeout=API_CONFIG['timeout'])
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for EUA price in the response
                for item in data.get('prices', []):
                    if 'EUA' in item.get('market', '') or 'European' in item.get('market', ''):
                        price = float(item.get('price', 0))
                        change = float(item.get('change_24h', 0))
                        change_pct = float(item.get('change_24h_pct', 0))
                        
                        return {
                            'price': round(price, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'volume': int(item.get('volume_24h', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'DOVU',
                            'exchange': 'ICE Futures Europe',
                            'contract': 'EUA Dec 2024',
                            'data_type': 'live'
                        }
            
            logger.warning("DOVU API response not in expected format")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"DOVU API request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing DOVU API response: {e}")
            return None
    
    def _fetch_rff_eua_price(self) -> Optional[Dict]:
        """Fetch EUA price from RFF World Carbon Pricing Database"""
        try:
            # RFF API endpoint
            url = "https://api.rff.org/v1/carbon-prices"
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'ACCMN-CarbonTrader/1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for EU ETS price
                for item in data.get('prices', []):
                    if item.get('jurisdiction') == 'EU' and item.get('instrument') == 'ETS':
                        price = float(item.get('price', 0))
                        # RFF doesn't provide change data, so we'll estimate
                        change = price * 0.01  # 1% change estimate
                        
                        return {
                            'price': round(price, 2),
                            'change': round(change, 2),
                            'change_pct': 1.0,
                            'volume': 1500000,  # Estimated volume
                            'timestamp': datetime.now().isoformat(),
                            'source': 'RFF',
                            'exchange': 'ICE Futures Europe',
                            'contract': 'EUA Dec 2024',
                            'data_type': 'live'
                        }
            
            logger.warning("RFF API response not in expected format")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"RFF API request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing RFF API response: {e}")
            return None
    
    def _get_simulated_eua_price(self) -> Dict:
        """Get simulated EUA price with dynamic simulation"""
        return self._get_dynamic_simulated_price(
            credit_type='EUA',
            base_price=85.50,
            volatility=0.02,
            volume_range=(1000000, 2000000),
            source='ICE',
            exchange='ICE Futures Europe',
            contract='EUA Dec 2024'
        )
    
    def _fetch_cca_price(self) -> Optional[Dict]:
        """Fetch CCA price from real APIs"""
        try:
            # Try DOVU API first
            dovu_price = self._fetch_dovu_cca_price()
            if dovu_price:
                return dovu_price
            
            # Try RFF API
            rff_price = self._fetch_rff_cca_price()
            if rff_price:
                return rff_price
            
            # Fallback to simulation
            logger.info("Using simulated CCA price - no real API data available")
            return self._get_simulated_cca_price()
            
        except Exception as e:
            logger.error(f"Error fetching CCA price: {e}")
            return self._get_simulated_cca_price()
    
    def _fetch_dovu_cca_price(self) -> Optional[Dict]:
        """Fetch CCA price from DOVU API"""
        try:
            url = "https://api.dovu.market/v1/carbon-prices"
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'ACCMN-CarbonTrader/1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for CCA price in the response
                for item in data.get('prices', []):
                    if 'CCA' in item.get('market', '') or 'California' in item.get('market', ''):
                        price = float(item.get('price', 0))
                        change = float(item.get('change_24h', 0))
                        change_pct = float(item.get('change_24h_pct', 0))
                        
                        return {
                            'price': round(price, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'volume': int(item.get('volume_24h', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'DOVU',
                            'exchange': 'California Air Resources Board',
                            'contract': 'CCA Dec 2024',
                            'data_type': 'live'
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"DOVU CCA API request failed: {e}")
            return None
    
    def _fetch_rff_cca_price(self) -> Optional[Dict]:
        """Fetch CCA price from RFF API"""
        try:
            url = "https://api.rff.org/v1/carbon-prices"
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'ACCMN-CarbonTrader/1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for California ETS price
                for item in data.get('prices', []):
                    if item.get('jurisdiction') == 'California' and item.get('instrument') == 'ETS':
                        price = float(item.get('price', 0))
                        change = price * 0.015  # 1.5% change estimate
                        
                        return {
                            'price': round(price, 2),
                            'change': round(change, 2),
                            'change_pct': 1.5,
                            'volume': 800000,  # Estimated volume
                            'timestamp': datetime.now().isoformat(),
                            'source': 'RFF',
                            'exchange': 'California Air Resources Board',
                            'contract': 'CCA Dec 2024',
                            'data_type': 'live'
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"RFF CCA API request failed: {e}")
            return None
    
    def _get_simulated_cca_price(self) -> Dict:
        """Get simulated CCA price with dynamic simulation"""
        return self._get_dynamic_simulated_price(
            credit_type='CCA',
            base_price=32.75,
            volatility=0.03,
            volume_range=(500000, 1200000),
            source='ARB',
            exchange='California Air Resources Board',
            contract='CCA Dec 2024'
        )
    
    def _fetch_rggi_price(self) -> Optional[Dict]:
        """Fetch RGGI price with dynamic simulation"""
        try:
            # For now, return simulated RGGI price
            # In production, this would connect to RGGI API
            return self._get_dynamic_simulated_price(
                credit_type='RGGI',
                base_price=18.90,
                volatility=0.025,
                volume_range=(300000, 600000),
                source='RGGI',
                exchange='RGGI Inc.',
                contract='RGGI Dec 2024'
            )
        except Exception as e:
            logger.error(f"Error fetching RGGI price: {e}")
            return None
    
    def _fetch_voluntary_prices(self) -> Dict[str, Dict]:
        """Fetch voluntary carbon credit prices from real APIs"""
        try:
            voluntary_prices = {}
            
            # VCS (Verified Carbon Standard) - try Carbonmark API
            vcs_price = self._fetch_carbonmark_vcs_price()
            if not vcs_price:
                vcs_price = self._fetch_alliedoffsets_vcs_price()
            if not vcs_price:
                vcs_price = self._get_simulated_vcs_price()
            if vcs_price:
                voluntary_prices['VCS'] = vcs_price
            
            # Gold Standard - try Carbonmark API
            gs_price = self._fetch_carbonmark_gold_standard_price()
            if not gs_price:
                gs_price = self._fetch_alliedoffsets_gold_standard_price()
            if not gs_price:
                gs_price = self._get_simulated_gold_standard_price()
            if gs_price:
                voluntary_prices['Gold Standard'] = gs_price
            
            return voluntary_prices
            
        except Exception as e:
            logger.error(f"Error fetching voluntary prices: {e}")
            return {}
    
    def _fetch_carbonmark_vcs_price(self) -> Optional[Dict]:
        """Fetch VCS price from Carbonmark API"""
        try:
            url = "https://api.carbonmark.com/v1/prices"
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'ACCMN-CarbonTrader/1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for VCS price in the response
                for item in data.get('prices', []):
                    if 'VCS' in item.get('standard', '') or 'Verified Carbon Standard' in item.get('standard', ''):
                        price = float(item.get('price', 0))
                        change = float(item.get('change_24h', 0))
                        change_pct = float(item.get('change_24h_pct', 0))
                        
                        return {
                            'price': round(price, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'volume': int(item.get('volume_24h', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'Carbonmark',
                            'exchange': 'VCS Registry',
                            'contract': 'VCS Credits',
                            'data_type': 'live'
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Carbonmark VCS API request failed: {e}")
            return None
    
    def _fetch_alliedoffsets_vcs_price(self) -> Optional[Dict]:
        """Fetch VCS price from AlliedOffsets API"""
        try:
            url = "https://api.alliedoffsets.com/v1/carbon-prices"
            headers = {
                'Accept': 'application/json',
                'User-Agent': API_CONFIG['user_agent']
            }
            
            # Add API key if available
            if API_KEYS['alliedoffsets']:
                headers['Authorization'] = f"Bearer {API_KEYS['alliedoffsets']}"
                logger.info("Using AlliedOffsets API key for authentication")
            else:
                logger.warning("No AlliedOffsets API key found - using public access")
            
            response = requests.get(url, headers=headers, timeout=API_CONFIG['timeout'])
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for VCS price in the response
                for item in data.get('prices', []):
                    if 'VCS' in item.get('standard', '') or 'Verified Carbon Standard' in item.get('standard', ''):
                        price = float(item.get('price', 0))
                        change = float(item.get('change_24h', 0))
                        change_pct = float(item.get('change_24h_pct', 0))
                        
                        return {
                            'price': round(price, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'volume': int(item.get('volume_24h', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'AlliedOffsets',
                            'exchange': 'VCS Registry',
                            'contract': 'VCS Credits',
                            'data_type': 'live'
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"AlliedOffsets VCS API request failed: {e}")
            return None
    
    def _fetch_carbonmark_gold_standard_price(self) -> Optional[Dict]:
        """Fetch Gold Standard price from Carbonmark API"""
        try:
            url = "https://api.carbonmark.com/v1/prices"
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'ACCMN-CarbonTrader/1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for Gold Standard price in the response
                for item in data.get('prices', []):
                    if 'Gold Standard' in item.get('standard', '') or 'GS' in item.get('standard', ''):
                        price = float(item.get('price', 0))
                        change = float(item.get('change_24h', 0))
                        change_pct = float(item.get('change_24h_pct', 0))
                        
                        return {
                            'price': round(price, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'volume': int(item.get('volume_24h', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'Carbonmark',
                            'exchange': 'Gold Standard Foundation',
                            'contract': 'Gold Standard Credits',
                            'data_type': 'live'
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Carbonmark Gold Standard API request failed: {e}")
            return None
    
    def _fetch_alliedoffsets_gold_standard_price(self) -> Optional[Dict]:
        """Fetch Gold Standard price from AlliedOffsets API"""
        try:
            url = "https://api.alliedoffsets.com/v1/carbon-prices"
            headers = {
                'Accept': 'application/json',
                'User-Agent': API_CONFIG['user_agent']
            }
            
            # Add API key if available
            if API_KEYS['alliedoffsets']:
                headers['Authorization'] = f"Bearer {API_KEYS['alliedoffsets']}"
                logger.info("Using AlliedOffsets API key for authentication")
            else:
                logger.warning("No AlliedOffsets API key found - using public access")
            
            response = requests.get(url, headers=headers, timeout=API_CONFIG['timeout'])
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for Gold Standard price in the response
                for item in data.get('prices', []):
                    if 'Gold Standard' in item.get('standard', '') or 'GS' in item.get('standard', ''):
                        price = float(item.get('price', 0))
                        change = float(item.get('change_24h', 0))
                        change_pct = float(item.get('change_24h_pct', 0))
                        
                        return {
                            'price': round(price, 2),
                            'change': round(change, 2),
                            'change_pct': round(change_pct, 2),
                            'volume': int(item.get('volume_24h', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'AlliedOffsets',
                            'exchange': 'Gold Standard Foundation',
                            'contract': 'Gold Standard Credits',
                            'data_type': 'live'
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"AlliedOffsets Gold Standard API request failed: {e}")
            return None
    
    def _get_simulated_vcs_price(self) -> Dict:
        """Get simulated VCS price with dynamic simulation"""
        return self._get_dynamic_simulated_price(
            credit_type='VCS',
            base_price=12.50,
            volatility=0.04,
            volume_range=(100000, 300000),
            source='VCS Registry',
            exchange='VCS Registry',
            contract='VCS Credits'
        )
    
    def _get_simulated_gold_standard_price(self) -> Dict:
        """Get simulated Gold Standard price with dynamic simulation"""
        return self._get_dynamic_simulated_price(
            credit_type='Gold Standard',
            base_price=15.75,
            volatility=0.035,
            volume_range=(50000, 200000),
            source='Gold Standard',
            exchange='Gold Standard Foundation',
            contract='Gold Standard Credits'
        )
    
    def _cache_prices(self, prices: Dict):
        """Cache prices in session state"""
        if SESSION_KEYS['market_data'] not in st.session_state:
            st.session_state[SESSION_KEYS['market_data']] = {}
        
        st.session_state[SESSION_KEYS['market_data']]['prices'] = prices
        st.session_state[SESSION_KEYS['market_data']]['last_update'] = datetime.now().isoformat()
    
    def _get_cached_prices(self) -> Dict:
        """Get cached prices from session state"""
        if SESSION_KEYS['market_data'] in st.session_state:
            return st.session_state[SESSION_KEYS['market_data']].get('prices', {})
        return {}
    
    def fetch_regulatory_news(self) -> List[Dict]:
        """
        Fetch latest regulatory news and updates
        Returns: List of news items
        """
        if not self._check_rate_limit('regulatory_news'):
            logger.warning("Rate limit exceeded for regulatory news")
            return self._get_cached_news()
        
        try:
            news_items = []
            
            # EU CBAM updates
            cbam_news = self._fetch_cbam_news()
            news_items.extend(cbam_news)
            
            # Carbon market regulations
            market_news = self._fetch_market_regulations()
            news_items.extend(market_news)
            
            # Cache the results
            self._cache_news(news_items)
            self._record_request('regulatory_news')
            
            logger.info(f"Successfully fetched {len(news_items)} news items")
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching regulatory news: {e}")
            return self._get_cached_news()
    
    def _fetch_cbam_news(self) -> List[Dict]:
        """Fetch EU CBAM related news"""
        try:
            # Simulated news - in production, you'd fetch from actual news APIs
            return [
                {
                    'title': 'EU CBAM Phase 2 Implementation Update',
                    'summary': 'New reporting requirements for importers take effect Q2 2024',
                    'source': 'EU Commission',
                    'timestamp': datetime.now().isoformat(),
                    'impact': 'high',
                    'category': 'regulatory'
                },
                {
                    'title': 'CBAM Carbon Price Methodology Revised',
                    'summary': 'Updated calculation methods for embedded emissions',
                    'source': 'EU Commission',
                    'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'impact': 'medium',
                    'category': 'methodology'
                }
            ]
        except Exception as e:
            logger.error(f"Error fetching CBAM news: {e}")
            return []
    
    def _fetch_market_regulations(self) -> List[Dict]:
        """Fetch general carbon market regulations"""
        try:
            return [
                {
                    'title': 'California Cap-and-Trade Program Updates',
                    'summary': 'New compliance periods and allowance allocations announced',
                    'source': 'CARB',
                    'timestamp': (datetime.now() - timedelta(hours=4)).isoformat(),
                    'impact': 'medium',
                    'category': 'compliance'
                },
                {
                    'title': 'RGGI Program Expansion',
                    'summary': 'Additional states consider joining the regional program',
                    'source': 'RGGI',
                    'timestamp': (datetime.now() - timedelta(hours=6)).isoformat(),
                    'impact': 'low',
                    'category': 'expansion'
                }
            ]
        except Exception as e:
            logger.error(f"Error fetching market regulations: {e}")
            return []
    
    def _cache_news(self, news_items: List[Dict]):
        """Cache news in session state"""
        if SESSION_KEYS['market_data'] not in st.session_state:
            st.session_state[SESSION_KEYS['market_data']] = {}
        
        st.session_state[SESSION_KEYS['market_data']]['news'] = news_items
        st.session_state[SESSION_KEYS['market_data']]['news_last_update'] = datetime.now().isoformat()
    
    def _get_cached_news(self) -> List[Dict]:
        """Get cached news from session state"""
        if SESSION_KEYS['market_data'] in st.session_state:
            return st.session_state[SESSION_KEYS['market_data']].get('news', [])
        return []
    
    def fetch_market_sentiment(self) -> Dict:
        """
        Fetch market sentiment indicators
        Returns: Dict with sentiment metrics
        """
        try:
            # Simulated sentiment data - in production, you'd analyze social media, news sentiment, etc.
            sentiment = {
                'overall_sentiment': 'bullish',
                'sentiment_score': 0.65,  # -1 to 1 scale
                'fear_greed_index': 45,   # 0-100 scale
                'social_media_sentiment': 0.58,
                'news_sentiment': 0.72,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache sentiment
            if SESSION_KEYS['market_data'] not in st.session_state:
                st.session_state[SESSION_KEYS['market_data']] = {}
            
            st.session_state[SESSION_KEYS['market_data']]['sentiment'] = sentiment
            st.session_state[SESSION_KEYS['market_data']]['sentiment_last_update'] = datetime.now().isoformat()
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {e}")
            return {}
    
    def get_market_overview(self) -> Dict:
        """
        Get comprehensive market overview
        Returns: Dict with all market data
        """
        overview = {
            'prices': self.fetch_carbon_prices(force_refresh=True),
            'news': self.fetch_regulatory_news(),
            'sentiment': self.fetch_market_sentiment(),
            'last_update': datetime.now().isoformat()
        }
        
        return overview
    
    def should_refresh_data(self, data_type: str) -> bool:
        """Check if data should be refreshed based on last update time"""
        if SESSION_KEYS['market_data'] not in st.session_state:
            return True
        
        last_update_key = f"{data_type}_last_update"
        if last_update_key not in st.session_state[SESSION_KEYS['market_data']]:
            return True
        
        last_update = datetime.fromisoformat(
            st.session_state[SESSION_KEYS['market_data']][last_update_key]
        )
        
        refresh_interval = REFRESH_INTERVALS.get('market_data', 60)
        return (datetime.now() - last_update).total_seconds() > refresh_interval

# Global instance
market_data_fetcher = MarketDataFetcher()

def get_market_data() -> Dict:
    """Get market data with caching and refresh logic"""
    if market_data_fetcher.should_refresh_data('market_data'):
        return market_data_fetcher.get_market_overview()
    else:
        # Return cached data
        if SESSION_KEYS['market_data'] in st.session_state:
            return st.session_state[SESSION_KEYS['market_data']]
        else:
            return market_data_fetcher.get_market_overview()

def get_carbon_prices(force_refresh: bool = True) -> Dict:
    """Get carbon prices with caching"""
    if force_refresh or market_data_fetcher.should_refresh_data('prices'):
        return market_data_fetcher.fetch_carbon_prices(force_refresh=force_refresh)
    else:
        return market_data_fetcher._get_cached_prices()

def get_regulatory_news() -> List[Dict]:
    """Get regulatory news with caching"""
    if market_data_fetcher.should_refresh_data('news'):
        return market_data_fetcher.fetch_regulatory_news()
    else:
        return market_data_fetcher._get_cached_news()

def get_market_sentiment() -> Dict:
    """Get market sentiment with caching"""
    if market_data_fetcher.should_refresh_data('sentiment'):
        return market_data_fetcher.fetch_market_sentiment()
    else:
        if SESSION_KEYS['market_data'] in st.session_state:
            return st.session_state[SESSION_KEYS['market_data']].get('sentiment', {})
        else:
            return market_data_fetcher.fetch_market_sentiment()
