"""
Configuration file for ACCMN (Autonomous Carbon Credit Management Network)
Contains constants, settings, and configuration parameters
"""

import os
from datetime import timedelta

# Database Configuration
DATABASE_PATH = 'sec_financials.db'

# External Data Sources
EXTERNAL_DATA_SOURCES = {
    'carbon_prices': {
        'eua': 'https://www.ice.com/products/27996666/Dutch-TTF-Gas-Futures',
        'cca': 'https://www.arb.ca.gov/cc/capandtrade/auction/auction.htm',
        'rggi': 'https://www.rggi.org/market/co2-auctions'
    },
    'regulatory_news': {
        'eu_cbam': 'https://ec.europa.eu/taxation_customs/green-taxation-0/carbon-border-adjustment-mechanism_en',
        'carbon_markets': 'https://carbonpricingdashboard.worldbank.org/'
    }
}

# Data Refresh Intervals (in seconds)
REFRESH_INTERVALS = {
    'market_data': 60,      # 1 minute
    'portfolio_data': 30,   # 30 seconds
    'risk_metrics': 120,    # 2 minutes
    'compliance_data': 300  # 5 minutes
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'ACCMN: Sustainable Portfolio Management',
    'layout': 'wide',
    'theme': {
        'primary_color': '#1f77b4',
        'secondary_color': '#ff7f0e',
        'success_color': '#2ca02c',
        'warning_color': '#d62728',
        'info_color': '#9467bd'
    },
    'charts': {
        'color_sequence': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    }
}

# Trading Configuration
TRADING_CONFIG = {
    'default_credit_types': ['EUA', 'CCA', 'RGGI', 'VCS', 'Gold Standard'],
    'price_ranges': {
        'EUA': {'min': 20, 'max': 100},
        'CCA': {'min': 15, 'max': 80},
        'RGGI': {'min': 10, 'max': 60},
        'VCS': {'min': 5, 'max': 50},
        'Gold Standard': {'min': 8, 'max': 45}
    },
    'trading_limits': {
        'max_trade_size': 1000000,  # $1M
        'min_trade_size': 1000,     # $1K
        'max_daily_trades': 100
    }
}

# Risk Management
RISK_CONFIG = {
    'var_confidence_levels': [95, 99],
    'stress_test_scenarios': {
        'mild': 0.1,    # 10% price change
        'moderate': 0.25,  # 25% price change
        'severe': 0.5   # 50% price change
    },
    'exposure_limits': {
        'max_sector_exposure': 0.3,  # 30% per sector
        'max_counterparty_exposure': 0.1,  # 10% per counterparty
        'max_credit_type_exposure': 0.4  # 40% per credit type
    }
}

# Compliance Configuration
COMPLIANCE_CONFIG = {
    'offset_thresholds': {
        'minimum_offset_ratio': 0.8,  # 80% of emissions must be offset
        'target_offset_ratio': 1.0,   # 100% target
        'warning_threshold': 0.7      # 70% warning threshold
    },
    'reporting_periods': {
        'daily': 1,
        'weekly': 7,
        'monthly': 30,
        'quarterly': 90,
        'annually': 365
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'file': 'frontend.log',
        'console': True
    }
}

# Session State Keys
SESSION_KEYS = {
    'user_preferences': 'user_preferences',
    'market_data': 'market_data',
    'portfolio_data': 'portfolio_data',
    'risk_metrics': 'risk_metrics',
    'compliance_data': 'compliance_data',
    'last_refresh': 'last_refresh',
    'selected_company': 'selected_company',
    'active_tab': 'active_tab'
}

# API Configuration
API_CONFIG = {
    'timeout': 30,
    'retry_attempts': 3,
    'retry_delay': 1,
    'rate_limit': {
        'requests_per_minute': 60,
        'requests_per_hour': 1000
    },
    'user_agent': 'ACCMN-CarbonTrader/1.0'
}

# Carbon Price API Endpoints
CARBON_API_ENDPOINTS = {
    'dovu': {
        'base_url': 'https://api.dovu.market/v1',
        'prices_endpoint': '/carbon-prices',
        'free_tier': True,
        'rate_limit': 100,  # requests per hour
        'description': 'Global carbon credit pricing from voluntary and compliance markets'
    },
    'carbonmark': {
        'base_url': 'https://api.carbonmark.com/v1',
        'prices_endpoint': '/prices',
        'free_tier': True,
        'rate_limit': 1000,  # requests per month
        'description': 'Voluntary carbon market credits with real-time pricing'
    },
    'alliedoffsets': {
        'base_url': 'https://api.alliedoffsets.com/v1',
        'prices_endpoint': '/carbon-prices',
        'free_tier': True,
        'rate_limit': 500,  # requests per day
        'description': 'Voluntary market data with pricing and project information'
    },
    'rff': {
        'base_url': 'https://api.rff.org/v1',
        'prices_endpoint': '/carbon-prices',
        'free_tier': True,
        'rate_limit': 200,  # requests per hour
        'description': 'World Carbon Pricing Database with historical and current data'
    }
}

# API Keys (set as environment variables in production)
import os
API_KEYS = {
    'dovu': os.getenv('DOVU_API_KEY', ''),
    'carbonmark': os.getenv('CARBONMARK_API_KEY', ''),
    'alliedoffsets': os.getenv('ALLIEDOFFSETS_API_KEY', ''),
    'rff': os.getenv('RFF_API_KEY', '')
}

# File Paths
FILE_PATHS = {
    'logs': 'Logs/',
    'data': 'CSV Files/',
    'database': 'sec_financials.db',
    'backup': 'backup/'
}

# Sector Mapping
SECTOR_MAPPING = {
    1311: 'Energy',
    1623: 'Construction', 
    2000: 'Food',
    2080: 'Beverages',
    3531: 'Machinery',
    3826: 'Tech',
    7370: 'Tech',
    4011: 'Transportation'
}

# Default Values
DEFAULT_VALUES = {
    'emissions': {
        'scope1_co2e': 0,
        'scope2_co2e': 0,
        'scope3_co2e': 0,
        'credits_purchased_tonnes': 0,
        'uncertainty': 0
    },
    'lending': {
        'total_loans_outstanding_usd': 0,
        'total_exposure_usd': 0,
        'non_performing_loans_pct': 0,
        'default_probability_pct': 0
    },
    'trading': {
        'credits_traded_tonnes': 0,
        'trade_price_usd': 0
    }
}
