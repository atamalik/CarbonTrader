"""
Utility functions for ACCMN
Helper functions for data processing, formatting, and common operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging

# Setup logging
logger = logging.getLogger(__name__)

def format_currency(value: Union[float, int], currency: str = "USD") -> str:
    """Format currency values with proper formatting"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    if currency == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

def format_percentage(value: Union[float, int], decimals: int = 2) -> str:
    """Format percentage values"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    return f"{value:.{decimals}f}%"

def format_number(value: Union[float, int], decimals: int = 0) -> str:
    """Format large numbers with commas"""
    if pd.isna(value) or value is None:
        return "N/A"
    
    return f"{value:,.{decimals}f}"

def format_timestamp(timestamp: Union[str, datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp for display"""
    if pd.isna(timestamp) or timestamp is None:
        return "N/A"
    
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except:
            return str(timestamp)
    
    return timestamp.strftime(format_str)

def calculate_change_percentage(current: float, previous: float) -> float:
    """Calculate percentage change between two values"""
    if previous == 0:
        return 0.0 if current == 0 else float('inf')
    
    return ((current - previous) / previous) * 100

def get_change_indicator(change: float, threshold: float = 0.1) -> str:
    """Get change indicator based on percentage change"""
    if abs(change) < threshold:
        return "→"
    elif change > 0:
        return "↗"
    else:
        return "↘"

def get_change_color(change: float) -> str:
    """Get color for change indicator"""
    if change > 0:
        return "green"
    elif change < 0:
        return "red"
    else:
        return "gray"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator

def calculate_risk_score(default_prob: float, npl_ratio: float, carbon_intensity: float, 
                        weights: Dict[str, float] = None) -> float:
    """Calculate composite risk score"""
    if weights is None:
        weights = {'default': 0.4, 'npl': 0.3, 'carbon': 0.3}
    
    # Normalize inputs (assuming they're in percentage form)
    default_risk = min(default_prob / 100, 1.0) if not pd.isna(default_prob) else 0
    npl_risk = min(npl_ratio / 100, 1.0) if not pd.isna(npl_ratio) else 0
    carbon_risk = min(carbon_intensity, 1.0) if not pd.isna(carbon_intensity) else 0
    
    risk_score = (weights['default'] * default_risk + 
                  weights['npl'] * npl_risk + 
                  weights['carbon'] * carbon_risk)
    
    return min(risk_score, 1.0)

def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Value at Risk"""
    if len(returns) == 0:
        return 0.0
    
    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculate Expected Shortfall (Conditional VaR)"""
    if len(returns) == 0:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns / returns.std()

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_volatility(returns: pd.Series, annualized: bool = True) -> float:
    """Calculate volatility"""
    if len(returns) == 0:
        return 0.0
    
    vol = returns.std()
    if annualized:
        vol *= np.sqrt(252)  # Annualize daily volatility
    
    return vol

def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for returns"""
    return returns_df.corr()

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta coefficient"""
    if len(asset_returns) == 0 or len(market_returns) == 0:
        return 0.0
    
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return 0.0
    
    return covariance / market_variance

def calculate_tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate tracking error"""
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    excess_returns = portfolio_returns - benchmark_returns
    return excess_returns.std()

def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate information ratio"""
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = calculate_tracking_error(portfolio_returns, benchmark_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return excess_returns.mean() / tracking_error

def calculate_herfindahl_index(weights: pd.Series) -> float:
    """Calculate Herfindahl-Hirschman Index for concentration"""
    if len(weights) == 0:
        return 0.0
    
    # Normalize weights to sum to 1
    normalized_weights = weights / weights.sum()
    return (normalized_weights ** 2).sum()

def calculate_effective_number_of_stocks(weights: pd.Series) -> float:
    """Calculate effective number of stocks"""
    hhi = calculate_herfindahl_index(weights)
    if hhi == 0:
        return 0.0
    return 1 / hhi

def calculate_turnover_rate(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """Calculate portfolio turnover rate"""
    if len(old_weights) == 0 or len(new_weights) == 0:
        return 0.0
    
    # Align indices
    common_indices = old_weights.index.intersection(new_weights.index)
    if len(common_indices) == 0:
        return 0.0
    
    old_aligned = old_weights.reindex(common_indices, fill_value=0)
    new_aligned = new_weights.reindex(common_indices, fill_value=0)
    
    return abs(new_aligned - old_aligned).sum() / 2

def calculate_esg_score(environmental: float, social: float, governance: float, 
                       weights: Dict[str, float] = None) -> float:
    """Calculate composite ESG score"""
    if weights is None:
        weights = {'environmental': 0.4, 'social': 0.3, 'governance': 0.3}
    
    # Normalize scores to 0-1 range
    e_score = min(max(environmental, 0), 100) / 100 if not pd.isna(environmental) else 0
    s_score = min(max(social, 0), 100) / 100 if not pd.isna(social) else 0
    g_score = min(max(governance, 0), 100) / 100 if not pd.isna(governance) else 0
    
    esg_score = (weights['environmental'] * e_score + 
                 weights['social'] * s_score + 
                 weights['governance'] * g_score)
    
    return min(esg_score, 1.0)

def calculate_carbon_intensity(emissions: float, revenue: float) -> float:
    """Calculate carbon intensity (emissions per unit revenue)"""
    if revenue == 0 or pd.isna(revenue) or pd.isna(emissions):
        return 0.0
    
    return emissions / revenue

def calculate_carbon_footprint(emissions: float, credits_purchased: float) -> float:
    """Calculate net carbon footprint"""
    if pd.isna(emissions) or pd.isna(credits_purchased):
        return 0.0
    
    return max(emissions - credits_purchased, 0)

def calculate_offset_ratio(credits_purchased: float, total_emissions: float) -> float:
    """Calculate offset ratio"""
    if total_emissions == 0 or pd.isna(total_emissions) or pd.isna(credits_purchased):
        return 0.0
    
    return min(credits_purchased / total_emissions, 1.0)

def get_compliance_status(offset_ratio: float, thresholds: Dict[str, float] = None) -> str:
    """Get compliance status based on offset ratio"""
    if thresholds is None:
        thresholds = {
            'target': 1.0,
            'minimum': 0.8,
            'warning': 0.7
        }
    
    if pd.isna(offset_ratio):
        return 'Unknown'
    elif offset_ratio >= thresholds['target']:
        return 'Compliant'
    elif offset_ratio >= thresholds['minimum']:
        return 'At Risk'
    elif offset_ratio >= thresholds['warning']:
        return 'Warning'
    else:
        return 'Non-Compliant'

def calculate_portfolio_metrics(weights: pd.Series, returns: pd.Series, 
                               risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Calculate comprehensive portfolio metrics"""
    if len(weights) == 0 or len(returns) == 0:
        return {}
    
    metrics = {
        'total_return': returns.sum(),
        'annualized_return': returns.mean() * 252,
        'volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(returns),
        'var_95': calculate_var(returns, 0.95),
        'var_99': calculate_var(returns, 0.99),
        'expected_shortfall': calculate_expected_shortfall(returns, 0.95),
        'herfindahl_index': calculate_herfindahl_index(weights),
        'effective_number_of_stocks': calculate_effective_number_of_stocks(weights)
    }
    
    return metrics

def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Validate data quality and return quality metrics"""
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_columns': [],
        'missing_data': {},
        'duplicate_rows': 0,
        'data_types': {},
        'quality_score': 0.0
    }
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    quality_report['missing_columns'] = missing_columns
    
    # Check for missing data
    for col in df.columns:
        missing_count = df[col].isna().sum()
        quality_report['missing_data'][col] = {
            'count': missing_count,
            'percentage': (missing_count / len(df)) * 100
        }
    
    # Check for duplicates
    quality_report['duplicate_rows'] = df.duplicated().sum()
    
    # Data types
    quality_report['data_types'] = df.dtypes.to_dict()
    
    # Calculate quality score
    total_checks = len(required_columns) + len(df.columns)
    passed_checks = len(required_columns) - len(missing_columns)
    
    for col in df.columns:
        if quality_report['missing_data'][col]['percentage'] < 10:  # Less than 10% missing
            passed_checks += 1
    
    quality_report['quality_score'] = passed_checks / total_checks
    
    return quality_report

def clean_data(df: pd.DataFrame, 
               numeric_columns: List[str] = None,
               categorical_columns: List[str] = None,
               date_columns: List[str] = None) -> pd.DataFrame:
    """Clean and preprocess data"""
    df_clean = df.copy()
    
    # Handle numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
    
    # Handle categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
                df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Handle date columns
    if date_columns:
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    return df_clean

def create_summary_statistics(df: pd.DataFrame, numeric_columns: List[str] = None) -> pd.DataFrame:
    """Create summary statistics for numeric columns"""
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        return pd.DataFrame()
    
    summary_stats = df[numeric_columns].describe()
    
    # Add additional statistics
    additional_stats = pd.DataFrame({
        'skewness': df[numeric_columns].skew(),
        'kurtosis': df[numeric_columns].kurtosis(),
        'missing_count': df[numeric_columns].isna().sum(),
        'missing_percentage': (df[numeric_columns].isna().sum() / len(df)) * 100
    })
    
    return pd.concat([summary_stats, additional_stats])

def export_to_excel(data_dict: Dict[str, pd.DataFrame], filename: str) -> None:
    """Export multiple DataFrames to Excel file"""
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"Successfully exported data to {filename}")
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise

def export_to_csv(df: pd.DataFrame, filename: str, index: bool = False) -> None:
    """Export DataFrame to CSV file"""
    try:
        df.to_csv(filename, index=index)
        logger.info(f"Successfully exported data to {filename}")
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise
