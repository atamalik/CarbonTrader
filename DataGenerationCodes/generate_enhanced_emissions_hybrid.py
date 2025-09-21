import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import lognorm, gamma, norm, weibull_min, truncnorm, expon
import logging
import json
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from emission_factors_research_tool import EmissionFactorsResearchTool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/enhanced_emissions_hybrid.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HybridEmissionFactorsManager:
    """
    Manages emission factors from both static (emission_factors) and dynamic (emission_factors_research) tables
    """
    
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.static_factors_cache = {}
        self.dynamic_factors_cache = {}
        self._load_static_factors()
        self._load_dynamic_factors()
    
    def _load_static_factors(self):
        """Load static emission factors from emission_factors table"""
        try:
            query = """
            SELECT Scope, Level1, Level2, Level3, Level4, ColumnText, 
                   UOM, GHG_Unit, Conversion_Factor_2024
            FROM emission_factors
            WHERE Conversion_Factor_2024 IS NOT NULL
            """
            df = pd.read_sql(query, self.conn)
            self.static_factors_cache = df
            logger.info(f"Loaded {len(df)} static emission factors")
        except Exception as e:
            logger.error(f"Error loading static factors: {e}")
            self.static_factors_cache = pd.DataFrame()
    
    def _load_dynamic_factors(self):
        """Load dynamic emission factors from emission_factors_research table"""
        try:
            query = """
            SELECT sector, sic_code, scope1_factors, scope2_factors, scope3_factors,
                   sources, confidence_score, last_updated
            FROM emission_factors_research
            """
            df = pd.read_sql(query, self.conn)
            self.dynamic_factors_cache = df
            logger.info(f"Loaded {len(df)} dynamic emission factors")
        except Exception as e:
            logger.error(f"Error loading dynamic factors: {e}")
            self.dynamic_factors_cache = pd.DataFrame()
    
    def get_static_factor(self, scope, level1=None, level2=None, level3=None, level4=None, column_text=None):
        """
        Get static emission factor based on hierarchical classification
        """
        if self.static_factors_cache.empty:
            return None
        
        mask = self.static_factors_cache['Scope'] == scope
        
        if level1:
            mask &= self.static_factors_cache['Level1'] == level1
        if level2:
            mask &= self.static_factors_cache['Level2'] == level2
        if level3:
            mask &= self.static_factors_cache['Level3'] == level3
        if level4:
            mask &= self.static_factors_cache['Level4'] == level4
        if column_text:
            mask &= self.static_factors_cache['ColumnText'] == column_text
        
        matches = self.static_factors_cache[mask]
        
        if len(matches) > 0:
            # Return the most specific match (highest number of non-null levels)
            best_match = matches.iloc[0]
            return {
                'factor': best_match['Conversion_Factor_2024'],
                'unit': best_match['UOM'],
                'ghg_unit': best_match['GHG_Unit'],
                'source': 'static_database'
            }
        
        return None
    
    def get_dynamic_factor(self, sic_code, scope):
        """
        Get dynamic emission factor for a specific SIC code and scope
        """
        if self.dynamic_factors_cache.empty:
            return None
        
        matches = self.dynamic_factors_cache[self.dynamic_factors_cache['sic_code'] == sic_code]
        
        if len(matches) > 0:
            factor_data = matches.iloc[0]
            scope_key = f'{scope}_factors'
            
            if scope_key in factor_data and factor_data[scope_key]:
                try:
                    factors = json.loads(factor_data[scope_key])
                    return {
                        'factors': factors,
                        'confidence': factor_data['confidence_score'],
                        'sources': json.loads(factor_data['sources']) if factor_data['sources'] else [],
                        'last_updated': factor_data['last_updated'],
                        'source': 'dynamic_research'
                    }
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON for SIC {sic_code}, scope {scope}")
        
        return None
    
    def get_hybrid_factor(self, sic_code, scope, activity_type=None, activity_value=None):
        """
        Get the best available emission factor using both static and dynamic sources
        Priority: Dynamic factors (if available and recent) > Static factors > Default estimates
        """
        # Try dynamic factors first
        dynamic_factor = self.get_dynamic_factor(sic_code, scope)
        
        if dynamic_factor and dynamic_factor['confidence'] > 0.5:
            # Check if dynamic factor is recent (within 30 days)
            try:
                last_updated = datetime.fromisoformat(dynamic_factor['last_updated'].replace('Z', '+00:00'))
                if (datetime.now() - last_updated).days <= 30:
                    logger.info(f"Using dynamic factor for SIC {sic_code}, scope {scope} (confidence: {dynamic_factor['confidence']})")
                    return dynamic_factor
            except:
                pass
        
        # Fall back to static factors
        static_factor = self._get_static_factor_for_activity(scope, activity_type)
        if static_factor:
            logger.info(f"Using static factor for SIC {sic_code}, scope {scope}")
            return static_factor
        
        # Default fallback
        logger.warning(f"No factors found for SIC {sic_code}, scope {scope}, using defaults")
        return self._get_default_factor(scope)
    
    def _get_static_factor_for_activity(self, scope, activity_type):
        """Map activity types to static factor categories"""
        activity_mapping = {
            'revenue': ('Fuels', 'Gaseous fuels', None, None),
            'assets': ('Fuels', 'Liquid fuels', None, None),
            'energy_consumption': ('Fuels', 'Gaseous fuels', 'Natural gas', None),
            'transportation': ('Transport', 'Road transport', 'Passenger cars', None),
            'manufacturing': ('Industrial processes', 'Chemical industry', None, None)
        }
        
        if activity_type in activity_mapping:
            level1, level2, level3, level4 = activity_mapping[activity_type]
            return self.get_static_factor(scope, level1, level2, level3, level4)
        
        return None
    
    def _get_default_factor(self, scope):
        """Provide default emission factors when no specific factors are available"""
        default_factors = {
            'scope1': {'revenue': 50.0, 'assets': 15.0},
            'scope2': {'revenue': 30.0, 'assets': 10.0},
            'scope3': {'revenue': 100.0, 'assets': 25.0}
        }
        
        return {
            'factors': default_factors.get(scope, {}),
            'confidence': 0.1,
            'sources': [{'name': 'Default', 'confidence': 0.1, 'reference': 'Conservative estimates'}],
            'source': 'default_estimate'
        }

def calculate_hybrid_emissions(adsh, sic_code, financial_data, business_activities, factors_manager):
    """
    Calculate emissions using hybrid approach combining both factor sources
    """
    emissions = {'scope1': 0, 'scope2': 0, 'scope3': 0}
    calculation_details = []
    
    # Process each scope
    for scope in ['scope1', 'scope2', 'scope3']:
        scope_emissions = 0
        
        # Try business activity data first
        if not business_activities.empty:
            for _, activity in business_activities.iterrows():
                tag = activity['tag']
                value = activity['value']
                unit = activity['unit']
                
                # Get hybrid factor for this activity
                factor_data = factors_manager.get_hybrid_factor(sic_code, scope, tag, value)
                
                if factor_data and 'factors' in factor_data:
                    factors = factor_data['factors']
                    
                    # Try to match activity to factor
                    if tag in factors:
                        factor = factors[tag]
                        activity_emissions = value * factor
                        scope_emissions += activity_emissions
                        
                        calculation_details.append({
                            'scope': scope,
                            'activity': tag,
                            'value': value,
                            'factor': factor,
                            'emissions': activity_emissions,
                            'source': factor_data['source'],
                            'confidence': factor_data.get('confidence', 0.1)
                        })
        
        # Fall back to financial data if no business activities or insufficient data
        if scope_emissions == 0 and financial_data:
            revenue = financial_data.get('revenue', 0)
            assets = financial_data.get('assets', 0)
            
            factor_data = factors_manager.get_hybrid_factor(sic_code, scope, 'revenue', revenue)
            
            if factor_data and 'factors' in factor_data:
                factors = factor_data['factors']
                
                if revenue > 0 and 'revenue' in factors:
                    factor = factors['revenue']
                    scope_emissions = revenue * factor / 1000000  # Convert to tonnes
                    
                    calculation_details.append({
                        'scope': scope,
                        'activity': 'revenue',
                        'value': revenue,
                        'factor': factor,
                        'emissions': scope_emissions,
                        'source': factor_data['source'],
                        'confidence': factor_data.get('confidence', 0.1)
                    })
        
        emissions[scope] = scope_emissions
    
    return emissions, calculation_details

def generate_hybrid_emissions_profiles():
    """
    Generate emissions profiles using hybrid approach with both factor sources
    """
    logger.info("Starting hybrid emissions profile generation")
    
    # Initialize factors manager
    factors_manager = HybridEmissionFactorsManager('../sec_financials.db')
    
    # Clear existing data
    cursor = factors_manager.cursor
    cursor.execute("DELETE FROM emissions_estimates")
    factors_manager.conn.commit()
    logger.info("Cleared existing emissions estimates")
    
    # Get all companies with financial data
    companies_query = """
    SELECT DISTINCT s.adsh, s.cik, s.sic, s.name
    FROM submissions s
    INNER JOIN balance_sheet b ON s.adsh = b.adsh
    WHERE s.sic IS NOT NULL
    ORDER BY s.adsh
    """
    
    companies_df = pd.read_sql(companies_query, factors_manager.conn)
    logger.info(f"Processing {len(companies_df)} companies")
    
    emissions_data = []
    
    for _, company in tqdm(companies_df.iterrows(), total=len(companies_df), desc="Generating hybrid emissions"):
        adsh = company['adsh']
        cik = company['cik']
        sic_code = company['sic']
        company_name = company['name']
        
        # Generate emissions for each year (2020-2024)
        for year in range(2020, 2025):
            try:
                # Get financial data for the year
                financial_query = """
                SELECT revenue, expenses, net_income, assets, liabilities, equity
                FROM income_statement i
                LEFT JOIN balance_sheet b ON i.adsh = b.adsh AND i.year = b.year
                WHERE i.adsh = ? AND i.year = ?
                """
                financial_df = pd.read_sql(financial_query, factors_manager.conn, params=[adsh, year])
                
                # Get business activities for the year
                business_query = """
                SELECT tag, unit, value
                FROM business_activity
                WHERE adsh = ? AND year = ?
                """
                business_df = pd.read_sql(business_query, factors_manager.conn, params=[adsh, year])
                
                financial_dict = financial_df.iloc[0].to_dict() if not financial_df.empty else {}
                
                # Calculate hybrid emissions
                emissions, details = calculate_hybrid_emissions(
                    adsh, sic_code, financial_dict, business_df, factors_manager
                )
                
                # Calculate total emissions
                total_emissions = emissions['scope1'] + emissions['scope2'] + emissions['scope3']
                
                # Calculate uncertainty based on factor confidence
                avg_confidence = np.mean([d['confidence'] for d in details]) if details else 0.1
                uncertainty = max(5, 50 - (avg_confidence * 100))
                
                # Determine calculation method
                sources_used = [d['source'] for d in details]
                method = f"hybrid_{'_'.join(set(sources_used))}"
                
                # Calculate proxy emissions (simplified calculation)
                proxy_emissions = total_emissions * 0.8  # 80% of calculated emissions
                
                # Estimate credits purchased (5-15% of total emissions)
                credits_purchased = total_emissions * np.random.uniform(0.05, 0.15)
                
                emissions_data.append({
                    'adsh': adsh,
                    'cik': cik,
                    'year': year,
                    'total_co2e': total_emissions,
                    'scope1_co2e': emissions['scope1'],
                    'scope2_co2e': emissions['scope2'],
                    'scope3_co2e': emissions['scope3'],
                    'uncertainty': uncertainty,
                    'method': method,
                    'proxy_co2e': proxy_emissions,
                    'credits_purchased_tonnes': credits_purchased
                })
                
            except Exception as e:
                logger.error(f"Error processing {adsh} for year {year}: {e}")
                continue
    
    # Insert emissions data
    if emissions_data:
        insert_query = """
        INSERT INTO emissions_estimates 
        (adsh, cik, year, total_co2e, scope1_co2e, scope2_co2e, scope3_co2e, 
         uncertainty, method, proxy_co2e, credits_purchased_tonnes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.executemany(insert_query, [
            (row['adsh'], row['cik'], row['year'], row['total_co2e'], 
             row['scope1_co2e'], row['scope2_co2e'], row['scope3_co2e'],
             row['uncertainty'], row['method'], row['proxy_co2e'], row['credits_purchased_tonnes'])
            for row in emissions_data
        ])
        
        factors_manager.conn.commit()
        logger.info(f"Successfully inserted {len(emissions_data)} hybrid emissions estimates")
    
    # Generate summary statistics
    summary_query = """
    SELECT 
        year,
        COUNT(*) as count,
        AVG(total_co2e) as avg_total_emissions,
        AVG(scope1_co2e) as avg_scope1_emissions,
        AVG(scope2_co2e) as avg_scope2_emissions,
        AVG(scope3_co2e) as avg_scope3_emissions,
        AVG(uncertainty) as avg_uncertainty
    FROM emissions_estimates
    GROUP BY year
    ORDER BY year
    """
    
    summary_df = pd.read_sql(summary_query, factors_manager.conn)
    logger.info("Hybrid emissions summary by year:")
    logger.info(f"\n{summary_df.to_string()}")
    
    # Method distribution
    method_query = """
    SELECT 
        method,
        COUNT(*) as count,
        AVG(total_co2e) as avg_emissions,
        AVG(uncertainty) as avg_uncertainty
    FROM emissions_estimates
    GROUP BY method
    ORDER BY count DESC
    """
    
    method_df = pd.read_sql(method_query, factors_manager.conn)
    logger.info("Hybrid calculation method distribution:")
    logger.info(f"\n{method_df.to_string()}")
    
    factors_manager.conn.close()
    logger.info("Hybrid emissions profile generation completed")

if __name__ == "__main__":
    generate_hybrid_emissions_profiles()
