import sqlite3
import pandas as pd
import numpy as np
import logging
import json
from tqdm import tqdm
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/emissions_complementary.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComplementaryEmissionCalculator:
    """
    Uses both static and dynamic emission factors in a complementary way:
    - Static factors: For detailed activity-based calculations
    - Dynamic factors: For validation and correction
    """
    
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Activity to static factor mapping
        self.activity_mapping = {
            'DrugProductionVolume': ('Industrial processes', 'Chemical industry', 'Pharmaceuticals'),
            'ResearchAndDevelopmentExpense': ('Industrial processes', 'Chemical industry', 'R&D'),
            'EnergyConsumption': ('Fuels', 'Gaseous fuels', 'Natural gas'),
            'TransportationMiles': ('Transport', 'Road transport', 'Freight trucks'),
            'WasteGenerated': ('Waste', 'Industrial waste', 'Hazardous waste'),
            'WaterConsumption': ('Utilities', 'Water supply', 'Industrial water'),
            'RawMaterialUsage': ('Industrial processes', 'Chemical industry', 'Raw materials'),
            'ManufacturingOutput': ('Industrial processes', 'Manufacturing', 'Production'),
            'OfficeSpace': ('Buildings', 'Commercial', 'Office buildings'),
            'VehicleFleet': ('Transport', 'Road transport', 'Passenger cars')
        }
    
    def get_static_factor(self, activity_type):
        """Get static emission factor for a specific activity"""
        if activity_type not in self.activity_mapping:
            return None
        
        level1, level2, level3 = self.activity_mapping[activity_type]
        
        query = """
        SELECT Conversion_Factor_2024, UOM, GHG_Unit
        FROM emission_factors
        WHERE Level1 = ? AND Level2 = ? AND Level3 = ?
        LIMIT 1
        """
        
        result = pd.read_sql(query, self.conn, params=[level1, level2, level3])
        
        if not result.empty:
            return {
                'factor': result.iloc[0]['Conversion_Factor_2024'],
                'unit': result.iloc[0]['UOM'],
                'ghg_unit': result.iloc[0]['GHG_Unit']
            }
        
        return None
    
    def get_dynamic_factor(self, sic_code):
        """Get dynamic emission factor for validation"""
        query = """
        SELECT scope1_factors, scope2_factors, scope3_factors, 
               confidence_score, last_updated, sources
        FROM emission_factors_research
        WHERE sic_code = ?
        """
        
        result = pd.read_sql(query, self.conn, params=[sic_code])
        
        if not result.empty:
            factor_data = result.iloc[0]
            return {
                'scope1_factors': json.loads(factor_data['scope1_factors']),
                'scope2_factors': json.loads(factor_data['scope2_factors']),
                'scope3_factors': json.loads(factor_data['scope3_factors']),
                'confidence_score': factor_data['confidence_score'],
                'last_updated': factor_data['last_updated'],
                'sources': json.loads(factor_data['sources']) if factor_data['sources'] else []
            }
        
        return None
    
    def calculate_detailed_emissions(self, adsh, year):
        """Calculate emissions using static factors for specific activities"""
        # Get business activities
        activities_query = """
        SELECT tag, unit, value
        FROM business_activity
        WHERE adsh = ? AND year = ?
        """
        
        activities = pd.read_sql(activities_query, self.conn, params=[adsh, year])
        
        emissions = {'scope1': 0, 'scope2': 0, 'scope3': 0}
        calculation_details = []
        
        for _, activity in activities.iterrows():
            tag = activity['tag']
            value = activity['value']
            unit = activity['unit']
            
            # Get static factor
            static_factor = self.get_static_factor(tag)
            
            if static_factor:
                factor = static_factor['factor']
                activity_emissions = value * factor
                
                # Determine scope based on activity type
                if 'Energy' in tag or 'Fuel' in tag:
                    scope = 'scope1'
                elif 'Transportation' in tag or 'Transport' in tag:
                    scope = 'scope3'
                elif 'Waste' in tag:
                    scope = 'scope1'
                else:
                    scope = 'scope1'  # Default for manufacturing activities
                
                emissions[scope] += activity_emissions
                
                calculation_details.append({
                    'activity': tag,
                    'value': value,
                    'unit': unit,
                    'factor': factor,
                    'emissions': activity_emissions,
                    'scope': scope,
                    'source': 'static_factor'
                })
        
        return emissions, calculation_details
    
    def calculate_benchmark_emissions(self, sic_code, financial_data):
        """Calculate benchmark emissions using dynamic factors"""
        dynamic_factor = self.get_dynamic_factor(sic_code)
        
        if not dynamic_factor:
            return None
        
        revenue = financial_data.get('revenue', 0)
        assets = financial_data.get('assets', 0)
        
        if revenue <= 0:
            return None
        
        revenue_millions = revenue / 1_000_000
        
        benchmark_emissions = {
            'scope1': revenue_millions * dynamic_factor['scope1_factors']['revenue'],
            'scope2': revenue_millions * dynamic_factor['scope2_factors']['revenue'],
            'scope3': revenue_millions * dynamic_factor['scope3_factors']['revenue']
        }
        
        return {
            'emissions': benchmark_emissions,
            'confidence': dynamic_factor['confidence_score'],
            'last_updated': dynamic_factor['last_updated']
        }
    
    def apply_correction_logic(self, detailed_emissions, benchmark_data, calculation_details):
        """Apply correction logic based on comparison between detailed and benchmark calculations"""
        
        if not benchmark_data:
            # No benchmark available, use detailed calculation
            return detailed_emissions, "detailed_calculation", 15
        
        benchmark_emissions = benchmark_data['emissions']
        confidence = benchmark_data['confidence']
        
        total_detailed = sum(detailed_emissions.values())
        total_benchmark = sum(benchmark_emissions.values())
        
        if total_detailed == 0:
            # No detailed calculation available, use benchmark
            return benchmark_emissions, "benchmark_only", 20
        
        # Calculate percentage difference
        percentage_diff = abs(total_detailed - total_benchmark) / total_benchmark * 100
        
        if percentage_diff > 50:  # Material difference
            if confidence > 0.8:
                # High confidence benchmark - use benchmark
                return benchmark_emissions, "high_confidence_benchmark", 10
            elif confidence > 0.5:
                # Medium confidence - blend
                blend_ratio = confidence
                blended_emissions = {
                    'scope1': detailed_emissions['scope1'] * (1 - blend_ratio) + benchmark_emissions['scope1'] * blend_ratio,
                    'scope2': detailed_emissions['scope2'] * (1 - blend_ratio) + benchmark_emissions['scope2'] * blend_ratio,
                    'scope3': detailed_emissions['scope3'] * (1 - blend_ratio) + benchmark_emissions['scope3'] * blend_ratio
                }
                return blended_emissions, f"blended_{blend_ratio:.2f}", 20
            else:
                # Low confidence benchmark - use detailed
                return detailed_emissions, "detailed_calculation", 15
        else:
            # Good agreement - use detailed calculation
            return detailed_emissions, "detailed_calculation", 15
    
    def calculate_emissions(self, adsh, sic_code, year):
        """Main method to calculate emissions using complementary approach"""
        
        # Get financial data
        financial_query = """
        SELECT revenue, expenses, net_income, assets, liabilities, equity
        FROM income_statement i
        LEFT JOIN balance_sheet b ON i.adsh = b.adsh AND i.year = b.year
        WHERE i.adsh = ? AND i.year = ?
        """
        
        financial_df = pd.read_sql(financial_query, self.conn, params=[adsh, year])
        financial_data = financial_df.iloc[0].to_dict() if not financial_df.empty else {}
        
        # Step 1: Calculate detailed emissions using static factors
        detailed_emissions, calculation_details = self.calculate_detailed_emissions(adsh, year)
        
        # Step 2: Calculate benchmark emissions using dynamic factors
        benchmark_data = self.calculate_benchmark_emissions(sic_code, financial_data)
        
        # Step 3: Apply correction logic
        final_emissions, method, uncertainty = self.apply_correction_logic(
            detailed_emissions, benchmark_data, calculation_details
        )
        
        return final_emissions, method, uncertainty, calculation_details

def generate_complementary_emissions():
    """Generate emissions using complementary approach"""
    
    logger.info("Starting complementary emissions generation")
    
    # Initialize calculator
    calculator = ComplementaryEmissionCalculator('../sec_financials.db')
    
    # Clear existing data
    calculator.cursor.execute("DELETE FROM emissions_estimates")
    calculator.conn.commit()
    logger.info("Cleared existing emissions estimates")
    
    # Get all companies
    companies_query = """
    SELECT DISTINCT s.adsh, s.cik, s.sic, s.name
    FROM submissions s
    INNER JOIN balance_sheet b ON s.adsh = b.adsh
    WHERE s.sic IS NOT NULL
    ORDER BY s.adsh
    """
    
    companies_df = pd.read_sql(companies_query, calculator.conn)
    logger.info(f"Processing {len(companies_df)} companies")
    
    emissions_data = []
    
    for _, company in tqdm(companies_df.iterrows(), total=len(companies_df), desc="Generating complementary emissions"):
        adsh = company['adsh']
        cik = company['cik']
        sic_code = company['sic']
        
        # Generate emissions for each year (2020-2024)
        for year in range(2020, 2025):
            try:
                final_emissions, method, uncertainty, details = calculator.calculate_emissions(
                    adsh, sic_code, year
                )
                
                total_emissions = sum(final_emissions.values())
                
                # Calculate proxy emissions and credits
                proxy_emissions = total_emissions * 0.8
                credits_purchased = total_emissions * np.random.uniform(0.05, 0.15)
                
                emissions_data.append({
                    'adsh': adsh,
                    'cik': cik,
                    'year': year,
                    'total_co2e': total_emissions,
                    'scope1_co2e': final_emissions['scope1'],
                    'scope2_co2e': final_emissions['scope2'],
                    'scope3_co2e': final_emissions['scope3'],
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
        
        calculator.cursor.executemany(insert_query, [
            (row['adsh'], row['cik'], row['year'], row['total_co2e'], 
             row['scope1_co2e'], row['scope2_co2e'], row['scope3_co2e'],
             row['uncertainty'], row['method'], row['proxy_co2e'], row['credits_purchased_tonnes'])
            for row in emissions_data
        ])
        
        calculator.conn.commit()
        logger.info(f"Successfully inserted {len(emissions_data)} complementary emissions estimates")
    
    # Generate summary
    summary_query = """
    SELECT 
        method,
        COUNT(*) as count,
        AVG(total_co2e) as avg_emissions,
        AVG(uncertainty) as avg_uncertainty
    FROM emissions_estimates
    GROUP BY method
    ORDER BY count DESC
    """
    
    summary_df = pd.read_sql(summary_query, calculator.conn)
    logger.info("Complementary calculation method distribution:")
    logger.info(f"\n{summary_df.to_string()}")
    
    calculator.conn.close()
    logger.info("Complementary emissions generation completed")

if __name__ == "__main__":
    generate_complementary_emissions()
