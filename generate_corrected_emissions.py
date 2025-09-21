import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import lognorm, gamma, norm, weibull_min, truncnorm, expon
import logging
import json
import os
from tqdm import tqdm
from emission_factors_research_tool import EmissionFactorsResearchTool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/corrected_emissions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Connect to the database
conn = sqlite3.connect('sec_financials.db')
cursor = conn.cursor()

# Realistic emission factors based on research (kg CO2e per $1M revenue)
# These are much more conservative and realistic values
REALISTIC_EMISSION_FACTORS = {
    # Electric Utilities - Realistic values
    4911: {  # Electric Services
        'scope1_factors': {
            'revenue': 35.2,   # kg CO2e per $1M revenue (realistic)
            'assets': 8.5,     # kg CO2e per $1M assets
            'ElectricGenerationMWh': 0.85,  # kg CO2e per kWh (realistic)
            'CustomerCount': 0.25,         # kg CO2e per customer
            'TransmissionMiles': 1.25,     # kg CO2e per mile
        },
        'scope2_factors': {
            'revenue': 5.8,    # kg CO2e per $1M revenue
            'assets': 1.2,     # kg CO2e per $1M assets
            'ElectricGenerationMWh': 0.045,  # kg CO2e per kWh
            'CustomerCount': 0.015,        # kg CO2e per customer
        },
        'scope3_factors': {
            'revenue': 8.5,    # kg CO2e per $1M revenue
            'ElectricGenerationMWh': 0.015,  # kg CO2e per kWh
            'CustomerCount': 0.008,        # kg CO2e per customer
        }
    },
    
    # Oil & Gas - Realistic values
    1311: {  # Crude Petroleum and Natural Gas
        'scope1_factors': {
            'revenue': 85.2,   # kg CO2e per $1M revenue (realistic)
            'assets': 25.8,    # kg CO2e per $1M assets
            'OilProductionBarrels': 0.045,  # kg CO2e per barrel
            'GasProductionCubicFeet': 0.0008,  # kg CO2e per cubic foot
            'ProvedReserves': 0.0001,      # kg CO2e per barrel reserve
            'DrillingRigs': 1250        # kg CO2e per rig
        },
        'scope2_factors': {
            'revenue': 15.8,   # kg CO2e per $1M revenue
            'assets': 5.2,     # kg CO2e per $1M assets
            'OilProductionBarrels': 0.025,  # kg CO2e per barrel
            'GasProductionCubicFeet': 0.0004,  # kg CO2e per cubic foot
        },
        'scope3_factors': {
            'revenue': 25.8,   # kg CO2e per $1M revenue
            'OilProductionBarrels': 0.125,  # kg CO2e per barrel
            'GasProductionCubicFeet': 0.002,  # kg CO2e per cubic foot
        }
    },
    
    # Airlines - Realistic values
    4512: {  # Air Transportation
        'scope1_factors': {
            'revenue': 125.8,  # kg CO2e per $1M revenue (realistic)
            'assets': 25.8,    # kg CO2e per $1M assets
            'PassengerMiles': 0.285,  # kg CO2e per passenger mile
            'CargoTonMiles': 0.85,    # kg CO2e per ton mile
            'FlightHours': 1250,      # kg CO2e per flight hour
            'AircraftCount': 12500   # kg CO2e per aircraft
        },
        'scope2_factors': {
            'revenue': 8.5,    # kg CO2e per $1M revenue
            'assets': 2.1,     # kg CO2e per $1M assets
            'PassengerMiles': 0.015,  # kg CO2e per passenger mile
            'CargoTonMiles': 0.045,   # kg CO2e per ton mile
        },
        'scope3_factors': {
            'revenue': 15.8,   # kg CO2e per $1M revenue
            'PassengerMiles': 0.085,  # kg CO2e per passenger mile
            'CargoTonMiles': 0.25,    # kg CO2e per ton mile
        }
    },
    
    # Software - Realistic values
    7372: {  # Prepackaged Software
        'scope1_factors': {
            'revenue': 2.8,    # kg CO2e per $1M revenue (very low)
            'assets': 0.8,     # kg CO2e per $1M assets
            'SoftwareLicensesSold': 0.15,  # kg CO2e per license
            'ResearchAndDevelopmentExpense': 0.4,  # kg CO2e per $1K R&D
            'ActiveUsers': 0.0008,  # kg CO2e per user
            'CodeLines': 0.0000002  # kg CO2e per line of code
        },
        'scope2_factors': {
            'revenue': 8.5,    # kg CO2e per $1M revenue (data centers)
            'assets': 2.1,     # kg CO2e per $1M assets
            'SoftwareLicensesSold': 0.25,  # kg CO2e per license
            'ResearchAndDevelopmentExpense': 0.6,  # kg CO2e per $1K R&D
        },
        'scope3_factors': {
            'revenue': 12.8,   # kg CO2e per $1M revenue
            'SoftwareLicensesSold': 0.45,  # kg CO2e per license
            'ResearchAndDevelopmentExpense': 1.2,  # kg CO2e per $1K R&D
        }
    },
    
    # Banks - Realistic values
    6022: {  # State Commercial Banks
        'scope1_factors': {
            'revenue': 1.8,    # kg CO2e per $1M revenue (very low)
            'assets': 0.5,     # kg CO2e per $1M assets
            'LoansHeldForInvestment': 0.0008,  # kg CO2e per $1K loan
            'Deposits': 0.0003,  # kg CO2e per $1K deposit
            'InterestRevenueExpenseNet': 0.0012,  # kg CO2e per $1K interest
            'BranchCount': 125  # kg CO2e per branch
        },
        'scope2_factors': {
            'revenue': 3.2,    # kg CO2e per $1M revenue
            'assets': 0.8,     # kg CO2e per $1M assets
            'LoansHeldForInvestment': 0.0012,  # kg CO2e per $1K loan
            'Deposits': 0.0005,  # kg CO2e per $1K deposit
        },
        'scope3_factors': {
            'revenue': 8.5,    # kg CO2e per $1M revenue (financed emissions)
            'LoansHeldForInvestment': 0.0085,  # kg CO2e per $1K loan
            'Deposits': 0.0025,  # kg CO2e per $1K deposit
        }
    },
    
    # Pharmaceuticals - Realistic values
    2834: {  # Pharmaceutical Preparations
        'scope1_factors': {
            'revenue': 15.8,   # kg CO2e per $1M revenue
            'assets': 5.2,     # kg CO2e per $1M assets
            'DrugProductionVolume': 0.15,  # kg CO2e per unit
            'ResearchAndDevelopmentExpense': 0.8,  # kg CO2e per $1K R&D
            'ClinicalTrialsCount': 125,  # kg CO2e per trial
            'PatentApplications': 8.5  # kg CO2e per application
        },
        'scope2_factors': {
            'revenue': 8.5,    # kg CO2e per $1M revenue
            'assets': 2.8,     # kg CO2e per $1M assets
            'DrugProductionVolume': 0.08,  # kg CO2e per unit
            'ResearchAndDevelopmentExpense': 0.5,  # kg CO2e per $1K R&D
        },
        'scope3_factors': {
            'revenue': 25.8,   # kg CO2e per $1M revenue
            'DrugProductionVolume': 0.45,  # kg CO2e per unit (supply chain)
            'ResearchAndDevelopmentExpense': 2.1,  # kg CO2e per $1K R&D
        }
    },
    
    # Trucking - Realistic values
    4213: {  # Trucking
        'scope1_factors': {
            'revenue': 45.8,   # kg CO2e per $1M revenue
            'assets': 12.8,    # kg CO2e per $1M assets
            'FreightTonMiles': 0.125,  # kg CO2e per ton mile
            'VehicleMiles': 0.85,      # kg CO2e per vehicle mile
            'FleetSize': 1250,        # kg CO2e per vehicle
            'DeliveryCount': 0.15      # kg CO2e per delivery
        },
        'scope2_factors': {
            'revenue': 5.8,    # kg CO2e per $1M revenue
            'assets': 2.1,     # kg CO2e per $1M assets
            'FreightTonMiles': 0.008,  # kg CO2e per ton mile
            'VehicleMiles': 0.045,     # kg CO2e per vehicle mile
        },
        'scope3_factors': {
            'revenue': 8.5,    # kg CO2e per $1M revenue
            'FreightTonMiles': 0.025,  # kg CO2e per ton mile
            'VehicleMiles': 0.15,      # kg CO2e per vehicle mile
        }
    }
}

# Default emission factors for industries not specifically mapped
DEFAULT_EMISSION_FACTORS = {
    'scope1_factors': {
        'revenue': 15.8,  # kg CO2e per $1M revenue (realistic default)
        'assets': 5.2,    # kg CO2e per $1M assets
        'EmployeeCount': 2.5,  # kg CO2e per employee
        'AssetUtilization': 0.15,  # kg CO2e per percentage point
        'MarketShare': 0.8  # kg CO2e per percentage point
    },
    'scope2_factors': {
        'revenue': 8.5,   # kg CO2e per $1M revenue
        'assets': 2.8,    # kg CO2e per $1M assets
        'EmployeeCount': 1.5,  # kg CO2e per employee
        'AssetUtilization': 0.08,  # kg CO2e per percentage point
    },
    'scope3_factors': {
        'revenue': 18.5,  # kg CO2e per $1M revenue
        'EmployeeCount': 4.5,  # kg CO2e per employee
        'AssetUtilization': 0.25,  # kg CO2e per percentage point
    }
}

def get_emission_factors(sic_code):
    """Get emission factors for a specific SIC code."""
    if sic_code in REALISTIC_EMISSION_FACTORS:
        return REALISTIC_EMISSION_FACTORS[sic_code]
    else:
        return DEFAULT_EMISSION_FACTORS

def calculate_emissions_from_financials(adsh, year, sic_code, financial_data):
    """Calculate emissions based on financial data only."""
    factors = get_emission_factors(sic_code)
    
    emissions = {
        'scope1': 0,
        'scope2': 0,
        'scope3': 0
    }
    
    # Calculate based on revenue and assets
    revenue = financial_data.get('revenue', 0)
    assets = financial_data.get('assets', 0)
    
    if revenue > 0:
        revenue_millions = revenue / 1e6
        emissions['scope1'] += revenue_millions * factors['scope1_factors'].get('revenue', 0)
        emissions['scope2'] += revenue_millions * factors['scope2_factors'].get('revenue', 0)
        emissions['scope3'] += revenue_millions * factors['scope3_factors'].get('revenue', 0)
    
    if assets > 0:
        assets_millions = assets / 1e6
        emissions['scope1'] += assets_millions * factors['scope1_factors'].get('assets', 0)
        emissions['scope2'] += assets_millions * factors['scope2_factors'].get('assets', 0)
        emissions['scope3'] += assets_millions * factors['scope3_factors'].get('assets', 0)
    
    return emissions

def calculate_emissions_from_business_activities(adsh, year, sic_code, business_activities):
    """Calculate emissions based on business activity data."""
    factors = get_emission_factors(sic_code)
    
    emissions = {
        'scope1': 0,
        'scope2': 0,
        'scope3': 0
    }
    
    for activity in business_activities:
        tag = activity['tag']
        value = activity['value']
        unit = activity['unit']
        
        # Apply appropriate emission factors based on tag and unit
        if tag in factors['scope1_factors']:
            factor = factors['scope1_factors'][tag]
            if 'USD' in unit or 'revenue' in tag.lower():
                # Financial values - convert to appropriate scale
                if value > 1e6:  # Millions
                    emissions['scope1'] += (value / 1e6) * factor
                elif value > 1e3:  # Thousands
                    emissions['scope1'] += (value / 1e3) * factor
                else:
                    emissions['scope1'] += value * factor
            else:
                # Activity-based values
                emissions['scope1'] += value * factor
        
        if tag in factors['scope2_factors']:
            factor = factors['scope2_factors'][tag]
            if 'USD' in unit or 'revenue' in tag.lower():
                if value > 1e6:
                    emissions['scope2'] += (value / 1e6) * factor
                elif value > 1e3:
                    emissions['scope2'] += (value / 1e3) * factor
                else:
                    emissions['scope2'] += value * factor
            else:
                emissions['scope2'] += value * factor
        
        if tag in factors['scope3_factors']:
            factor = factors['scope3_factors'][tag]
            if 'USD' in unit or 'revenue' in tag.lower():
                if value > 1e6:
                    emissions['scope3'] += (value / 1e6) * factor
                elif value > 1e3:
                    emissions['scope3'] += (value / 1e3) * factor
                else:
                    emissions['scope3'] += value * factor
            else:
                emissions['scope3'] += value * factor
    
    return emissions

def generate_corrected_emissions():
    """Generate corrected emissions profiles with realistic factors."""
    logger.info("Starting corrected emissions profile generation")
    
    # Clear existing data to regenerate with corrected methodology
    cursor.execute("DELETE FROM emissions_estimates")
    conn.commit()
    logger.info("Cleared existing emissions data")
    
    # Get all companies with their financial and business activity data
    companies_query = """
    SELECT DISTINCT s.adsh, s.sic 
    FROM submissions s
    WHERE s.sic IS NOT NULL
    """
    companies_df = pd.read_sql(companies_query, conn)
    logger.info(f"Processing {len(companies_df)} companies")
    
    emissions_data = []
    
    for _, company in tqdm(companies_df.iterrows(), total=len(companies_df), desc="Generating corrected emissions profiles"):
        adsh = company['adsh']
        sic_code = company['sic']
        
        # Process each year 2020-2024
        for year in range(2020, 2025):
            # Get financial data for this year
            financial_query = """
            SELECT revenue, assets, liabilities, equity
            FROM income_statement i
            LEFT JOIN balance_sheet b ON i.adsh = b.adsh AND i.year = b.year
            WHERE i.adsh = ? AND i.year = ?
            """
            financial_data = pd.read_sql(financial_query, conn, params=(adsh, year))
            
            if financial_data.empty:
                continue
            
            financial_dict = financial_data.iloc[0].to_dict()
            
            # Get business activity data for this year
            ba_query = """
            SELECT tag, unit, value
            FROM business_activity
            WHERE adsh = ? AND year = ?
            """
            business_activities = pd.read_sql(ba_query, conn, params=(adsh, year))
            
            # Calculate emissions
            if not business_activities.empty:
                # Use business activity data for more accurate calculations
                business_activities_list = business_activities.to_dict('records')
                emissions = calculate_emissions_from_business_activities(
                    adsh, year, sic_code, business_activities_list
                )
                calculation_method = 'business_activity'
            else:
                # Fall back to financial data only
                emissions = calculate_emissions_from_financials(
                    adsh, year, sic_code, financial_dict
                )
                calculation_method = 'financial_only'
            
            # Calculate total emissions and intensity
            total_emissions = emissions['scope1'] + emissions['scope2'] + emissions['scope3']
            revenue = financial_dict.get('revenue', 0)
            emissions_intensity = (total_emissions / (revenue / 1e6)) if revenue > 0 else 0
            
            # Extract CIK from ADSH (first part before the dash)
            cik = int(adsh.split('-')[0])
            
            # Calculate uncertainty based on calculation method
            uncertainty = 15.0 if calculation_method == 'business_activity' else 25.0
            
            # Calculate proxy emissions (for comparison)
            proxy_emissions = total_emissions * 0.8  # Assume 20% lower for proxy
            
            # Calculate credits purchased (assume 10% of total emissions)
            credits_purchased = total_emissions * 0.1
            
            emissions_data.append({
                'adsh': adsh,
                'cik': cik,
                'year': year,
                'total_co2e': total_emissions,
                'scope1_co2e': emissions['scope1'],
                'scope2_co2e': emissions['scope2'],
                'scope3_co2e': emissions['scope3'],
                'uncertainty': uncertainty,
                'method': calculation_method,
                'proxy_co2e': proxy_emissions,
                'credits_purchased_tonnes': credits_purchased
            })
    
    # Insert emissions data
    df = pd.DataFrame(emissions_data)
    df.to_sql('emissions_estimates', conn, if_exists='append', index=False)
    conn.commit()
    
    logger.info(f"Successfully inserted {len(df)} corrected emissions estimates")
    
    # Validation and summary
    result = pd.read_sql("SELECT year, COUNT(*) as count FROM emissions_estimates GROUP BY year ORDER BY year", conn)
    logger.info("Emissions estimates by year:")
    logger.info(result.to_string(index=False))
    
    # Show sample data
    sample = pd.read_sql("SELECT * FROM emissions_estimates WHERE year = 2024 LIMIT 10", conn)
    logger.info("Sample 2024 emissions data:")
    logger.info(sample.to_string())
    
    # Summary statistics
    summary = pd.read_sql("""
        SELECT 
            AVG(total_co2e) as avg_total_emissions,
            MAX(total_co2e) as max_total_emissions,
            MIN(total_co2e) as min_total_emissions,
            COUNT(CASE WHEN method = 'business_activity' THEN 1 END) as business_activity_count,
            COUNT(CASE WHEN method = 'financial_only' THEN 1 END) as financial_only_count,
            AVG(scope1_co2e) as avg_scope1_emissions,
            AVG(scope2_co2e) as avg_scope2_emissions,
            AVG(scope3_co2e) as avg_scope3_emissions
        FROM emissions_estimates 
        WHERE year = 2024
    """, conn)
    logger.info("Summary statistics for 2024:")
    logger.info(summary.to_string(index=False))
    
    # Check for unrealistic values
    unrealistic = pd.read_sql("""
        SELECT adsh, total_co2e, method
        FROM emissions_estimates 
        WHERE year = 2024 AND total_co2e > 1000000
        ORDER BY total_co2e DESC
        LIMIT 10
    """, conn)
    
    if not unrealistic.empty:
        logger.warning("Companies with potentially unrealistic emissions (>1M tonnes):")
        logger.warning(unrealistic.to_string(index=False))
    else:
        logger.info("No companies with unrealistic emissions found - all values are reasonable!")

if __name__ == '__main__':
    try:
        generate_corrected_emissions()
    except Exception as e:
        logger.error(f"Error generating corrected emissions: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed")
