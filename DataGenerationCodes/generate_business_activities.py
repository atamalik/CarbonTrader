import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import lognorm, gamma, norm, weibull_min, truncnorm, expon
import logging
import json
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('Logs/business_activities.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Connect to the database
conn = sqlite3.connect('../sec_financials.db')
cursor = conn.cursor()

# Load SIC industries data
sic_industries = pd.read_csv('../CSV Files/sic_top_200_industries.csv')
logger.info(f"Loaded {len(sic_industries)} industries from SIC data")

# Industry-specific business activity tags and units
# This maps SIC codes to their specific business activities with appropriate units
industry_business_activities = {
    # Manufacturing - Pharmaceuticals
    2834: {  # Pharmaceutical Preparations
        'tags': ['DrugProductionVolume', 'ResearchAndDevelopmentExpense', 'ClinicalTrialsCount', 'PatentApplications'],
        'units': ['units', 'USD', 'trials', 'applications'],
        'value_ranges': [(1e6, 1e9), (1e7, 5e8), (10, 1000), (5, 200)]
    },
    2836: {  # Biological Products
        'tags': ['BiologicalProductVolume', 'ResearchAndDevelopmentExpense', 'BiotechPatents', 'ManufacturingCapacity'],
        'units': ['units', 'USD', 'patents', 'units_per_year'],
        'value_ranges': [(1e5, 1e8), (1e7, 3e8), (3, 150), (1e6, 1e9)]
    },
    
    # Manufacturing - Technology
    3674: {  # Semiconductors
        'tags': ['ChipProductionVolume', 'ResearchAndDevelopmentExpense', 'ManufacturingCapacity', 'TechnologyNodes'],
        'units': ['chips', 'USD', 'chips_per_year', 'nanometers'],
        'value_ranges': [(1e8, 1e12), (1e8, 1e9), (1e9, 1e12), (7, 180)]
    },
    3841: {  # Medical Instruments
        'tags': ['MedicalDeviceProduction', 'ResearchAndDevelopmentExpense', 'RegulatoryApprovals', 'QualityTests'],
        'units': ['devices', 'USD', 'approvals', 'tests'],
        'value_ranges': [(1e4, 1e7), (1e6, 1e8), (1, 50), (100, 10000)]
    },
    
    # Finance - Banking
    6022: {  # State Commercial Banks
        'tags': ['LoansHeldForInvestment', 'Deposits', 'InterestRevenueExpenseNet', 'BranchCount'],
        'units': ['USD', 'USD', 'USD', 'branches'],
        'value_ranges': [(1e8, 1e11), (1e9, 1e12), (1e7, 1e10), (10, 5000)]
    },
    6021: {  # National Commercial Banks
        'tags': ['LoansHeldForInvestment', 'Deposits', 'InterestRevenueExpenseNet', 'ATMCount'],
        'units': ['USD', 'USD', 'USD', 'ATMs'],
        'value_ranges': [(1e9, 1e12), (1e10, 1e13), (1e8, 1e11), (100, 50000)]
    },
    
    # Services - Software
    7372: {  # Prepackaged Software
        'tags': ['SoftwareLicensesSold', 'ResearchAndDevelopmentExpense', 'ActiveUsers', 'CodeLines'],
        'units': ['licenses', 'USD', 'users', 'lines'],
        'value_ranges': [(1e3, 1e8), (1e6, 1e9), (1e4, 1e9), (1e5, 1e8)]
    },
    7374: {  # Computer Processing Services
        'tags': ['DataProcessingVolume', 'ServerCapacity', 'ClientCount', 'UptimePercentage'],
        'units': ['GB_processed', 'servers', 'clients', 'percentage'],
        'value_ranges': [(1e9, 1e12), (10, 10000), (100, 100000), (95, 99.99)]  # Realistic ranges
    },
    
    # Transportation
    4512: {  # Air Transportation
        'tags': ['PassengerMiles', 'CargoTonMiles', 'FlightHours', 'AircraftCount'],
        'units': ['passenger_miles', 'ton_miles', 'hours', 'aircraft'],
        'value_ranges': [(1e8, 2e11), (1e6, 2e9), (1e4, 1e6), (5, 1000)]  # Realistic ranges
    },
    4213: {  # Trucking
        'tags': ['FreightTonMiles', 'VehicleMiles', 'FleetSize', 'DeliveryCount'],
        'units': ['ton_miles', 'miles', 'vehicles', 'deliveries'],
        'value_ranges': [(1e6, 1e10), (1e7, 1e9), (10, 10000), (1e4, 1e7)]  # Realistic ranges
    },
    
    # Utilities
    4911: {  # Electric Services
        'tags': ['ElectricGenerationMWh', 'CustomerCount', 'TransmissionMiles', 'RenewablePercentage'],
        'units': ['MWh', 'customers', 'miles', 'percentage'],
        'value_ranges': [(1e6, 1e9), (1e4, 1e6), (100, 10000), (0, 100)]  # Realistic ranges
    },
    4922: {  # Natural Gas Transmission
        'tags': ['GasTransmissionVolume', 'PipelineMiles', 'CustomerCount', 'StorageCapacity'],
        'units': ['cubic_feet', 'miles', 'customers', 'cubic_feet'],
        'value_ranges': [(1e9, 1e11), (100, 10000), (1e3, 1e5), (1e8, 1e10)]  # Realistic ranges
    },
    
    # Retail
    5812: {  # Eating Places
        'tags': ['MealsServed', 'RestaurantCount', 'AverageCheckSize', 'CustomerCount'],
        'units': ['meals', 'restaurants', 'USD', 'customers'],
        'value_ranges': [(1e5, 1e9), (1, 10000), (5, 100), (1e3, 1e8)]
    },
    5411: {  # Grocery Stores
        'tags': ['StoreCount', 'SalesVolume', 'ProductSKUs', 'CustomerTransactions'],
        'units': ['stores', 'USD', 'SKUs', 'transactions'],
        'value_ranges': [(1, 5000), (1e6, 1e10), (1000, 100000), (1e5, 1e9)]
    },
    
    # Mining
    1311: {  # Crude Petroleum and Natural Gas
        'tags': ['OilProductionBarrels', 'GasProductionCubicFeet', 'ProvedReserves', 'DrillingRigs'],
        'units': ['barrels', 'cubic_feet', 'barrels', 'rigs'],
        'value_ranges': [(1e5, 1e9), (1e8, 1e12), (1e6, 1e10), (1, 100)]
    },
    1000: {  # Metal Mining
        'tags': ['OreProductionTons', 'MetalProductionTons', 'MineCount', 'ProcessingCapacity'],
        'units': ['tons', 'tons', 'mines', 'tons_per_year'],
        'value_ranges': [(1e4, 1e8), (1e3, 1e7), (1, 50), (1e5, 1e9)]
    },
    
    # Construction
    1531: {  # Operative Builders
        'tags': ['HomesBuilt', 'ConstructionRevenue', 'ProjectCount', 'LaborHours'],
        'units': ['homes', 'USD', 'projects', 'hours'],
        'value_ranges': [(10, 10000), (1e6, 1e9), (5, 1000), (1e4, 1e7)]
    },
    
    # Wholesale Trade
    5065: {  # Electronic Parts
        'tags': ['PartsSold', 'InventoryTurnover', 'SupplierCount', 'DistributionCenters'],
        'units': ['parts', 'ratio', 'suppliers', 'centers'],
        'value_ranges': [(1e4, 1e9), (2, 20), (10, 1000), (1, 100)]
    }
}

# Default business activities for industries not specifically mapped
default_business_activities = {
    'tags': ['Revenue', 'EmployeeCount', 'AssetUtilization', 'MarketShare'],
    'units': ['USD', 'employees', 'percentage', 'percentage'],
    'value_ranges': [(1e6, 1e10), (10, 100000), (50, 95), (1, 50)]
}

def get_industry_activities(sic_code):
    """Get business activities for a specific SIC code."""
    if sic_code in industry_business_activities:
        return industry_business_activities[sic_code]
    else:
        return default_business_activities

def generate_business_activity_value(tag, unit, value_range, company_revenue, company_assets):
    """Generate realistic business activity values based on company size and industry."""
    min_val, max_val = value_range
    
    # Scale based on company size (revenue and assets) - more conservative scaling
    size_factor = min(company_revenue / 1e8, company_assets / 1e8, 2)  # Cap at 2x instead of 10x
    scaled_min = min_val * (0.5 + size_factor * 0.5)  # More conservative scaling
    scaled_max = max_val * (0.5 + size_factor * 0.5)
    
    # Generate value based on tag type
    if 'Revenue' in tag or 'USD' in unit:
        # Financial metrics - use lognormal distribution
        mean = (scaled_min + scaled_max) / 2
        std = (scaled_max - scaled_min) / 4
        sigma = np.sqrt(np.log(1 + (std / mean)**2))
        mu = np.log(mean) - 0.5 * sigma**2
        value = lognorm(s=sigma, scale=np.exp(mu)).rvs()
    elif 'Count' in tag or 'Count' in unit or 'branches' in unit or 'ATMs' in unit:
        # Count metrics - use Poisson-like distribution
        mean = (scaled_min + scaled_max) / 2
        value = np.random.poisson(mean)
    elif 'Percentage' in tag or 'percentage' in unit:
        # Percentage metrics - use beta distribution
        value = np.random.beta(2, 2) * (scaled_max - scaled_min) + scaled_min
    else:
        # Other metrics - use gamma distribution
        shape = 2
        scale = (scaled_min + scaled_max) / (2 * shape)
        value = gamma(shape, scale=scale).rvs()
    
    return max(scaled_min, min(value, scaled_max))

def get_company_sector(adsh):
    """Get the sector for a company."""
    cursor.execute("SELECT sector FROM balance_sheet WHERE adsh = ? LIMIT 1", (adsh,))
    result = cursor.fetchone()
    return result[0] if result else 'Other'

def generate_historical_business_activity_value(base_value, year, sector, tag, unit, volatility=0.20):
    """Generate historical business activity value based on economic factors and trends."""
    if year == 2024:
        return base_value
    
    # Economic factors for business activities (similar to financial data)
    economic_factors = {
        2020: {
            'overall_growth': -0.05,  # COVID-19 impact
            'sector_multipliers': {
                'Manufacturing': 0.85,  # Supply chain disruptions
                'Finance, Insurance & Real Estate': 0.90,  # Market volatility
                'Services': 0.80,  # Service sector hit hard
                'Transportation and Public Utilities': 0.75,  # Travel restrictions
                'Mining': 0.70,  # Reduced demand
                'Retail Trade': 0.85,  # Store closures
                'Wholesale Trade': 0.80,  # Supply chain issues
                'Construction': 0.75,  # Project delays
                'Agriculture': 0.95,  # Essential services
                'Other': 0.85
            }
        },
        2021: {
            'overall_growth': 0.15,  # Recovery year
            'sector_multipliers': {
                'Manufacturing': 1.20,  # Supply chain recovery
                'Finance, Insurance & Real Estate': 1.15,  # Market recovery
                'Services': 1.10,  # Gradual recovery
                'Transportation and Public Utilities': 1.25,  # Pent-up demand
                'Mining': 1.30,  # Commodity boom
                'Retail Trade': 1.20,  # E-commerce growth
                'Wholesale Trade': 1.15,  # Supply chain normalization
                'Construction': 1.10,  # Infrastructure spending
                'Agriculture': 1.05,  # Stable growth
                'Other': 1.10
            }
        },
        2022: {
            'overall_growth': 0.08,  # Moderate growth
            'sector_multipliers': {
                'Manufacturing': 1.05,  # Normalization
                'Finance, Insurance & Real Estate': 0.95,  # Interest rate impact
                'Services': 1.08,  # Continued recovery
                'Transportation and Public Utilities': 1.10,  # Steady growth
                'Mining': 1.15,  # Continued commodity strength
                'Retail Trade': 1.05,  # Normalization
                'Wholesale Trade': 1.08,  # Steady growth
                'Construction': 1.12,  # Infrastructure projects
                'Agriculture': 1.08,  # Food security focus
                'Other': 1.05
            }
        },
        2023: {
            'overall_growth': 0.12,  # Strong growth
            'sector_multipliers': {
                'Manufacturing': 1.15,  # Reshoring trends
                'Finance, Insurance & Real Estate': 1.10,  # Market stability
                'Services': 1.12,  # Digital transformation
                'Transportation and Public Utilities': 1.08,  # Steady growth
                'Mining': 1.20,  # Energy transition
                'Retail Trade': 1.15,  # Omnichannel growth
                'Wholesale Trade': 1.12,  # Supply chain optimization
                'Construction': 1.18,  # Green building
                'Agriculture': 1.10,  # Sustainability focus
                'Other': 1.12
            }
        }
    }
    
    # Get economic factors for the year
    factors = economic_factors[year]
    sector_multiplier = factors['sector_multipliers'].get(sector, 1.0)
    overall_growth = factors['overall_growth']
    
    # Calculate compound growth from 2024 backwards
    years_from_2024 = 2024 - year
    growth_factor = (1 + overall_growth) ** years_from_2024
    sector_factor = sector_multiplier ** (1/years_from_2024) if years_from_2024 > 0 else 1.0
    
    # Add some randomness for realism
    random_factor = np.random.normal(1.0, volatility)
    
    historical_value = base_value * growth_factor * sector_factor * random_factor
    
    # Ensure non-negative values
    return max(0, historical_value)

def populate_business_activities():
    """Populate business_activity table with industry-specific data for 2020-2024."""
    logger.info("Starting business activities generation for 2020-2024")
    
    # Get all companies with their SIC codes and financial data
    query = """
    SELECT s.adsh, s.sic, 
           COALESCE(i.revenue, 0) as revenue,
           COALESCE(b.assets, 0) as assets
    FROM submissions s
    LEFT JOIN income_statement i ON s.adsh = i.adsh AND i.year = 2024
    LEFT JOIN balance_sheet b ON s.adsh = b.adsh AND b.year = 2024
    WHERE s.sic IS NOT NULL
    """
    
    companies_df = pd.read_sql(query, conn)
    logger.info(f"Processing {len(companies_df)} companies")
    
    business_activities_data = []
    
    for _, company in tqdm(companies_df.iterrows(), total=len(companies_df), desc="Generating business activities"):
        adsh = company['adsh']
        sic = company['sic']
        revenue = company['revenue'] or 1e6  # Default if missing
        assets = company['assets'] or 1e6   # Default if missing
        sector = get_company_sector(adsh)
        
        # Get industry-specific activities
        activities = get_industry_activities(sic)
        
        # Generate business activities for each year 2020-2024
        for year in range(2020, 2025):
            for i, tag in enumerate(activities['tags']):
                unit = activities['units'][i]
                value_range = activities['value_ranges'][i]
                
                # Generate base value for 2024
                base_value = generate_business_activity_value(tag, unit, value_range, revenue, assets)
                
                # Generate historical value for the specific year
                historical_value = generate_historical_business_activity_value(
                    base_value, year, sector, tag, unit
                )
                
                business_activities_data.append({
                    'adsh': adsh,
                    'year': year,
                    'tag': tag,
                    'unit': unit,
                    'value': historical_value
                })
    
    # Insert data into database
    df = pd.DataFrame(business_activities_data)
    df.to_sql('business_activity', conn, if_exists='append', index=False)
    conn.commit()
    
    logger.info(f"Successfully inserted {len(df)} business activity records")
    
    # Validation
    result = pd.read_sql("SELECT year, COUNT(*) as count FROM business_activity GROUP BY year ORDER BY year", conn)
    logger.info("Business activity records by year:")
    logger.info(result.to_string(index=False))
    
    # Show sample data
    sample = pd.read_sql("SELECT * FROM business_activity WHERE year = 2024 LIMIT 10", conn)
    logger.info("Sample 2024 business activity data:")
    logger.info(sample.to_string())

if __name__ == '__main__':
    try:
        populate_business_activities()
    except Exception as e:
        logger.error(f"Error generating business activities: {e}")
    finally:
        conn.close()
        logger.info("Database connection closed")
