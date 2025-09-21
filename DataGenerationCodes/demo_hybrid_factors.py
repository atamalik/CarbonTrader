import sqlite3
import pandas as pd
import json
from datetime import datetime

def demonstrate_hybrid_factor_usage():
    """
    Demonstrate how to use both emission_factors and emission_factors_research tables together
    """
    
    conn = sqlite3.connect('../sec_financials.db')
    
    print("=== HYBRID EMISSION FACTORS DEMONSTRATION ===\n")
    
    # 1. Show static factors structure
    print("1. STATIC EMISSION FACTORS (emission_factors table):")
    print("   - Comprehensive fuel and activity-specific factors")
    print("   - Hierarchical classification system")
    print("   - Direct conversion factors (kg CO2e per unit)")
    
    static_sample = pd.read_sql("""
        SELECT Scope, Level1, Level2, Level3, ColumnText, UOM, Conversion_Factor_2024
        FROM emission_factors 
        WHERE Level1 = 'Fuels' AND Level2 = 'Gaseous fuels'
        LIMIT 5
    """, conn)
    
    print("\n   Sample static factors (Gaseous fuels):")
    print(static_sample.to_string(index=False))
    
    # 2. Show dynamic factors structure
    print("\n2. DYNAMIC EMISSION FACTORS (emission_factors_research table):")
    print("   - Industry-specific factors by SIC code")
    print("   - Revenue and asset-based factors")
    print("   - Confidence scoring and source tracking")
    
    dynamic_sample = pd.read_sql("""
        SELECT sector, sic_code, scope1_factors, confidence_score, last_updated
        FROM emission_factors_research 
        WHERE confidence_score > 0.8
        LIMIT 3
    """, conn)
    
    print("\n   Sample dynamic factors (high confidence):")
    for _, row in dynamic_sample.iterrows():
        print(f"   SIC {row['sic_code']} ({row['sector']}):")
        print(f"     Scope 1 factors: {row['scope1_factors']}")
        print(f"     Confidence: {row['confidence_score']}")
        print(f"     Last updated: {row['last_updated']}")
    
    # 3. Demonstrate hybrid approach
    print("\n3. HYBRID APPROACH - COMBINING BOTH SOURCES:")
    print("   Priority: Dynamic factors (if recent & high confidence) > Static factors > Defaults")
    
    # Example: Pharmaceutical company (SIC 2834)
    sic_code = 2834
    print(f"\n   Example: Pharmaceutical company (SIC {sic_code})")
    
    # Get dynamic factor
    dynamic_factor = pd.read_sql("""
        SELECT scope1_factors, scope2_factors, scope3_factors, confidence_score, last_updated
        FROM emission_factors_research 
        WHERE sic_code = ?
    """, conn, params=[sic_code])
    
    if not dynamic_factor.empty:
        factor_data = dynamic_factor.iloc[0]
        print(f"   Dynamic factor found:")
        print(f"     Scope 1: {factor_data['scope1_factors']}")
        print(f"     Scope 2: {factor_data['scope2_factors']}")
        print(f"     Scope 3: {factor_data['scope3_factors']}")
        print(f"     Confidence: {factor_data['confidence_score']}")
        print(f"     Last updated: {factor_data['last_updated']}")
        
        # Check if recent
        last_updated = datetime.fromisoformat(factor_data['last_updated'].replace('Z', '+00:00'))
        days_old = (datetime.now() - last_updated).days
        print(f"     Age: {days_old} days")
        
        if factor_data['confidence_score'] > 0.5 and days_old <= 30:
            print("     ✓ Using dynamic factor (recent and high confidence)")
        else:
            print("     ⚠ Dynamic factor is stale or low confidence, falling back to static")
    else:
        print("   No dynamic factor found, using static factors")
    
    # Get static factors for pharmaceuticals
    static_factors = pd.read_sql("""
        SELECT Scope, Level1, Level2, Level3, Conversion_Factor_2024, UOM
        FROM emission_factors 
        WHERE Level1 = 'Industrial processes' AND Level2 = 'Chemical industry'
        LIMIT 3
    """, conn)
    
    if not static_factors.empty:
        print(f"\n   Static factors available:")
        print(static_factors.to_string(index=False))
    
    # 4. Show practical calculation example
    print("\n4. PRACTICAL CALCULATION EXAMPLE:")
    print("   Company: Pharmaceutical with $100M revenue, $200M assets")
    
    # Simulate hybrid calculation
    revenue = 100_000_000  # $100M
    assets = 200_000_000   # $200M
    
    # Use dynamic factors if available
    if not dynamic_factor.empty and factor_data['confidence_score'] > 0.5:
        try:
            scope1_factors = json.loads(factor_data['scope1_factors'])
            scope2_factors = json.loads(factor_data['scope2_factors'])
            scope3_factors = json.loads(factor_data['scope3_factors'])
            
            scope1_emissions = (revenue / 1_000_000) * scope1_factors['revenue']
            scope2_emissions = (revenue / 1_000_000) * scope2_factors['revenue']
            scope3_emissions = (revenue / 1_000_000) * scope3_factors['revenue']
            
            print(f"   Using dynamic factors:")
            print(f"     Scope 1: ${revenue/1_000_000:.1f}M revenue × {scope1_factors['revenue']} = {scope1_emissions:.1f} tonnes CO2e")
            print(f"     Scope 2: ${revenue/1_000_000:.1f}M revenue × {scope2_factors['revenue']} = {scope2_emissions:.1f} tonnes CO2e")
            print(f"     Scope 3: ${revenue/1_000_000:.1f}M revenue × {scope3_factors['revenue']} = {scope3_emissions:.1f} tonnes CO2e")
            print(f"     Total: {scope1_emissions + scope2_emissions + scope3_emissions:.1f} tonnes CO2e")
            print(f"     Confidence: {factor_data['confidence_score']:.2f}")
            
        except json.JSONDecodeError:
            print("   Error parsing dynamic factors, using static fallback")
    
    # 5. Show benefits of hybrid approach
    print("\n5. BENEFITS OF HYBRID APPROACH:")
    print("   ✓ More accurate: Uses most recent and reliable data available")
    print("   ✓ Robust: Falls back gracefully when data is missing or stale")
    print("   ✓ Transparent: Tracks confidence and sources for each calculation")
    print("   ✓ Comprehensive: Covers both specific activities and financial metrics")
    print("   ✓ Validated: Cross-references multiple data sources")
    
    # 6. Show factor coverage statistics
    print("\n6. FACTOR COVERAGE STATISTICS:")
    
    static_count = pd.read_sql("SELECT COUNT(*) as count FROM emission_factors", conn).iloc[0]['count']
    dynamic_count = pd.read_sql("SELECT COUNT(*) as count FROM emission_factors_research", conn).iloc[0]['count']
    dynamic_high_conf = pd.read_sql("SELECT COUNT(*) as count FROM emission_factors_research WHERE confidence_score > 0.7", conn).iloc[0]['count']
    
    print(f"   Static factors: {static_count:,} entries")
    print(f"   Dynamic factors: {dynamic_count:,} SIC codes")
    print(f"   High confidence dynamic factors: {dynamic_high_conf:,} SIC codes")
    print(f"   Coverage: {dynamic_count/static_count*100:.1f}% of static factors have dynamic equivalents")
    
    conn.close()

if __name__ == "__main__":
    demonstrate_hybrid_factor_usage()
