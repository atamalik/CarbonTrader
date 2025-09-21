import sqlite3
import pandas as pd
import json
import numpy as np
from datetime import datetime

def demonstrate_complementary_factor_usage():
    """
    Demonstrate how to use emission_factors and emission_factors_research tables 
    in a complementary way for more accurate emissions calculations
    """
    
    conn = sqlite3.connect('../sec_financials.db')
    
    print("=== COMPLEMENTARY EMISSION FACTORS APPROACH ===\n")
    
    # 1. Explain the complementary roles
    print("1. COMPLEMENTARY ROLES OF BOTH TABLES:")
    print("   emission_factors (Static):")
    print("     - Specific fuel/activity conversion factors")
    print("     - Direct calculations (kg CO2e per unit)")
    print("     - Detailed, granular data")
    print("     - Use for: Fuel consumption, energy use, transportation")
    
    print("\n   emission_factors_research (Dynamic):")
    print("     - Industry-level validation factors")
    print("     - Revenue/asset-based benchmarks")
    print("     - Confidence scoring and freshness")
    print("     - Use for: Validation, correction, industry benchmarks")
    
    # 2. Show practical example: Manufacturing company
    print("\n2. PRACTICAL EXAMPLE: Manufacturing Company (SIC 2834 - Pharmaceuticals)")
    
    # Get company data
    company_data = pd.read_sql("""
        SELECT s.adsh, s.name, s.sic, 
               i.revenue, i.expenses, i.net_income,
               b.assets, b.liabilities, b.equity
        FROM submissions s
        LEFT JOIN income_statement i ON s.adsh = i.adsh AND i.year = 2024
        LEFT JOIN balance_sheet b ON s.adsh = b.adsh AND b.year = 2024
        WHERE s.sic = 2834 AND i.revenue IS NOT NULL
        LIMIT 1
    """, conn)
    
    if not company_data.empty:
        company = company_data.iloc[0]
        print(f"   Company: {company['name']}")
        print(f"   Revenue: ${company['revenue']:,.0f}")
        print(f"   Assets: ${company['assets']:,.0f}")
        
        # Get business activities
        activities = pd.read_sql("""
            SELECT tag, unit, value
            FROM business_activity
            WHERE adsh = ? AND year = 2024
        """, conn, params=[company['adsh']])
        
        print(f"   Business activities: {len(activities)} activities tracked")
        
        # 3. Step 1: Calculate emissions using static factors (detailed approach)
        print("\n3. STEP 1: DETAILED CALCULATION USING STATIC FACTORS")
        print("   (For specific activities and fuel consumption)")
        
        detailed_emissions = {'scope1': 0, 'scope2': 0, 'scope3': 0}
        calculation_details = []
        
        # Map business activities to static factors
        activity_mapping = {
            'DrugProductionVolume': ('Industrial processes', 'Chemical industry', 'Pharmaceuticals'),
            'ResearchAndDevelopmentExpense': ('Industrial processes', 'Chemical industry', 'R&D'),
            'EnergyConsumption': ('Fuels', 'Gaseous fuels', 'Natural gas'),
            'TransportationMiles': ('Transport', 'Road transport', 'Freight trucks'),
            'WasteGenerated': ('Waste', 'Industrial waste', 'Hazardous waste')
        }
        
        for _, activity in activities.iterrows():
            tag = activity['tag']
            value = activity['value']
            unit = activity['unit']
            
            if tag in activity_mapping:
                level1, level2, level3 = activity_mapping[tag]
                
                # Get static factor
                static_factor = pd.read_sql("""
                    SELECT Conversion_Factor_2024, UOM, GHG_Unit
                    FROM emission_factors
                    WHERE Level1 = ? AND Level2 = ? AND Level3 = ?
                    LIMIT 1
                """, conn, params=[level1, level2, level3])
                
                if not static_factor.empty:
                    factor = static_factor.iloc[0]['Conversion_Factor_2024']
                    emissions = value * factor
                    
                    # Determine scope based on activity type
                    if 'Energy' in tag or 'Fuel' in tag:
                        scope = 'scope1'
                    elif 'Transportation' in tag:
                        scope = 'scope3'
                    else:
                        scope = 'scope1'  # Default for manufacturing activities
                    
                    detailed_emissions[scope] += emissions
                    
                    calculation_details.append({
                        'activity': tag,
                        'value': value,
                        'unit': unit,
                        'factor': factor,
                        'emissions': emissions,
                        'scope': scope,
                        'source': 'static_factor'
                    })
                    
                    print(f"     {tag}: {value:,.0f} {unit} × {factor:.3f} = {emissions:,.1f} kg CO2e ({scope})")
        
        total_detailed = sum(detailed_emissions.values())
        print(f"   Total detailed emissions: {total_detailed:,.1f} kg CO2e")
        
        # 4. Step 2: Get industry benchmark using dynamic factors
        print("\n4. STEP 2: INDUSTRY BENCHMARK USING DYNAMIC FACTORS")
        print("   (For validation and correction)")
        
        dynamic_factor = pd.read_sql("""
            SELECT scope1_factors, scope2_factors, scope3_factors, 
                   confidence_score, last_updated, sources
            FROM emission_factors_research
            WHERE sic_code = 2834
        """, conn)
        
        if not dynamic_factor.empty:
            factor_data = dynamic_factor.iloc[0]
            
            # Parse JSON factors
            scope1_factors = json.loads(factor_data['scope1_factors'])
            scope2_factors = json.loads(factor_data['scope2_factors'])
            scope3_factors = json.loads(factor_data['scope3_factors'])
            
            # Calculate benchmark emissions
            revenue_millions = company['revenue'] / 1_000_000
            assets_millions = company['assets'] / 1_000_000
            
            benchmark_emissions = {
                'scope1': revenue_millions * scope1_factors['revenue'],
                'scope2': revenue_millions * scope2_factors['revenue'],
                'scope3': revenue_millions * scope3_factors['revenue']
            }
            
            total_benchmark = sum(benchmark_emissions.values())
            
            print(f"   Industry benchmark (revenue-based):")
            print(f"     Scope 1: ${revenue_millions:.1f}M × {scope1_factors['revenue']} = {benchmark_emissions['scope1']:,.1f} kg CO2e")
            print(f"     Scope 2: ${revenue_millions:.1f}M × {scope2_factors['revenue']} = {benchmark_emissions['scope2']:,.1f} kg CO2e")
            print(f"     Scope 3: ${revenue_millions:.1f}M × {scope3_factors['revenue']} = {benchmark_emissions['scope3']:,.1f} kg CO2e")
            print(f"     Total benchmark: {total_benchmark:,.1f} kg CO2e")
            print(f"     Confidence: {factor_data['confidence_score']:.2f}")
            
            # 5. Step 3: Compare and apply correction logic
            print("\n5. STEP 3: COMPARISON AND CORRECTION LOGIC")
            
            if total_detailed > 0:
                percentage_diff = abs(total_detailed - total_benchmark) / total_benchmark * 100
                print(f"   Difference: {percentage_diff:.1f}%")
                
                if percentage_diff > 50:  # Material difference
                    print("   ⚠ Material difference detected - applying correction")
                    
                    confidence = factor_data['confidence_score']
                    
                    if confidence > 0.8:
                        print("   ✓ High confidence benchmark - using benchmark values")
                        final_emissions = benchmark_emissions
                        correction_method = "high_confidence_benchmark"
                    elif confidence > 0.5:
                        print("   ⚖ Medium confidence - blending detailed and benchmark")
                        blend_ratio = confidence
                        final_emissions = {
                            'scope1': detailed_emissions['scope1'] * (1 - blend_ratio) + benchmark_emissions['scope1'] * blend_ratio,
                            'scope2': detailed_emissions['scope2'] * (1 - blend_ratio) + benchmark_emissions['scope2'] * blend_ratio,
                            'scope3': detailed_emissions['scope3'] * (1 - blend_ratio) + benchmark_emissions['scope3'] * blend_ratio
                        }
                        correction_method = f"blended_{blend_ratio:.2f}"
                    else:
                        print("   ⚠ Low confidence benchmark - using detailed calculation")
                        final_emissions = detailed_emissions
                        correction_method = "detailed_calculation"
                else:
                    print("   ✓ Good agreement - using detailed calculation")
                    final_emissions = detailed_emissions
                    correction_method = "detailed_calculation"
            else:
                print("   ⚠ No detailed calculation available - using benchmark")
                final_emissions = benchmark_emissions
                correction_method = "benchmark_only"
            
            # 6. Final results
            print("\n6. FINAL RESULTS:")
            total_final = sum(final_emissions.values())
            print(f"   Final emissions: {total_final:,.1f} kg CO2e")
            print(f"   Correction method: {correction_method}")
            print(f"   Scope breakdown:")
            print(f"     Scope 1: {final_emissions['scope1']:,.1f} kg CO2e")
            print(f"     Scope 2: {final_emissions['scope2']:,.1f} kg CO2e")
            print(f"     Scope 3: {final_emissions['scope3']:,.1f} kg CO2e")
            
            # Calculate uncertainty
            if correction_method == "high_confidence_benchmark":
                uncertainty = 10  # Low uncertainty for high confidence
            elif "blended" in correction_method:
                uncertainty = 20  # Medium uncertainty for blended
            else:
                uncertainty = 15  # Standard uncertainty for detailed calculation
            
            print(f"   Uncertainty: ±{uncertainty}%")
    
    # 7. Show benefits of complementary approach
    print("\n7. BENEFITS OF COMPLEMENTARY APPROACH:")
    print("   ✓ Accuracy: Detailed calculations for specific activities")
    print("   ✓ Validation: Industry benchmarks for reality checks")
    print("   ✓ Robustness: Graceful handling of missing data")
    print("   ✓ Transparency: Clear audit trail of calculations")
    print("   ✓ Confidence: Uncertainty quantification based on data quality")
    print("   ✓ Flexibility: Adapts to available data quality")
    
    # 8. Show coverage analysis
    print("\n8. COVERAGE ANALYSIS:")
    
    # Count companies with business activities
    companies_with_activities = pd.read_sql("""
        SELECT COUNT(DISTINCT adsh) as count
        FROM business_activity
    """, conn).iloc[0]['count']
    
    # Count companies with dynamic factors
    companies_with_dynamic = pd.read_sql("""
        SELECT COUNT(DISTINCT s.adsh) as count
        FROM submissions s
        INNER JOIN emission_factors_research efr ON s.sic = efr.sic_code
    """, conn).iloc[0]['count']
    
    total_companies = pd.read_sql("""
        SELECT COUNT(DISTINCT adsh) as count
        FROM submissions
    """, conn).iloc[0]['count']
    
    print(f"   Companies with business activities: {companies_with_activities:,}")
    print(f"   Companies with dynamic factors: {companies_with_dynamic:,}")
    print(f"   Total companies: {total_companies:,}")
    print(f"   Coverage for detailed calculation: {companies_with_activities/total_companies*100:.1f}%")
    print(f"   Coverage for validation: {companies_with_dynamic/total_companies*100:.1f}%")
    
    conn.close()

if __name__ == "__main__":
    demonstrate_complementary_factor_usage()
