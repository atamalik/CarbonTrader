#!/usr/bin/env python3
"""
Demo script to show real-time price updates
This demonstrates how the carbon prices update dynamically
"""

import time
import os
import sys
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from market_data import get_carbon_prices

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_price_display(credit_type, data):
    """Format price data for display"""
    price = data.get('price', 0)
    change = data.get('change', 0)
    change_pct = data.get('change_pct', 0)
    volume = data.get('volume', 0)
    source = data.get('source', 'Unknown')
    timestamp = data.get('timestamp', '')
    
    # Determine color and arrow
    if change > 0:
        color = "🟢"
        arrow = "▲"
        change_str = f"+{change:.2f}"
    elif change < 0:
        color = "🔴"
        arrow = "▼"
        change_str = f"{change:.2f}"
    else:
        color = "⚪"
        arrow = "●"
        change_str = "0.00"
    
    return f"{color} {credit_type:12} ${price:6.2f} {arrow} {change_str:>6} ({change_pct:+5.2f}%) Vol: {volume:>8,} {source}"

def main():
    """Main demo function"""
    print("🚀 ACCMN Real-Time Carbon Price Demo")
    print("=" * 80)
    print("Watch the prices update in real-time with dynamic simulation!")
    print("Press Ctrl+C to stop the demo")
    print("=" * 80)
    
    try:
        while True:
            clear_screen()
            
            print("🚀 ACCMN Real-Time Carbon Price Demo")
            print("=" * 80)
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Get current prices
            prices = get_carbon_prices()
            
            if prices:
                print("📈 LIVE CARBON PRICES TICKER")
                print("-" * 80)
                
                for credit_type, data in prices.items():
                    print(format_price_display(credit_type, data))
                
                print("-" * 80)
                
                # Show simulation details
                simulated_count = sum(1 for data in prices.values() if data.get('data_type') == 'simulated')
                live_count = len(prices) - simulated_count
                
                print(f"📊 Status: {live_count} Live APIs, {simulated_count} Simulated")
                print(f"🔄 Auto-refreshing every 3 seconds...")
                print("=" * 80)
                
            else:
                print("❌ No price data available")
            
            # Wait before next update
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo stopped. Thanks for watching!")
        print("The ACCMN system provides real-time price updates with:")
        print("• Dynamic market simulation")
        print("• Visual price animations")
        print("• Auto-refresh capabilities")
        print("• Professional trading desk experience")

if __name__ == "__main__":
    main()
