#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.collector import DataCollector
from src.database.connection import SessionLocal
from src.database.models import Stock, StockPrice
from src.utils.config import Config

def test_database_collector():
    """Test the full data collector with database storage"""
    print("Testing DataCollector with Database Storage")
    print("=" * 50)
    
    # Initialize collector
    collector = DataCollector()
    config = Config()
    
    print(f"Database URL: {config.database_url}")
    print(f"Default symbols: {config.DEFAULT_STOCK_SYMBOLS}")
    
    # Test with a single symbol first
    test_symbol = "AAPL"
    print(f"\n1. Testing single stock: {test_symbol}")
    
    db = SessionLocal()
    try:
        # Fetch data
        print(f"   Fetching data for {test_symbol}...")
        data = collector.fetch_stock_data(test_symbol, period="5d")
        
        if data is not None:
            print(f"   ✅ Fetched {len(data)} days of data")
            
            # Save stock info
            print(f"   Saving stock info...")
            stock = collector.save_stock_info(db, test_symbol, "Apple Inc.")
            print(f"   ✅ Stock saved with ID: {stock.id}")
            
            # Save price data
            print(f"   Saving price data...")
            collector.save_price_data(db, stock.id, data)
            print(f"   ✅ Saved {len(data)} price records")
            
            # Verify data in database
            stock_count = db.query(Stock).count()
            price_count = db.query(StockPrice).count()
            print(f"   Database now has {stock_count} stocks and {price_count} price records")
            
        else:
            print("   ❌ Failed to fetch data")
            
    finally:
        db.close()
    
    print(f"\n2. Testing collect_all method")
    try:
        collector.collect_all()
        print("   ✅ collect_all completed successfully")
        
        # Check final counts
        db = SessionLocal()
        try:
            stock_count = db.query(Stock).count()
            price_count = db.query(StockPrice).count()
            print(f"   Final database: {stock_count} stocks, {price_count} price records")
            
            # Show some sample data
            print("\n3. Sample data from database:")
            stocks = db.query(Stock).all()
            for stock in stocks[:3]:  # Show first 3 stocks
                recent_prices = db.query(StockPrice)\
                    .filter(StockPrice.stock_id == stock.id)\
                    .order_by(StockPrice.date.desc())\
                    .limit(1).all()
                
                if recent_prices:
                    price = recent_prices[0]
                    print(f"   {stock.symbol} ({stock.name}): ${price.close:.2f} on {price.date.date()}")
                    
        finally:
            db.close()
            
    except Exception as e:
        print(f"   ❌ Error in collect_all: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_collector()