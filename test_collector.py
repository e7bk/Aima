#!/usr/bin/env python3

import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_stock_data(symbol: str, period: str = "1y"):
    """Test basic stock data fetching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def test_basic_fetch():
    """Test basic data fetching for a single stock"""
    print("Testing data collection...")
    
    # Test with a single symbol first
    test_symbol = "AAPL"
    print(f"\nFetching data for {test_symbol}...")
    
    data = fetch_stock_data(test_symbol, period="5d")
    
    if data is not None:
        print(f"✅ Successfully fetched {len(data)} days of data")
        print("\nSample data:")
        print(data.head())
        print("\nColumn info:")
        print(data.dtypes)
        print(f"\nDate range: {data.index.min()} to {data.index.max()}")
    else:
        print("❌ Failed to fetch data")

def test_multiple_symbols():
    """Test fetching data for multiple symbols"""    
    symbols = ["AAPL", "GOOGL", "MSFT"]
    print(f"\nTesting multiple symbols: {symbols}")
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        data = fetch_stock_data(symbol, period="1d")
        
        if data is not None:
            print(f"✅ {symbol}: {len(data)} records")
            if len(data) > 0:
                latest = data.iloc[-1]
                print(f"   Latest close: ${latest['Close']:.2f}")
        else:
            print(f"❌ {symbol}: Failed")

if __name__ == "__main__":
    print("Stock Data Collector Test")
    print("=" * 40)
    
    try:
        test_basic_fetch()
        test_multiple_symbols()
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()