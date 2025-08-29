import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from src.utils.config import Config
from src.database.models import Stock, StockPrice
from src.database.connection import get_db
from sqlalchemy.orm import Session

class DataCollector:
    """
    Service for collecting stock data from external APIs and storing in database.
    
    Handles fetching historical stock data using yfinance, saving stock metadata,
    and storing price data with duplicate prevention. Supports configurable
    time periods and symbol lists.
    
    Attributes:
        symbols (List[str]): List of stock symbols to collect data for
        days (int): Number of days of historical data to fetch
    """
    def __init__(self):
       self.symbols = Config.DEFAULT_STOCK_SYMBOLS
       self.days = Config.DAYS_OF_HISTORY
   
   def fetch_stock_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
       try:
           ticker = yf.Ticker(symbol)
           data = ticker.history(period=period)
           return data
       except Exception as e:
           print(f"Error fetching data for {symbol}: {e}")
           return None
   
   def save_stock_info(self, db: Session, symbol: str, name: str = None):
       existing = db.query(Stock).filter(Stock.symbol == symbol).first()
       if not existing:
           stock = Stock(symbol=symbol, name=name or symbol)
           db.add(stock)
           db.commit()
           return stock
       return existing
   
   def save_price_data(self, db: Session, stock_id: int, df: pd.DataFrame):
       for date, row in df.iterrows():
           existing = db.query(StockPrice).filter(
               StockPrice.stock_id == stock_id,
               StockPrice.date == date
           ).first()
           
           if not existing:
               price = StockPrice(
                   stock_id=stock_id,
                   date=date,
                   open_price=row['Open'],
                   high=row['High'],
                   low=row['Low'],
                   close=row['Close'],
                   volume=row['Volume']
               )
               db.add(price)
       db.commit()
   
   def collect_all(self):
       db = next(get_db())
       try:
           for symbol in self.symbols:
               print(f"Collecting data for {symbol}...")
               df = self.fetch_stock_data(symbol)
               if df is not None:
                   stock = self.save_stock_info(db, symbol)
                   self.save_price_data(db, stock.id, df)
                   print(f"Saved {len(df)} records for {symbol}")
       finally:
           db.close()
