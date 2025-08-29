from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Stock(Base):
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)  # AAPL, GOOGL, etc
    name = Column(String(200))  # Apple Inc, Alphabet Inc
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Link to price data
    prices = relationship("StockPrice", back_populates="stock")

class StockPrice(Base):
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'))
    date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    
    stock = relationship("Stock", back_populates="prices")
