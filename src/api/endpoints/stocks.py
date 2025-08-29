from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from src.database.connection import get_db
from src.database.models import Stock, StockPrice
from src.api.models import (
    StockRequest, StocksListResponse, AvailableStock, 
    HistoricalDataPoint, ChartDataResponse, ErrorResponse
)

router = APIRouter(prefix="/stocks", tags=["stocks"])

@router.get("/", response_model=StocksListResponse, summary="List all available stocks")
async def list_stocks(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of stocks to return"),
    offset: int = Query(0, ge=0, description="Number of stocks to skip"),
    db: Session = Depends(get_db)
):
    """
    Get a list of all available stocks in the database.
    
    Returns stock symbols, names, latest prices, and data availability.
    """
    try:
        # Get total count
        total_count = db.query(Stock).count()
        
        # Get stocks with pagination
        stocks = db.query(Stock).offset(offset).limit(limit).all()
        
        available_stocks = []
        for stock in stocks:
            # Get latest price data
            latest_price = db.query(StockPrice)\
                .filter(StockPrice.stock_id == stock.id)\
                .order_by(StockPrice.date.desc())\
                .first()
            
            # Count total records
            total_records = db.query(StockPrice)\
                .filter(StockPrice.stock_id == stock.id)\
                .count()
            
            available_stock = AvailableStock(
                symbol=stock.symbol,
                name=stock.name,
                last_price=latest_price.close if latest_price else None,
                last_updated=latest_price.date if latest_price else stock.created_at,
                total_records=total_records
            )
            available_stocks.append(available_stock)
        
        return StocksListResponse(
            stocks=available_stocks,
            total_count=total_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stocks: {str(e)}")

@router.get("/{symbol}/latest", summary="Get latest stock price")
async def get_latest_price(
    symbol: str,
    db: Session = Depends(get_db)
):
    """
    Get the most recent price data for a stock.
    """
    try:
        # Find stock
        stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get latest price
        latest_price = db.query(StockPrice)\
            .filter(StockPrice.stock_id == stock.id)\
            .order_by(StockPrice.date.desc())\
            .first()
        
        if not latest_price:
            raise HTTPException(status_code=404, detail=f"No price data available for {symbol}")
        
        return {
            "symbol": stock.symbol,
            "name": stock.name,
            "current_price": latest_price.close,
            "date": latest_price.date,
            "open": latest_price.open_price,
            "high": latest_price.high,
            "low": latest_price.low,
            "volume": latest_price.volume
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve latest price: {str(e)}")