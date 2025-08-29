from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from src.database.connection import get_db
from src.api.models import PredictionRequest, PredictionResponse, ModelType, PredictionHorizon
from src.api.model_service import model_service

router = APIRouter(prefix="/predictions", tags=["predictions"])

@router.post("/", response_model=PredictionResponse, summary="Get stock predictions")
async def get_predictions(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Generate predictions for a stock using specified models and horizons."""
    try:
        # Validate symbol exists
        from src.database.models import Stock, StockPrice
        stock = db.query(Stock).filter(Stock.symbol == request.symbol.upper()).first()
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {request.symbol} not found")
        
        # Get current price
        latest_price = db.query(StockPrice)\
            .filter(StockPrice.stock_id == stock.id)\
            .order_by(StockPrice.date.desc()).first()
        
        current_price = latest_price.close if latest_price else None
        
        # Generate predictions
        predictions = await model_service.predict(
            symbol=request.symbol.upper(),
            horizons=request.horizons,
            models=request.models
        )
        
        # Create response
        response = PredictionResponse(
            symbol=request.symbol.upper(),
            current_price=current_price,
            timestamp=datetime.now(),
            predictions=predictions,
            model_metrics={}
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/{symbol}", summary="Get default predictions for symbol")
async def get_default_predictions(
    symbol: str,
    model: ModelType = ModelType.ENSEMBLE,
    horizon: PredictionHorizon = PredictionHorizon.ONE_DAY,
    db: Session = Depends(get_db)
):
    """Get predictions for a symbol using default model and horizon."""
    try:
        request = PredictionRequest(
            symbol=symbol,
            horizons=[horizon],
            models=[model]
        )
        
        response = await get_predictions(request, db)
        
        prediction = response.predictions[0] if response.predictions else None
        
        return {
            "symbol": symbol,
            "current_price": response.current_price,
            "prediction": prediction.dict() if prediction else None,
            "timestamp": response.timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/train/{symbol}", summary="Train models for specific stock")
async def train_models_for_stock(
    symbol: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger training of ML models for a specific stock."""
    try:
        # Validate symbol exists
        from src.database.models import Stock
        stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Add training task to background
        background_tasks.add_task(model_service.train_models, symbol.upper())
        
        return {
            "message": f"Model training started for {symbol}",
            "symbol": symbol.upper(),
            "status": "training_started",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training request failed: {str(e)}")

@router.get("/models/status", summary="Get model training status")
async def get_model_status():
    """Get the current status of all ML models."""
    try:
        status = await model_service.get_model_status()
        return {
            "status": "active" if status["initialized"] else "initializing",
            "models": status["models"],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")