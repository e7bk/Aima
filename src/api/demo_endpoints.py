from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from src.database.connection import get_db
from src.api.models import PredictionRequest, PredictionResponse, ModelType, PredictionHorizon
from src.api.demo_model_service import demo_model_service

router = APIRouter(prefix="/predictions", tags=["predictions"])

@router.post("/", response_model=PredictionResponse, summary="Get mock stock predictions")
async def get_predictions(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Generate MOCK predictions for a stock using simulated models and horizons."""
    try:
        # Validate symbol exists (still check database for realism)
        from src.database.models import Stock, StockPrice
        stock = db.query(Stock).filter(Stock.symbol == request.symbol.upper()).first()
        if not stock:
            # For demo, create a mock stock if it doesn't exist
            logger.warning(f"Stock {request.symbol} not in database - using mock data")
            current_price = 150.0 + hash(request.symbol) % 300  # Deterministic mock price
        else:
            # Get current price from database
            latest_price = db.query(StockPrice)\
                .filter(StockPrice.stock_id == stock.id)\
                .order_by(StockPrice.date.desc()).first()
            current_price = latest_price.close if latest_price else None
        
        # Generate MOCK predictions using demo service
        predictions = await demo_model_service.predict(
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
            model_metrics={}  # Could add mock metrics here if needed
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock prediction failed: {str(e)}")

@router.get("/{symbol}", summary="Get default mock predictions for symbol")
async def get_default_predictions(
    symbol: str,
    model: ModelType = ModelType.ENSEMBLE,
    horizon: PredictionHorizon = PredictionHorizon.ONE_DAY,
    db: Session = Depends(get_db)
):
    """Get MOCK predictions for a symbol using default model and horizon."""
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
            "timestamp": response.timestamp,
            "demo_mode": True,
            "note": "This is a mock prediction for demonstration purposes"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock prediction failed: {str(e)}")

@router.post("/train/{symbol}", summary="Simulate model training for specific stock")
async def train_models_for_stock(
    symbol: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Simulate training of ML models for a specific stock - returns immediately."""
    try:
        # For demo, we don't need to validate stock exists as strictly
        
        # Add mock training task to background (will complete almost instantly)
        background_tasks.add_task(demo_model_service.train_models, symbol.upper())
        
        return {
            "message": f"DEMO: Mock training started for {symbol}",
            "symbol": symbol.upper(),
            "status": "mock_training_started",
            "timestamp": datetime.now(),
            "demo_mode": True,
            "note": "Training is simulated and will complete in ~1 second",
            "estimated_completion": "< 1 second"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock training request failed: {str(e)}")

@router.get("/models/status", summary="Get mock model training status")
async def get_model_status():
    """Get the current status of all MOCK ML models."""
    try:
        status = await demo_model_service.get_model_status()
        return {
            "status": "active",
            "models": status["models"],
            "timestamp": datetime.now(),
            "demo_mode": True,
            "note": "All models are mock/simulated - no actual ML training occurred"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock status check failed: {str(e)}")

@router.get("/models/{model}/importance/{symbol}", summary="Get mock feature importance")
async def get_feature_importance(
    model: ModelType,
    symbol: str, 
    horizon: PredictionHorizon = PredictionHorizon.ONE_DAY
):
    """Get MOCK feature importance data for a model."""
    try:
        importance = await demo_model_service.get_feature_importance(symbol, model, horizon)
        
        if importance is None:
            raise HTTPException(status_code=404, detail=f"No mock importance data for {model.value}")
        
        return {
            "symbol": symbol,
            "model": model,
            "horizon": horizon,
            "feature_importance": importance,
            "timestamp": datetime.now(),
            "demo_mode": True,
            "note": "Feature importance values are mock data for demonstration"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mock feature importance failed: {str(e)}")

# Module-level router for importing
demo_predictions = router