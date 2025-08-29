import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
import random
from pathlib import Path

from src.database.connection import SessionLocal
from src.api.models import ModelType, PredictionHorizon, PredictionResult, ModelMetrics
from src.database.models import Stock

logger = logging.getLogger(__name__)

class DemoModelService:
    """Lightweight demo service for generating mock predictions without ML overhead"""
    
    def __init__(self):
        self.models = {}
        self.prediction_horizons = [1, 5, 10]
        self.model_metrics = {}
        self._initialized = True  # Always ready for demo
        
        # Pre-define realistic mock metrics for consistency
        self._mock_metrics = {
            ModelType.RANDOM_FOREST: {"rmse": 0.045, "mae": 0.032, "r2_score": 0.78, "accuracy": 0.68},
            ModelType.GRADIENT_BOOSTING: {"rmse": 0.041, "mae": 0.029, "r2_score": 0.82, "accuracy": 0.71},
            ModelType.LSTM: {"rmse": 0.038, "mae": 0.027, "r2_score": 0.85, "accuracy": 0.74},
            ModelType.BI_LSTM: {"rmse": 0.036, "mae": 0.025, "r2_score": 0.87, "accuracy": 0.76},
            ModelType.ENSEMBLE: {"rmse": 0.033, "mae": 0.023, "r2_score": 0.89, "accuracy": 0.79}
        }
        
        # Stock-specific factors to make predictions more realistic
        self._stock_volatility = {
            "AAPL": 0.25,
            "GOOGL": 0.28, 
            "MSFT": 0.22,
            "TSLA": 0.45,
            "AMZN": 0.30,
            "NVDA": 0.35,
            "META": 0.32,
            "JPM": 0.18,
            "V": 0.20,
            "JNJ": 0.15
        }
        
    async def initialize_models(self) -> None:
        """Demo initialization - instantly ready"""
        if self._initialized:
            return
            
        logger.info("Initializing demo model service (no ML models loaded)...")
        self._initialized = True
        logger.info("Demo service ready - using mock predictions")
    
    async def train_models(self, symbol: str) -> Dict[str, Any]:
        """Mock training - returns fake training results instantly"""
        logger.info(f"Demo mode: Simulating training for {symbol}...")
        
        # Simulate brief training delay for realism
        await asyncio.sleep(0.5)
        
        # Return mock training results
        training_results = {}
        
        for model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, 
                          ModelType.LSTM, ModelType.BI_LSTM, ModelType.ENSEMBLE]:
            
            # Generate realistic mock training metrics
            base_loss = 0.1 + random.uniform(-0.02, 0.02)
            final_loss = base_loss * 0.3 + random.uniform(-0.01, 0.01)
            
            if model_type in [ModelType.LSTM, ModelType.BI_LSTM]:
                # Simulate LSTM training history
                epochs = random.randint(15, 30)
                history = {
                    "loss": [base_loss * (0.95 ** i) + random.uniform(-0.005, 0.005) for i in range(epochs)],
                    "val_loss": [base_loss * (0.93 ** i) + random.uniform(-0.008, 0.008) for i in range(epochs)],
                    "epochs": epochs,
                    "best_epoch": epochs - random.randint(3, 8)
                }
            else:
                # Traditional model results
                history = {
                    "training_score": 0.85 + random.uniform(-0.05, 0.05),
                    "validation_score": 0.78 + random.uniform(-0.08, 0.08),
                    "n_estimators": 100,
                    "feature_importance": [random.uniform(0, 1) for _ in range(20)]
                }
            
            training_results[model_type.value] = history
        
        logger.info(f"Demo training completed for {symbol}")
        return training_results
    
    def _generate_mock_prediction(self, symbol: str, model_type: ModelType, horizon: int) -> float:
        """Generate realistic mock prediction based on symbol and model characteristics"""
        
        # Base volatility for the symbol
        volatility = self._stock_volatility.get(symbol, 0.25)
        
        # Model-specific biases (some models tend to be more/less optimistic)
        model_bias = {
            ModelType.RANDOM_FOREST: -0.002,
            ModelType.GRADIENT_BOOSTING: 0.001,
            ModelType.LSTM: 0.003,
            ModelType.BI_LSTM: 0.002,
            ModelType.ENSEMBLE: 0.0
        }
        
        # Horizon-specific scaling (longer term predictions are less certain)
        horizon_scale = {1: 1.0, 5: 0.8, 10: 0.6}
        
        # Generate prediction: small bias + random component scaled by volatility and horizon
        base_return = model_bias.get(model_type, 0.0)
        random_component = random.gauss(0, volatility * horizon_scale.get(horizon, 0.5) * 0.1)
        
        prediction = base_return + random_component
        
        # Clamp predictions to reasonable range (-15% to +15%)
        return max(-0.15, min(0.15, prediction))
    
    def _calculate_confidence(self, prediction: float, model_type: ModelType) -> float:
        """Calculate realistic confidence score"""
        # Base confidence varies by model type
        base_confidence = {
            ModelType.RANDOM_FOREST: 0.65,
            ModelType.GRADIENT_BOOSTING: 0.68,
            ModelType.LSTM: 0.72,
            ModelType.BI_LSTM: 0.74,
            ModelType.ENSEMBLE: 0.78
        }
        
        # Higher magnitude predictions get lower confidence
        magnitude_penalty = abs(prediction) * 2
        confidence = base_confidence.get(model_type, 0.65) - magnitude_penalty
        
        # Ensure confidence stays in reasonable range
        return max(0.5, min(0.95, confidence))
    
    async def predict(self, symbol: str, horizons: List[PredictionHorizon], 
                     models: List[ModelType]) -> List[PredictionResult]:
        """Generate mock predictions instantly"""
        
        # Small delay to simulate processing
        await asyncio.sleep(0.1)
        
        predictions = []
        
        for model_type in models:
            for horizon_str in horizons:
                horizon = int(horizon_str.value.replace('d', ''))
                
                # Generate mock prediction
                predicted_return = self._generate_mock_prediction(symbol, model_type, horizon)
                confidence = self._calculate_confidence(predicted_return, model_type)
                
                # Create prediction result
                result = PredictionResult(
                    model=model_type,
                    horizon=horizon_str,
                    predicted_return=round(predicted_return, 4),
                    direction="up" if predicted_return > 0 else "down",
                    probability=round(confidence, 3),
                    confidence_score=round(confidence, 3)
                )
                predictions.append(result)
        
        return predictions
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get mock model status - all models always ready"""
        status = {
            "initialized": True,
            "models": {}
        }
        
        for model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, 
                          ModelType.LSTM, ModelType.BI_LSTM, ModelType.ENSEMBLE]:
            
            model_status = {
                "is_trained": True,  # Always trained in demo
                "prediction_horizons": self.prediction_horizons,
                "model_name": model_type.value,
                "demo_mode": True,
                "last_training": "Demo - No actual training",
                "performance_metrics": self._mock_metrics[model_type]
            }
            
            status["models"][model_type.value] = model_status
        
        return status
    
    async def get_feature_importance(self, symbol: str, model_type: ModelType, 
                                   horizon: PredictionHorizon) -> Optional[Dict[str, float]]:
        """Return mock feature importance data"""
        
        # Mock feature names (typical technical indicators)
        feature_names = [
            "close_price", "volume", "rsi_14", "macd", "bollinger_upper", 
            "bollinger_lower", "sma_20", "sma_50", "ema_12", "ema_26",
            "atr_14", "stoch_k", "stoch_d", "williams_r", "cci_14",
            "momentum_10", "roc_10", "price_change_1d", "volume_sma_20", "volatility_20"
        ]
        
        # Generate mock importance scores (some features more important than others)
        importances = []
        for i, name in enumerate(feature_names):
            # Make price-related features generally more important
            if "price" in name or "close" in name or "sma" in name or "ema" in name:
                importance = random.uniform(0.08, 0.15)
            elif "volume" in name:
                importance = random.uniform(0.04, 0.08)  
            else:
                importance = random.uniform(0.01, 0.06)
            importances.append((name, importance))
        
        # Sort by importance and normalize
        importances.sort(key=lambda x: x[1], reverse=True)
        total_importance = sum(imp for _, imp in importances)
        
        feature_dict = {}
        for name, imp in importances:
            feature_dict[name] = round(imp / total_importance, 4)
        
        return feature_dict

# Create demo service instance
demo_model_service = DemoModelService()