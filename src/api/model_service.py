import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio
import logging
from pathlib import Path

from src.database.connection import SessionLocal
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMStockPredictor, BidirectionalLSTMPredictor
from src.models.random_forest import RandomForestStockPredictor, GradientBoostingStockPredictor
from src.models.ensemble import EnsembleStockPredictor
from src.api.models import ModelType, PredictionHorizon, PredictionResult, ModelMetrics
from src.database.models import Stock

logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing and serving ML models"""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor(scaler_type='standard')
        self.prediction_horizons = [1, 5, 10]
        self.model_metrics = {}
        self._initialized = False
        
    async def initialize_models(self) -> None:
        """Initialize all ML models"""
        if self._initialized:
            return
            
        try:
            logger.info("Initializing ML models...")
            
            # Initialize models with same configuration as training
            self.models = {
                ModelType.RANDOM_FOREST: RandomForestStockPredictor(
                    prediction_horizons=self.prediction_horizons,
                    n_estimators=100,
                    max_depth=15,
                    random_state=42
                ),
                ModelType.GRADIENT_BOOSTING: GradientBoostingStockPredictor(
                    prediction_horizons=self.prediction_horizons,
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=42
                ),
                ModelType.LSTM: LSTMStockPredictor(
                    prediction_horizons=self.prediction_horizons,
                    lstm_units=[64, 32],
                    epochs=30,
                    patience=8,
                    sequence_length=60
                ),
                ModelType.BI_LSTM: BidirectionalLSTMPredictor(
                    prediction_horizons=self.prediction_horizons,
                    lstm_units=[32, 16],
                    epochs=20,
                    patience=6,
                    sequence_length=30
                ),
                ModelType.ENSEMBLE: EnsembleStockPredictor(
                    prediction_horizons=self.prediction_horizons,
                    base_models=[
                        RandomForestStockPredictor(
                            prediction_horizons=self.prediction_horizons,
                            n_estimators=80,
                            random_state=42
                        ),
                        GradientBoostingStockPredictor(
                            prediction_horizons=self.prediction_horizons,
                            n_estimators=80,
                            random_state=42
                        )
                    ],
                    ensemble_method='weighted_average'
                )
            }
            
            # Try to load pre-trained models if available
            await self._load_pretrained_models()
            
            self._initialized = True
            logger.info(f"Successfully initialized {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _load_pretrained_models(self) -> None:
        """Load pre-trained models from disk if available"""
        models_dir = Path("data/models")
        
        if not models_dir.exists():
            logger.info("No pre-trained models found. Models will need to be trained.")
            return
        
        for model_type, model in self.models.items():
            model_file = models_dir / f"{model_type.value.lower()}_model.pkl"
            
            if model_file.exists():
                try:
                    model.load_model(str(model_file))
                    logger.info(f"Loaded pre-trained {model_type.value} model")
                except Exception as e:
                    logger.warning(f"Failed to load {model_type.value} model: {e}")
    
    async def train_models(self, symbol: str) -> Dict[str, Any]:
        """Train all models for a specific symbol"""
        logger.info(f"Training models for {symbol}...")
        
        db = SessionLocal()
        try:
            # Prepare data
            data = self.preprocessor.process_stock_for_ml(
                db, symbol,
                target_column='target_return_1d',
                test_size=0.2,
                scale_features=True,
                sequence_length=60
            )
            
            # Prepare targets for all horizons
            targets = {}
            features_df = data['features_df']
            
            for horizon in self.prediction_horizons:
                target_col = f'target_return_{horizon}d'
                if target_col in features_df.columns:
                    X_all, y_all = self.preprocessor.prepare_ml_dataset(features_df, target_col)
                    split_idx = len(data['X_train'])
                    targets[horizon] = {
                        'y_train': y_all.iloc[:split_idx],
                        'y_test': y_all.iloc[split_idx:]
                    }
            
            training_results = {}
            
            # Train traditional models
            traditional_models = [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, ModelType.ENSEMBLE]
            
            for model_type in traditional_models:
                if model_type not in self.models:
                    continue
                
                try:
                    model = self.models[model_type]
                    
                    # Prepare target dictionaries
                    y_train_dict = {h: targets[h]['y_train'] for h in self.prediction_horizons if h in targets}
                    y_test_dict = {h: targets[h]['y_test'] for h in self.prediction_horizons if h in targets}
                    
                    # Train model
                    histories = model.train_multiple_horizons(
                        data['X_train_scaled'].values, y_train_dict,
                        data['X_test_scaled'].values, y_test_dict
                    )
                    
                    training_results[model_type.value] = histories
                    logger.info(f"Successfully trained {model_type.value}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type.value}: {e}")
                    training_results[model_type.value] = {'error': str(e)}
            
            # Train sequence models (LSTM)
            sequence_models = [ModelType.LSTM, ModelType.BI_LSTM]
            
            for model_type in sequence_models:
                if model_type not in self.models:
                    continue
                
                try:
                    model = self.models[model_type]
                    
                    # Prepare sequence targets
                    sequence_length = data['sequence_length']
                    
                    for horizon in self.prediction_horizons:
                        if horizon in targets:
                            y_train_full = targets[horizon]['y_train']
                            y_test_full = targets[horizon]['y_test']
                            
                            if len(y_train_full) > sequence_length:
                                y_train = y_train_full.iloc[sequence_length:].values
                                y_test = y_test_full.iloc[sequence_length:].values if len(y_test_full) > sequence_length else None
                                
                                X_train = data['X_seq_train']
                                X_test = data['X_seq_test'] if y_test is not None else None
                                
                                # Ensure data alignment
                                min_samples = min(len(X_train), len(y_train))
                                if min_samples >= 50:
                                    history = model.train(
                                        X_train[:min_samples], 
                                        y_train[:min_samples],
                                        X_test, y_test, horizon
                                    )
                                    
                                    if model_type.value not in training_results:
                                        training_results[model_type.value] = {}
                                    training_results[model_type.value][f'{horizon}d'] = history
                    
                    logger.info(f"Successfully trained {model_type.value}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type.value}: {e}")
                    training_results[model_type.value] = {'error': str(e)}
            
            # Save models
            await self._save_models()
            
            return training_results
            
        finally:
            db.close()
    
    async def _save_models(self) -> None:
        """Save trained models to disk"""
        models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_type, model in self.models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                try:
                    model_file = models_dir / f"{model_type.value.lower()}_model.pkl"
                    model.save_model(str(model_file))
                    logger.info(f"Saved {model_type.value} model")
                except Exception as e:
                    logger.warning(f"Failed to save {model_type.value} model: {e}")
    
    async def predict(self, symbol: str, horizons: List[PredictionHorizon], 
                     models: List[ModelType]) -> List[PredictionResult]:
        """Generate predictions for given symbol"""
        if not self._initialized:
            await self.initialize_models()
        
        db = SessionLocal()
        try:
            # Load and prepare data
            raw_data = self.preprocessor.load_stock_data(db, symbol)
            if len(raw_data) < 50:
                raise ValueError(f"Insufficient data for {symbol}: {len(raw_data)} records")
            
            features_df = self.preprocessor.create_features(raw_data, symbol)
            
            predictions = []
            
            for model_type in models:
                if model_type not in self.models:
                    continue
                
                model = self.models[model_type]
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    # Try to train on-the-fly for this symbol
                    logger.info(f"Model {model_type.value} not trained. Training now...")
                    await self.train_models(symbol)
                
                try:
                    for horizon_str in horizons:
                        horizon = int(horizon_str.value.replace('d', ''))
                        
                        # Prepare input data based on model type
                        if model_type in [ModelType.LSTM, ModelType.BI_LSTM]:
                            # Sequence models
                            X, _ = self.preprocessor.prepare_ml_dataset(
                                features_df, f'target_return_{horizon}d'
                            )
                            X_scaled, _ = self.preprocessor.scale_features(X, fit_scaler=False)
                            
                            # Create sequences
                            sequence_length = getattr(model, 'sequence_length', 60)
                            if len(X_scaled) >= sequence_length:
                                X_seq = X_scaled.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
                                prediction = model.predict(X_seq, horizon)[0]
                            else:
                                continue
                        else:
                            # Traditional models
                            X, _ = self.preprocessor.prepare_ml_dataset(
                                features_df, f'target_return_{horizon}d'
                            )
                            X_scaled, _ = self.preprocessor.scale_features(X, fit_scaler=False)
                            
                            # Use latest data point
                            X_latest = X_scaled.iloc[-1:].values
                            prediction = model.predict(X_latest, horizon)[0]
                        
                        # Create prediction result
                        result = PredictionResult(
                            model=model_type,
                            horizon=horizon_str,
                            predicted_return=float(prediction),
                            direction="up" if prediction > 0 else "down",
                            probability=self._calculate_confidence(prediction)
                        )
                        predictions.append(result)
                        
                except Exception as e:
                    logger.error(f"Error predicting with {model_type.value}: {e}")
                    continue
            
            return predictions
            
        finally:
            db.close()
    
    def _calculate_confidence(self, prediction: float) -> float:
        """Calculate confidence score for prediction"""
        # Simple confidence based on magnitude of prediction
        # In practice, you might use model uncertainty estimates
        abs_pred = abs(prediction)
        confidence = min(0.5 + abs_pred * 10, 0.95)  # Scale to 0.5-0.95
        return round(confidence, 3)
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        if not self._initialized:
            await self.initialize_models()
        
        status = {
            "initialized": self._initialized,
            "models": {}
        }
        
        for model_type, model in self.models.items():
            model_status = {
                "is_trained": hasattr(model, 'is_trained') and model.is_trained,
                "prediction_horizons": getattr(model, 'prediction_horizons', []),
                "model_name": getattr(model, 'model_name', model_type.value)
            }
            
            if hasattr(model, 'training_history'):
                model_status["last_training"] = model.training_history
                
            status["models"][model_type.value] = model_status
        
        return status
    
    async def get_feature_importance(self, symbol: str, model_type: ModelType, 
                                   horizon: PredictionHorizon) -> Optional[Dict[str, float]]:
        """Get feature importance for a model"""
        if model_type not in self.models:
            return None
        
        model = self.models[model_type]
        horizon_int = int(horizon.value.replace('d', ''))
        
        importance = model.get_feature_importance(horizon_int)
        if importance is None:
            return None
        
        # Get feature names
        feature_names = getattr(self.preprocessor, 'feature_columns', [])
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        # Create feature importance dictionary
        feature_dict = {}
        for i, (name, imp) in enumerate(zip(feature_names, importance)):
            feature_dict[name] = float(imp)
        
        # Sort by importance
        return dict(sorted(feature_dict.items(), key=lambda x: x[1], reverse=True))

# Global model service instance
model_service = ModelService()