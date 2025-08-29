import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import warnings

from src.models.base_model import BaseStockPredictor
from src.models.lstm_model import LSTMStockPredictor
from src.models.random_forest import RandomForestStockPredictor, GradientBoostingStockPredictor

warnings.filterwarnings('ignore')

class EnsembleStockPredictor(BaseStockPredictor):
    """
    Ensemble model combining multiple prediction models
    """
    
    def __init__(self,
                 prediction_horizons: List[int] = [1, 5, 10],
                 base_models: Optional[List[BaseStockPredictor]] = None,
                 ensemble_method: str = 'weighted_average',
                 meta_model: str = 'linear',
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize Ensemble predictor
        
        Args:
            prediction_horizons: List of prediction horizons in days
            base_models: List of base models to ensemble
            ensemble_method: Method for combining predictions ('weighted_average', 'meta_learning', 'voting')
            meta_model: Meta-learner for meta_learning method ('linear', 'ridge')
            weights: Manual weights for weighted_average method
        """
        super().__init__("Ensemble", prediction_horizons)
        
        # Initialize base models if not provided
        if base_models is None:
            self.base_models = self._initialize_default_models()
        else:
            self.base_models = base_models
        
        self.ensemble_method = ensemble_method
        self.meta_model_type = meta_model
        self.manual_weights = weights or {}
        
        # Storage for ensemble components
        self.meta_models = {}  # Meta-learning models for each horizon
        self.learned_weights = {}  # Learned weights for each horizon
        self.base_model_names = [model.model_name for model in self.base_models]
        
    def _initialize_default_models(self) -> List[BaseStockPredictor]:
        """Initialize default set of base models"""
        return [
            RandomForestStockPredictor(
                prediction_horizons=self.prediction_horizons,
                n_estimators=150,
                max_depth=20,
                random_state=42
            ),
            GradientBoostingStockPredictor(
                prediction_horizons=self.prediction_horizons,
                n_estimators=150,
                learning_rate=0.1,
                random_state=42
            ),
            LSTMStockPredictor(
                prediction_horizons=self.prediction_horizons,
                lstm_units=[64, 32],
                epochs=50,
                patience=10,
                sequence_length=30  # Shorter sequence for ensemble
            )
        ]
    
    def build_model(self, input_shape: Tuple, horizon: int) -> Any:
        """
        Build ensemble model (not used directly, kept for interface compatibility)
        """
        if self.ensemble_method == 'meta_learning':
            if self.meta_model_type == 'linear':
                return LinearRegression()
            elif self.meta_model_type == 'ridge':
                return Ridge(alpha=1.0)
        return None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              horizon: int = 1) -> Dict:
        """
        Train ensemble model
        
        Args:
            X_train: Training features
            y_train: Training targets  
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            horizon: Prediction horizon
            
        Returns:
            Training history
        """
        print(f"\nTraining Ensemble for {horizon}-day prediction")
        print(f"Base models: {self.base_model_names}")
        print(f"Ensemble method: {self.ensemble_method}")
        
        # Train base models
        base_predictions_train = {}
        base_predictions_val = {}
        training_histories = {}
        
        for i, model in enumerate(self.base_models):
            model_name = model.model_name
            print(f"\n--- Training base model {i+1}/{len(self.base_models)}: {model_name} ---")
            
            try:
                # For LSTM models, we need sequences
                if isinstance(model, LSTMStockPredictor):
                    # Convert flat features to sequences
                    if hasattr(model, 'create_sequences_from_features') and len(X_train.shape) == 2:
                        # Create sequences from feature matrix
                        X_train_seq, y_train_seq = self._create_sequences_for_lstm(X_train, y_train, model.sequence_length)
                        
                        if X_val is not None and y_val is not None:
                            X_val_seq, y_val_seq = self._create_sequences_for_lstm(X_val, y_val, model.sequence_length)
                        else:
                            X_val_seq, y_val_seq = None, None
                        
                        history = model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, horizon)
                        
                        # Get predictions
                        pred_train = model.predict(X_train_seq, horizon)
                        if X_val_seq is not None:
                            pred_val = model.predict(X_val_seq, horizon)
                        else:
                            pred_val = None
                    else:
                        # Already sequences
                        history = model.train(X_train, y_train, X_val, y_val, horizon)
                        pred_train = model.predict(X_train, horizon)
                        pred_val = model.predict(X_val, horizon) if X_val is not None else None
                
                else:
                    # Traditional ML models
                    history = model.train(X_train, y_train, X_val, y_val, horizon)
                    pred_train = model.predict(X_train, horizon)
                    pred_val = model.predict(X_val, horizon) if X_val is not None else None
                
                base_predictions_train[model_name] = pred_train
                if pred_val is not None:
                    base_predictions_val[model_name] = pred_val
                training_histories[model_name] = history
                
                print(f" {model_name} trained successfully")
                
            except Exception as e:
                print(f"L Error training {model_name}: {e}")
                continue
        
        if not base_predictions_train:
            raise ValueError("No base models trained successfully")
        
        # Train ensemble combination
        ensemble_history = self._train_ensemble_combination(
            base_predictions_train, y_train,
            base_predictions_val, y_val,
            horizon
        )
        
        # Store ensemble model
        self.models[horizon] = {
            'base_models': {model.model_name: model for model in self.base_models 
                          if model.model_name in base_predictions_train},
            'ensemble_method': self.ensemble_method,
            'weights': self.learned_weights.get(horizon, {}),
            'meta_model': self.meta_models.get(horizon, None)
        }
        
        # Combine training histories
        combined_history = {
            'base_models': training_histories,
            'ensemble': ensemble_history,
            'final_loss': ensemble_history.get('final_loss', 0),
            'method': self.ensemble_method
        }
        
        return combined_history
    
    def _create_sequences_for_lstm(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM from flat features"""
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _train_ensemble_combination(self, base_predictions_train: Dict[str, np.ndarray], y_train: np.ndarray,
                                  base_predictions_val: Dict[str, np.ndarray], y_val: Optional[np.ndarray],
                                  horizon: int) -> Dict:
        """Train the ensemble combination method"""
        
        if self.ensemble_method == 'weighted_average':
            return self._train_weighted_average(base_predictions_train, y_train, 
                                              base_predictions_val, y_val, horizon)
        elif self.ensemble_method == 'meta_learning':
            return self._train_meta_learning(base_predictions_train, y_train,
                                           base_predictions_val, y_val, horizon)
        else:
            # Simple average
            return self._train_simple_average(base_predictions_train, y_train,
                                            base_predictions_val, y_val, horizon)
    
    def _train_weighted_average(self, base_predictions_train: Dict[str, np.ndarray], y_train: np.ndarray,
                               base_predictions_val: Dict[str, np.ndarray], y_val: Optional[np.ndarray],
                               horizon: int) -> Dict:
        """Train weighted average ensemble"""
        
        if self.manual_weights:
            # Use manual weights
            weights = self.manual_weights
        else:
            # Learn weights based on validation performance
            weights = {}
            
            if base_predictions_val and y_val is not None:
                # Use validation set to determine weights
                for model_name, pred_val in base_predictions_val.items():
                    if len(pred_val) == len(y_val):
                        mse = mean_squared_error(y_val, pred_val)
                        weights[model_name] = 1.0 / (mse + 1e-8)  # Inverse MSE weighting
                    
            else:
                # Use training set (less reliable)
                for model_name, pred_train in base_predictions_train.items():
                    if len(pred_train) == len(y_train):
                        mse = mean_squared_error(y_train, pred_train)
                        weights[model_name] = 1.0 / (mse + 1e-8)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Equal weights fallback
            n_models = len(base_predictions_train)
            weights = {k: 1.0/n_models for k in base_predictions_train.keys()}
        
        self.learned_weights[horizon] = weights
        
        # Calculate ensemble performance
        ensemble_pred_train = self._combine_predictions(base_predictions_train, weights)
        train_mse = mean_squared_error(y_train, ensemble_pred_train)
        
        history = {
            'weights': weights,
            'train_mse': train_mse,
            'final_loss': train_mse
        }
        
        if base_predictions_val and y_val is not None:
            ensemble_pred_val = self._combine_predictions(base_predictions_val, weights)
            val_mse = mean_squared_error(y_val, ensemble_pred_val)
            history.update({
                'val_mse': val_mse,
                'final_val_loss': val_mse
            })
        
        print(f"Learned weights: {weights}")
        print(f"Ensemble train MSE: {train_mse:.6f}")
        
        return history
    
    def _train_meta_learning(self, base_predictions_train: Dict[str, np.ndarray], y_train: np.ndarray,
                           base_predictions_val: Dict[str, np.ndarray], y_val: Optional[np.ndarray],
                           horizon: int) -> Dict:
        """Train meta-learning ensemble"""
        
        # Stack base model predictions
        X_meta = np.column_stack(list(base_predictions_train.values()))
        
        # Build and train meta model
        meta_model = self.build_model(X_meta.shape, horizon)
        meta_model.fit(X_meta, y_train)
        self.meta_models[horizon] = meta_model
        
        # Calculate performance
        ensemble_pred_train = meta_model.predict(X_meta)
        train_mse = mean_squared_error(y_train, ensemble_pred_train)
        
        history = {
            'train_mse': train_mse,
            'final_loss': train_mse,
            'meta_model': str(meta_model)
        }
        
        if base_predictions_val and y_val is not None:
            X_meta_val = np.column_stack(list(base_predictions_val.values()))
            ensemble_pred_val = meta_model.predict(X_meta_val)
            val_mse = mean_squared_error(y_val, ensemble_pred_val)
            history.update({
                'val_mse': val_mse,
                'final_val_loss': val_mse
            })
        
        print(f"Meta-model: {meta_model}")
        print(f"Ensemble train MSE: {train_mse:.6f}")
        
        return history
    
    def _train_simple_average(self, base_predictions_train: Dict[str, np.ndarray], y_train: np.ndarray,
                            base_predictions_val: Dict[str, np.ndarray], y_val: Optional[np.ndarray],
                            horizon: int) -> Dict:
        """Train simple average ensemble"""
        
        weights = {k: 1.0/len(base_predictions_train) for k in base_predictions_train.keys()}
        self.learned_weights[horizon] = weights
        
        ensemble_pred_train = self._combine_predictions(base_predictions_train, weights)
        train_mse = mean_squared_error(y_train, ensemble_pred_train)
        
        history = {
            'weights': weights,
            'train_mse': train_mse,
            'final_loss': train_mse
        }
        
        if base_predictions_val and y_val is not None:
            ensemble_pred_val = self._combine_predictions(base_predictions_val, weights)
            val_mse = mean_squared_error(y_val, ensemble_pred_val)
            history.update({
                'val_mse': val_mse,
                'final_val_loss': val_mse
            })
        
        print(f"Simple average ensemble")
        print(f"Ensemble train MSE: {train_mse:.6f}")
        
        return history
    
    def _combine_predictions(self, predictions: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """Combine predictions using weights"""
        combined = None
        
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0)
            if combined is None:
                combined = weight * pred
            else:
                combined += weight * pred
        
        return combined
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Make ensemble predictions
        
        Args:
            X: Input features
            horizon: Prediction horizon
            
        Returns:
            Ensemble predictions
        """
        if horizon not in self.models:
            raise ValueError(f"No trained ensemble for horizon {horizon}")
        
        ensemble_info = self.models[horizon]
        base_models = ensemble_info['base_models']
        method = ensemble_info['ensemble_method']
        
        # Get predictions from base models
        base_predictions = {}
        
        for model_name, model in base_models.items():
            try:
                if isinstance(model, LSTMStockPredictor) and len(X.shape) == 2:
                    # Convert to sequences if needed
                    if X.shape[0] > model.sequence_length:
                        X_seq = self._create_sequences_for_lstm(X, np.zeros(len(X)), model.sequence_length)[0]
                        pred = model.predict(X_seq, horizon)
                    else:
                        # Not enough data for sequences
                        continue
                else:
                    pred = model.predict(X, horizon)
                
                base_predictions[model_name] = pred
                
            except Exception as e:
                print(f"Warning: Error getting predictions from {model_name}: {e}")
                continue
        
        if not base_predictions:
            raise ValueError("No base model predictions available")
        
        # Combine predictions
        if method == 'meta_learning':
            meta_model = ensemble_info['meta_model']
            X_meta = np.column_stack(list(base_predictions.values()))
            return meta_model.predict(X_meta)
        else:
            weights = ensemble_info['weights']
            return self._combine_predictions(base_predictions, weights)
    
    def get_feature_importance(self, horizon: int = 1) -> Optional[Dict[str, np.ndarray]]:
        """
        Get feature importance from base models
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Dictionary of feature importances from base models
        """
        if horizon not in self.models:
            return None
        
        base_models = self.models[horizon]['base_models']
        importances = {}
        
        for model_name, model in base_models.items():
            importance = model.get_feature_importance(horizon)
            if importance is not None:
                importances[model_name] = importance
        
        return importances if importances else None
    
    def summary(self) -> None:
        """Print detailed ensemble summary"""
        super().summary()
        
        print(f"Ensemble Configuration:")
        print(f"  Method: {self.ensemble_method}")
        print(f"  Meta Model: {self.meta_model_type}")
        print(f"  Base Models: {self.base_model_names}")
        
        if self.learned_weights:
            print(f"  Learned Weights:")
            for horizon, weights in self.learned_weights.items():
                print(f"    {horizon}d: {weights}")
        
        if self.is_trained:
            print(f"\nBase Model Details:")
            for model in self.base_models:
                if hasattr(model, 'is_trained') and model.is_trained:
                    print(f"  {model.model_name}: Trained")
                else:
                    print(f"  {model.model_name}: Not trained")

class StackingEnsemble(EnsembleStockPredictor):
    """
    Stacking ensemble using cross-validation for meta-learning
    """
    
    def __init__(self, cv_folds: int = 5, **kwargs):
        kwargs['ensemble_method'] = 'meta_learning'
        super().__init__(**kwargs)
        self.model_name = "Stacking_Ensemble"
        self.cv_folds = cv_folds
    
    def _train_meta_learning(self, base_predictions_train: Dict[str, np.ndarray], y_train: np.ndarray,
                           base_predictions_val: Dict[str, np.ndarray], y_val: Optional[np.ndarray],
                           horizon: int) -> Dict:
        """Enhanced meta-learning with cross-validation"""
        from sklearn.model_selection import cross_val_predict
        
        # Use cross-validation to get unbiased meta-features
        meta_features = []
        for model_name in base_predictions_train.keys():
            # Get the trained model
            model = None
            for base_model in self.base_models:
                if base_model.model_name == model_name:
                    model = base_model
                    break
            
            if model and horizon in model.models:
                # For now, use the direct predictions (could be enhanced with CV)
                meta_features.append(base_predictions_train[model_name])
        
        if meta_features:
            X_meta = np.column_stack(meta_features)
            return super()._train_meta_learning(
                {f"model_{i}": feat for i, feat in enumerate(meta_features)},
                y_train, {}, y_val, horizon
            )
        else:
            return super()._train_meta_learning(base_predictions_train, y_train,
                                              base_predictions_val, y_val, horizon)