import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import warnings

from src.models.base_model import BaseStockPredictor

warnings.filterwarnings('ignore')

class RandomForestStockPredictor(BaseStockPredictor):
    """
    Random Forest model for stock price prediction
    """
    
    def __init__(self,
                 prediction_horizons: List[int] = [1, 5, 10],
                 n_estimators: int = 200,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 5,
                 min_samples_leaf: int = 2,
                 max_features: str = 'sqrt',
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Initialize Random Forest predictor
        """
        super().__init__("RandomForest", prediction_horizons)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
    
    def build_model(self, input_shape: Tuple, horizon: int) -> RandomForestRegressor:
        """Build Random Forest model"""
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              horizon: int = 1) -> Dict:
        """Train Random Forest model"""
        print(f"\nTraining Random Forest for {horizon}-day prediction")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        
        # Build and train model
        model = self.build_model(X_train.shape, horizon)
        model.fit(X_train, y_train)
        self.models[horizon] = model
        
        # Calculate metrics
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_score = model.score(X_train, y_train)
        
        training_history = {
            'train_mse': train_mse,
            'train_r2': train_score,
            'final_loss': train_mse,
            'feature_importances': model.feature_importances_
        }
        
        print(f"Training MSE: {train_mse:.6f}")
        print(f"Training R2: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_score = model.score(X_val, y_val)
            
            training_history.update({
                'val_mse': val_mse,
                'val_r2': val_score,
                'final_val_loss': val_mse
            })
            
            print(f"Validation MSE: {val_mse:.6f}")
            print(f"Validation R2: {val_score:.4f}")
        
        return training_history
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Make predictions"""
        if horizon not in self.models:
            raise ValueError(f"No trained model for horizon {horizon}")
        
        model = self.models[horizon]
        return model.predict(X)
    
    def get_feature_importance(self, horizon: int = 1) -> Optional[np.ndarray]:
        """Get feature importance from Random Forest"""
        if horizon not in self.models:
            return None
        
        model = self.models[horizon]
        return model.feature_importances_

class GradientBoostingStockPredictor(BaseStockPredictor):
    """
    Gradient Boosting model for stock price prediction
    """
    
    def __init__(self,
                 prediction_horizons: List[int] = [1, 5, 10],
                 n_estimators: int = 200,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 random_state: int = 42):
        """Initialize Gradient Boosting predictor"""
        super().__init__("GradientBoosting", prediction_horizons)
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
    
    def build_model(self, input_shape: Tuple, horizon: int) -> GradientBoostingRegressor:
        """Build Gradient Boosting model"""
        model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              horizon: int = 1) -> Dict:
        """Train Gradient Boosting model"""
        print(f"\nTraining Gradient Boosting for {horizon}-day prediction")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        
        # Build and train model
        model = self.build_model(X_train.shape, horizon)
        model.fit(X_train, y_train)
        self.models[horizon] = model
        
        # Calculate metrics
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_score = model.score(X_train, y_train)
        
        training_history = {
            'train_mse': train_mse,
            'train_r2': train_score,
            'final_loss': train_mse,
            'feature_importances': model.feature_importances_
        }
        
        print(f"Training MSE: {train_mse:.6f}")
        print(f"Training R2: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_score = model.score(X_val, y_val)
            
            training_history.update({
                'val_mse': val_mse,
                'val_r2': val_score,
                'final_val_loss': val_mse
            })
            
            print(f"Validation MSE: {val_mse:.6f}")
            print(f"Validation R2: {val_score:.4f}")
        
        return training_history
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Make predictions"""
        if horizon not in self.models:
            raise ValueError(f"No trained model for horizon {horizon}")
        
        model = self.models[horizon]
        return model.predict(X)
    
    def get_feature_importance(self, horizon: int = 1) -> Optional[np.ndarray]:
        """Get feature importance from Gradient Boosting"""
        if horizon not in self.models:
            return None
        
        model = self.models[horizon]
        return model.feature_importances_