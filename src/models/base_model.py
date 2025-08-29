from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

class BaseStockPredictor(ABC):
    """
    Abstract base class for stock prediction models
    """
    
    def __init__(self, model_name: str, prediction_horizons: List[int] = [1, 5, 10]):
        """
        Initialize base predictor
        
        Args:
            model_name: Name of the model
            prediction_horizons: List of prediction horizons in days
        """
        self.model_name = model_name
        self.prediction_horizons = prediction_horizons
        self.models = {}  # Dictionary to store models for different horizons
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False
        self.training_history = {}
        
    @abstractmethod
    def build_model(self, input_shape: Tuple, horizon: int) -> Any:
        """
        Build the model architecture
        
        Args:
            input_shape: Input shape for the model
            horizon: Prediction horizon
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              horizon: int = 1) -> Dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            horizon: Prediction horizon
            
        Returns:
            Training history
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            horizon: Prediction horizon
            
        Returns:
            Predictions
        """
        pass
    
    def train_multiple_horizons(self, X_train: np.ndarray, y_train_dict: Dict[int, np.ndarray],
                              X_val: Optional[np.ndarray] = None, 
                              y_val_dict: Optional[Dict[int, np.ndarray]] = None) -> Dict:
        """
        Train models for multiple prediction horizons
        
        Args:
            X_train: Training features
            y_train_dict: Dictionary of training targets for each horizon
            X_val: Validation features (optional)
            y_val_dict: Dictionary of validation targets for each horizon (optional)
            
        Returns:
            Dictionary of training histories for each horizon
        """
        histories = {}
        
        for horizon in self.prediction_horizons:
            if horizon not in y_train_dict:
                print(f"Warning: No training data for horizon {horizon}d")
                continue
                
            print(f"Training model for {horizon}-day prediction...")
            
            y_train = y_train_dict[horizon]
            y_val = y_val_dict[horizon] if y_val_dict else None
            
            history = self.train(X_train, y_train, X_val, y_val, horizon)
            histories[horizon] = history
            
        self.is_trained = True
        self.training_history = histories
        
        return histories
    
    def predict_multiple_horizons(self, X: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Make predictions for multiple horizons
        
        Args:
            X: Input features
            
        Returns:
            Dictionary of predictions for each horizon
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        for horizon in self.prediction_horizons:
            if horizon in self.models:
                predictions[horizon] = self.predict(X, horizon)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test_dict: Dict[int, np.ndarray],
                task_type: str = 'regression') -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test_dict: Dictionary of test targets for each horizon
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        results = {}
        
        for horizon in self.prediction_horizons:
            if horizon not in y_test_dict or horizon not in self.models:
                continue
                
            y_true = y_test_dict[horizon]
            y_pred = self.predict(X_test, horizon)
            
            if task_type == 'regression':
                metrics = self._calculate_regression_metrics(y_true, y_pred)
            else:
                # Convert to binary classification
                y_true_binary = (y_true > 0).astype(int)
                y_pred_binary = (y_pred > 0).astype(int)
                metrics = self._calculate_classification_metrics(y_true_binary, y_pred_binary)
                metrics.update(self._calculate_regression_metrics(y_true, y_pred))
            
            results[horizon] = metrics
            
        return results
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_actual': np.mean(y_true),
            'std_actual': np.std(y_true),
            'mean_predicted': np.mean(y_pred),
            'std_predicted': np.std(y_pred)
        }
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def get_feature_importance(self, horizon: int = 1) -> Optional[np.ndarray]:
        """
        Get feature importance (if supported by the model)
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Feature importance array or None
        """
        return None
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model_name': self.model_name,
            'prediction_horizons': self.prediction_horizons,
            'models': self.models,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_name = model_data['model_name']
        self.prediction_horizons = model_data['prediction_horizons']
        self.models = model_data['models']
        self.scalers = model_data.get('scalers', {})
        self.feature_columns = model_data.get('feature_columns', [])
        self.is_trained = model_data.get('is_trained', False)
        self.training_history = model_data.get('training_history', {})
        
        print(f"Model loaded from {filepath}")
    
    def summary(self) -> None:
        """Print model summary"""
        print(f"\n{'='*50}")
        print(f"Model: {self.model_name}")
        print(f"{'='*50}")
        print(f"Prediction Horizons: {self.prediction_horizons}")
        print(f"Trained: {self.is_trained}")
        print(f"Number of Features: {len(self.feature_columns)}")
        
        if self.is_trained:
            print(f"Available Models: {list(self.models.keys())}")
            
            if self.training_history:
                print(f"\nTraining History:")
                for horizon, history in self.training_history.items():
                    if isinstance(history, dict) and 'final_loss' in history:
                        print(f"  {horizon}d: Final Loss = {history['final_loss']:.4f}")
        
        print(f"{'='*50}")

class ModelEvaluator:
    """
    Utility class for evaluating and comparing models
    """
    
    @staticmethod
    def compare_models(models: Dict[str, BaseStockPredictor], 
                      X_test: np.ndarray, 
                      y_test_dict: Dict[int, np.ndarray],
                      task_type: str = 'regression') -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            models: Dictionary of models to compare
            X_test: Test features
            y_test_dict: Test targets for each horizon
            task_type: 'regression' or 'classification'
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_name, model in models.items():
            try:
                metrics = model.evaluate(X_test, y_test_dict, task_type)
                
                for horizon, horizon_metrics in metrics.items():
                    row = {
                        'model': model_name,
                        'horizon': f"{horizon}d",
                        **horizon_metrics
                    }
                    results.append(row)
                    
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        return pd.DataFrame(results)
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, predictions: Dict[str, np.ndarray], 
                        title: str = "Model Predictions Comparison") -> None:
        """
        Plot actual vs predicted values for different models
        
        Args:
            y_true: Actual values
            predictions: Dictionary of predictions from different models
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Plot actual values
            plt.plot(y_true, label='Actual', alpha=0.7, linewidth=2)
            
            # Plot predictions from each model
            for model_name, y_pred in predictions.items():
                plt.plot(y_pred, label=f'{model_name}', alpha=0.7)
            
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Stock Return')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    @staticmethod
    def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (correct prediction of up/down movement)
        
        Args:
            y_true: Actual returns
            y_pred: Predicted returns
            
        Returns:
            Directional accuracy as percentage
        """
        true_direction = (y_true > 0).astype(int)
        pred_direction = (y_pred > 0).astype(int)
        
        return accuracy_score(true_direction, pred_direction)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio for predicted returns
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)