import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

from src.models.base_model import BaseStockPredictor

warnings.filterwarnings('ignore')

class LSTMStockPredictor(BaseStockPredictor):
    """
    LSTM Neural Network for stock price prediction
    """
    
    def __init__(self, 
                 prediction_horizons: List[int] = [1, 5, 10],
                 lstm_units: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: int = 10,
                 sequence_length: int = 60):
        """
        Initialize LSTM predictor
        
        Args:
            prediction_horizons: List of prediction horizons in days
            lstm_units: List of LSTM layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Early stopping patience
            sequence_length: Length of input sequences
        """
        super().__init__("LSTM", prediction_horizons)
        
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.sequence_length = sequence_length
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def build_model(self, input_shape: Tuple, horizon: int) -> keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: Input shape (sequence_length, n_features)
            horizon: Prediction horizon
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential(name=f"LSTM_{horizon}d")
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name=f"lstm_{i+1}"
            ))
            
            # Add batch normalization
            if i < len(self.lstm_units) - 1:
                model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout_rate))
        
        model.add(layers.Dense(32, activation='relu', name='dense_2'))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              horizon: int = 1) -> Dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences (samples, sequence_length, features)
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            horizon: Prediction horizon
            
        Returns:
            Training history
        """
        # Ensure correct input shape
        if len(X_train.shape) != 3:
            raise ValueError(f"X_train must have 3 dimensions (samples, sequence_length, features), got {X_train.shape}")
        
        input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, n_features)
        
        # Build model
        model = self.build_model(input_shape, horizon)
        self.models[horizon] = model
        
        print(f"\nTraining LSTM for {horizon}-day prediction")
        print(f"Input shape: {input_shape}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Model parameters: {model.count_params():,}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # Store training history
        training_history = {
            'loss': history.history['loss'],
            'mae': history.history['mae'],
            'final_loss': history.history['loss'][-1],
            'final_mae': history.history['mae'][-1],
            'epochs_trained': len(history.history['loss'])
        }
        
        if validation_data is not None:
            training_history.update({
                'val_loss': history.history['val_loss'],
                'val_mae': history.history['val_mae'],
                'final_val_loss': history.history['val_loss'][-1],
                'final_val_mae': history.history['val_mae'][-1]
            })
        
        print(f"Training completed in {training_history['epochs_trained']} epochs")
        print(f"Final training loss: {training_history['final_loss']:.6f}")
        if 'final_val_loss' in training_history:
            print(f"Final validation loss: {training_history['final_val_loss']:.6f}")
        
        return training_history
    
    def predict(self, X: np.ndarray, horizon: int = 1) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input sequences (samples, sequence_length, features)
            horizon: Prediction horizon
            
        Returns:
            Predictions array
        """
        if horizon not in self.models:
            raise ValueError(f"No trained model for horizon {horizon}")
        
        model = self.models[horizon]
        predictions = model.predict(X, batch_size=self.batch_size, verbose=0)
        
        return predictions.flatten()
    
    def create_sequences_from_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from feature matrix
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X.iloc[i-self.sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def get_feature_importance(self, horizon: int = 1) -> Optional[np.ndarray]:
        """
        Get feature importance using gradient-based method
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Feature importance array (averaged across sequence length)
        """
        if horizon not in self.models:
            return None
        
        model = self.models[horizon]
        
        # This is a simplified approach - in practice, you might want to use
        # more sophisticated methods like SHAP or LIME for LSTM interpretability
        try:
            # Get weights from the first LSTM layer
            lstm_weights = model.get_layer('lstm_1').get_weights()
            if len(lstm_weights) > 0:
                # Input weights shape: (input_dim, units * 4) for LSTM
                input_weights = lstm_weights[0]
                # Average absolute weights across units and gates
                feature_importance = np.mean(np.abs(input_weights), axis=1)
                return feature_importance
        except:
            pass
        
        return None
    
    def plot_training_history(self, horizon: int = 1) -> None:
        """
        Plot training history
        
        Args:
            horizon: Prediction horizon
        """
        if horizon not in self.training_history:
            print(f"No training history for horizon {horizon}")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            history = self.training_history[horizon]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss plot
            ax1.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_title(f'Model Loss - {horizon}d Prediction')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # MAE plot
            ax2.plot(history['mae'], label='Training MAE')
            if 'val_mae' in history:
                ax2.plot(history['val_mae'], label='Validation MAE')
            ax2.set_title(f'Model MAE - {horizon}d Prediction')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")
    
    def summary(self) -> None:
        """Print detailed model summary"""
        super().summary()
        
        print(f"LSTM Configuration:")
        print(f"  Units: {self.lstm_units}")
        print(f"  Dropout Rate: {self.dropout_rate}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Sequence Length: {self.sequence_length}")
        
        if self.is_trained:
            for horizon in self.models:
                model = self.models[horizon]
                print(f"\nModel Architecture ({horizon}d):")
                model.summary()

class BidirectionalLSTMPredictor(LSTMStockPredictor):
    """
    Bidirectional LSTM variant for improved pattern recognition
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "Bidirectional_LSTM"
    
    def build_model(self, input_shape: Tuple, horizon: int) -> keras.Model:
        """
        Build Bidirectional LSTM model
        """
        model = keras.Sequential(name=f"BiLSTM_{horizon}d")
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            model.add(layers.Bidirectional(
                layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.dropout_rate
                ),
                name=f"bidirectional_lstm_{i+1}"
            ))
            
            if i < len(self.lstm_units) - 1:
                model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout_rate))
        
        model.add(layers.Dense(64, activation='relu', name='dense_2'))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model