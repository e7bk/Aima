import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

from src.database.models import Stock, StockPrice
from src.data.indicators import TechnicalIndicators, FeatureEngineering

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Data preprocessing and feature engineering for stock prediction models
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax')
        """
        self.feature_engineering = FeatureEngineering()
        self.scaler_type = scaler_type
        self.scalers = {}
        self.feature_columns = []
        
    def load_stock_data(self, db: Session, symbol: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load stock data from database
        
        Args:
            db: Database session
            symbol: Stock symbol
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with stock data
        """
        # Get stock
        stock = db.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            raise ValueError(f"Stock {symbol} not found in database")
        
        # Build query
        query = db.query(StockPrice).filter(StockPrice.stock_id == stock.id)
        
        if start_date:
            query = query.filter(StockPrice.date >= start_date)
        if end_date:
            query = query.filter(StockPrice.date <= end_date)
            
        # Execute query and convert to DataFrame
        prices = query.order_by(StockPrice.date).all()
        
        if not prices:
            raise ValueError(f"No price data found for {symbol}")
        
        # Convert to DataFrame
        data = []
        for price in prices:
            data.append({
                'date': price.date,
                'open': price.open_price,
                'high': price.high,
                'low': price.low,
                'close': price.close,
                'volume': price.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def load_multiple_stocks(self, db: Session, symbols: List[str],
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple stocks
        
        Args:
            db: Database session
            symbols: List of stock symbols
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Dictionary with stock data DataFrames
        """
        stock_data = {}
        
        for symbol in symbols:
            try:
                df = self.load_stock_data(db, symbol, start_date, end_date)
                stock_data[symbol] = df
                print(f"Loaded {len(df)} records for {symbol}")
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        
        return stock_data
    
    def create_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Create comprehensive feature set for a stock
        
        Args:
            df: Raw stock data DataFrame
            symbol: Stock symbol (optional, for feature naming)
            
        Returns:
            DataFrame with engineered features
        """
        if len(df) < 50:  # Need minimum data for indicators
            raise ValueError(f"Insufficient data: {len(df)} rows. Need at least 50.")
        
        # Create all features
        features_df = self.feature_engineering.create_all_features(df)
        
        # Add symbol column if provided
        if symbol:
            features_df['symbol'] = symbol
        
        # Add target variables (future returns)
        for days_ahead in [1, 5, 10]:
            features_df[f'target_return_{days_ahead}d'] = (
                df['close'].shift(-days_ahead) / df['close'] - 1
            )
            features_df[f'target_direction_{days_ahead}d'] = (
                features_df[f'target_return_{days_ahead}d'] > 0
            ).astype(int)
        
        return features_df
    
    def prepare_ml_dataset(self, features_df: pd.DataFrame, 
                          target_column: str = 'target_return_1d',
                          feature_columns: Optional[List[str]] = None,
                          drop_na: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare dataset for machine learning
        
        Args:
            features_df: DataFrame with features
            target_column: Target column name
            feature_columns: List of feature columns (if None, auto-select)
            drop_na: Whether to drop rows with NaN values
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Auto-select feature columns if not provided
        if feature_columns is None:
            # Exclude target columns, date columns, and symbol
            exclude_patterns = [
                'target_', 'symbol', 'date'
            ]
            
            feature_columns = [
                col for col in features_df.columns 
                if not any(pattern in col for pattern in exclude_patterns)
            ]
        
        # Select features and target
        X = features_df[feature_columns]
        y = features_df[target_column]
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        # Handle missing values
        if drop_na:
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
        else:
            # Forward fill then backward fill
            X = X.fillna(method='ffill').fillna(method='bfill')
            y = y.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Dataset prepared: {len(X)} samples, {len(feature_columns)} features")
        print(f"Target: {target_column}")
        print(f"Feature columns: {len(feature_columns)} features")
        
        return X, y
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      fit_scaler: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale features using specified scaler
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Tuple of scaled features
        """
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        if fit_scaler:
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            self.scalers['features'] = scaler
        else:
            if 'features' not in self.scalers:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            X_train_scaled = pd.DataFrame(
                self.scalers['features'].transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scalers['features'].transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, X_test_scaled
    
    def create_sequences(self, X: pd.DataFrame, y: pd.Series, 
                        sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM models
        
        Args:
            X: Features DataFrame
            y: Target Series
            sequence_length: Length of sequences
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42,
                   time_based: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test data
            random_state: Random state for reproducibility
            time_based: Whether to use time-based split (recommended for time series)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if time_based:
            # Time-based split - use last portion as test set
            split_idx = int(len(X) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        print(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def process_stock_for_ml(self, db: Session, symbol: str,
                           target_column: str = 'target_return_1d',
                           test_size: float = 0.2,
                           scale_features: bool = True,
                           sequence_length: Optional[int] = None) -> Dict:
        """
        Complete preprocessing pipeline for a single stock
        
        Args:
            db: Database session
            symbol: Stock symbol
            target_column: Target variable
            test_size: Test set size
            scale_features: Whether to scale features
            sequence_length: Length for LSTM sequences (if None, no sequences)
            
        Returns:
            Dictionary with processed data
        """
        print(f"Processing {symbol} for ML...")
        
        # Load and create features
        raw_data = self.load_stock_data(db, symbol)
        features_df = self.create_features(raw_data, symbol)
        
        # Prepare ML dataset
        X, y = self.prepare_ml_dataset(features_df, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        # Scale features if requested
        if scale_features:
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        result = {
            'raw_data': raw_data,
            'features_df': features_df,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_columns': self.feature_columns,
            'target_column': target_column,
            'symbol': symbol
        }
        
        # Create sequences for LSTM if requested
        if sequence_length:
            X_seq_train, y_seq_train = self.create_sequences(
                X_train_scaled, y_train, sequence_length
            )
            X_seq_test, y_seq_test = self.create_sequences(
                X_test_scaled, y_test, sequence_length
            )
            
            result.update({
                'X_seq_train': X_seq_train,
                'X_seq_test': X_seq_test,
                'y_seq_train': y_seq_train,
                'y_seq_test': y_seq_test,
                'sequence_length': sequence_length
            })
        
        print(f" {symbol} processed successfully")
        return result
    
    def get_feature_importance_names(self) -> List[str]:
        """
        Get list of feature names for importance analysis
        
        Returns:
            List of feature names
        """
        return self.feature_columns.copy()
    
    def describe_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get descriptive statistics for features
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Description DataFrame
        """
        # Select only numeric columns
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        description = features_df[numeric_cols].describe()
        
        # Add additional statistics
        description.loc['missing'] = features_df[numeric_cols].isnull().sum()
        description.loc['missing_pct'] = (features_df[numeric_cols].isnull().sum() / len(features_df)) * 100
        
        return description