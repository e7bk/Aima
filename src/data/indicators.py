import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class TechnicalIndicators:
    """
    Technical indicators for stock market analysis
    """
    
    @staticmethod
    def simple_moving_average(data: pd.Series, window: int) -> pd.Series:
        """
        Simple Moving Average (SMA)
        
        Args:
            data: Price series (typically close prices)
            window: Period for moving average
            
        Returns:
            SMA series
        """
        return data.rolling(window=window).mean()
    
    @staticmethod
    def exponential_moving_average(data: pd.Series, window: int) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        
        Args:
            data: Price series (typically close prices)
            window: Period for EMA
            
        Returns:
            EMA series
        """
        return data.ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def relative_strength_index(data: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        
        Args:
            data: Price series (typically close prices)
            window: Period for RSI calculation (default: 14)
            
        Returns:
            RSI series
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD)
        
        Args:
            data: Price series (typically close prices)
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = TechnicalIndicators.exponential_moving_average(data, fast)
        ema_slow = TechnicalIndicators.exponential_moving_average(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """
        Bollinger Bands
        
        Args:
            data: Price series (typically close prices)
            window: Period for moving average (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            Dictionary with upper band, lower band, and middle band
        """
        sma = TechnicalIndicators.simple_moving_average(data, window)
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_window: %K period (default: 14)
            d_window: %D period (default: 3)
            
        Returns:
            Dictionary with %K and %D lines
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Williams %R
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for calculation (default: 14)
            
        Returns:
            Williams %R series
        """
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for ATR calculation (default: 14)
            
        Returns:
            ATR series
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            window: Period for CCI calculation (default: 20)
            
        Returns:
            CCI series
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci
    
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume (OBV)
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        price_change = close.diff()
        obv = volume.copy()
        obv.loc[price_change < 0] = -volume.loc[price_change < 0]
        obv.loc[price_change == 0] = 0
        
        return obv.cumsum()
    
    @staticmethod
    def price_rate_of_change(data: pd.Series, window: int = 12) -> pd.Series:
        """
        Price Rate of Change (ROC)
        
        Args:
            data: Price series
            window: Period for ROC calculation (default: 12)
            
        Returns:
            ROC series as percentage
        """
        roc = ((data - data.shift(window)) / data.shift(window)) * 100
        return roc

class FeatureEngineering:
    """
    Feature engineering for machine learning models
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic price-based features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        features = df.copy()
        
        # Price changes
        features['price_change'] = df['close'].diff()
        features['price_change_pct'] = df['close'].pct_change()
        
        # Price ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Volatility measures
        features['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift()),
                np.abs(df['low'] - df['close'].shift())
            )
        )
        
        # Volume features
        if 'volume' in df.columns:
            features['volume_change'] = df['volume'].pct_change()
            features['price_volume'] = df['close'] * df['volume']
            features['volume_sma'] = self.indicators.simple_moving_average(df['volume'], 20)
            features['volume_ratio'] = df['volume'] / features['volume_sma']
        
        return features
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        features = df.copy()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = self.indicators.simple_moving_average(df['close'], period)
            features[f'ema_{period}'] = self.indicators.exponential_moving_average(df['close'], period)
            
            # Price vs moving average ratios
            features[f'close_sma_{period}_ratio'] = df['close'] / features[f'sma_{period}']
            features[f'close_ema_{period}_ratio'] = df['close'] / features[f'ema_{period}']
        
        # RSI
        features['rsi'] = self.indicators.relative_strength_index(df['close'])
        
        # MACD
        macd_data = self.indicators.macd(df['close'])
        features['macd'] = macd_data['macd']
        features['macd_signal'] = macd_data['signal']
        features['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = self.indicators.bollinger_bands(df['close'])
        features['bb_upper'] = bb_data['upper']
        features['bb_middle'] = bb_data['middle']
        features['bb_lower'] = bb_data['lower']
        features['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        features['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Stochastic
        if all(col in df.columns for col in ['high', 'low']):
            stoch_data = self.indicators.stochastic_oscillator(df['high'], df['low'], df['close'])
            features['stoch_k'] = stoch_data['k']
            features['stoch_d'] = stoch_data['d']
            
            # Williams %R
            features['williams_r'] = self.indicators.williams_r(df['high'], df['low'], df['close'])
            
            # ATR
            features['atr'] = self.indicators.average_true_range(df['high'], df['low'], df['close'])
            
            # CCI
            features['cci'] = self.indicators.commodity_channel_index(df['high'], df['low'], df['close'])
        
        # Volume indicators
        if 'volume' in df.columns:
            features['obv'] = self.indicators.on_balance_volume(df['close'], df['volume'])
        
        # Rate of Change
        for period in [1, 5, 10, 20]:
            features[f'roc_{period}'] = self.indicators.price_rate_of_change(df['close'], period)
        
        return features
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            df: DataFrame with data
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        features = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return features
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Create rolling statistical features
        
        Args:
            df: DataFrame with data
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        features = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    features[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                    features[f'{col}_std_{window}'] = df[col].rolling(window).std()
                    features[f'{col}_min_{window}'] = df[col].rolling(window).min()
                    features[f'{col}_max_{window}'] = df[col].rolling(window).max()
                    features[f'{col}_median_{window}'] = df[col].rolling(window).median()
        
        return features
    
    def create_all_features(self, df: pd.DataFrame, 
                          lag_columns: Optional[List[str]] = None,
                          lags: Optional[List[int]] = None,
                          rolling_columns: Optional[List[str]] = None,
                          rolling_windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set
        
        Args:
            df: DataFrame with OHLCV data
            lag_columns: Columns for lag features
            lags: Lag periods
            rolling_columns: Columns for rolling features
            rolling_windows: Rolling window sizes
            
        Returns:
            DataFrame with all features
        """
        # Default parameters
        if lag_columns is None:
            lag_columns = ['close', 'volume', 'rsi', 'macd']
        if lags is None:
            lags = [1, 2, 3, 5]
        if rolling_columns is None:
            rolling_columns = ['close', 'volume']
        if rolling_windows is None:
            rolling_windows = [5, 10, 20]
        
        # Create features step by step
        features = self.create_price_features(df)
        features = self.create_technical_features(features)
        features = self.create_lag_features(features, lag_columns, lags)
        features = self.create_rolling_features(features, rolling_columns, rolling_windows)
        
        return features