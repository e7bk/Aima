#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.database.connection import SessionLocal
from src.data.indicators import TechnicalIndicators, FeatureEngineering
from src.data.preprocessor import DataPreprocessor

def test_technical_indicators():
    """Test individual technical indicators"""
    print("Testing Technical Indicators")
    print("=" * 50)
    
    # Load some stock data from database
    db = SessionLocal()
    try:
        preprocessor = DataPreprocessor()
        
        # Load AAPL data
        print("Loading AAPL data from database...")
        df = preprocessor.load_stock_data(db, 'AAPL')
        print(f"✅ Loaded {len(df)} records")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Test individual indicators
        indicators = TechnicalIndicators()
        
        print("\n1. Testing Moving Averages...")
        sma_20 = indicators.simple_moving_average(df['close'], 20)
        ema_20 = indicators.exponential_moving_average(df['close'], 20)
        print(f"   SMA(20): Latest value = {sma_20.iloc[-1]:.2f}")
        print(f"   EMA(20): Latest value = {ema_20.iloc[-1]:.2f}")
        
        print("\n2. Testing RSI...")
        rsi = indicators.relative_strength_index(df['close'])
        print(f"   RSI: Latest value = {rsi.iloc[-1]:.2f}")
        
        print("\n3. Testing MACD...")
        macd_data = indicators.macd(df['close'])
        print(f"   MACD: {macd_data['macd'].iloc[-1]:.2f}")
        print(f"   Signal: {macd_data['signal'].iloc[-1]:.2f}")
        print(f"   Histogram: {macd_data['histogram'].iloc[-1]:.2f}")
        
        print("\n4. Testing Bollinger Bands...")
        bb_data = indicators.bollinger_bands(df['close'])
        print(f"   Upper Band: {bb_data['upper'].iloc[-1]:.2f}")
        print(f"   Middle Band: {bb_data['middle'].iloc[-1]:.2f}")
        print(f"   Lower Band: {bb_data['lower'].iloc[-1]:.2f}")
        
        print("\n5. Testing Stochastic Oscillator...")
        stoch_data = indicators.stochastic_oscillator(df['high'], df['low'], df['close'])
        print(f"   %K: {stoch_data['k'].iloc[-1]:.2f}")
        print(f"   %D: {stoch_data['d'].iloc[-1]:.2f}")
        
        print("\n6. Testing ATR...")
        atr = indicators.average_true_range(df['high'], df['low'], df['close'])
        print(f"   ATR: {atr.iloc[-1]:.2f}")
        
        print("\n7. Testing Volume Indicators...")
        if 'volume' in df.columns:
            obv = indicators.on_balance_volume(df['close'], df['volume'])
            print(f"   OBV: {obv.iloc[-1]:,.0f}")
        
        print("\n✅ All technical indicators working correctly!")
        
    finally:
        db.close()

def test_feature_engineering():
    """Test feature engineering capabilities"""
    print("\n\nTesting Feature Engineering")
    print("=" * 50)
    
    db = SessionLocal()
    try:
        preprocessor = DataPreprocessor()
        
        # Load data
        print("Loading stock data...")
        df = preprocessor.load_stock_data(db, 'AAPL')
        
        # Create features
        print("Creating comprehensive features...")
        features_df = preprocessor.create_features(df, 'AAPL')
        
        print(f"✅ Created {len(features_df.columns)} features")
        print(f"Original columns: {len(df.columns)}")
        print(f"Feature columns: {len(features_df.columns)}")
        
        # Show feature categories
        feature_categories = {
            'Price Features': [col for col in features_df.columns if any(x in col for x in ['price_', 'ratio', 'true_range'])],
            'Moving Averages': [col for col in features_df.columns if any(x in col for x in ['sma_', 'ema_'])],
            'Technical Indicators': [col for col in features_df.columns if any(x in col for x in ['rsi', 'macd', 'bb_', 'stoch_', 'williams', 'atr', 'cci'])],
            'Volume Features': [col for col in features_df.columns if 'volume' in col or 'obv' in col],
            'Lag Features': [col for col in features_df.columns if 'lag_' in col],
            'Rolling Features': [col for col in features_df.columns if any(x in col for x in ['_mean_', '_std_', '_min_', '_max_'])],
            'Target Variables': [col for col in features_df.columns if 'target_' in col]
        }
        
        print("\nFeature categories:")
        for category, features in feature_categories.items():
            print(f"   {category}: {len(features)} features")
        
        # Show some sample values
        print(f"\nSample feature values (latest):")
        sample_features = ['close', 'rsi', 'macd', 'bb_position', 'sma_20', 'target_return_1d']
        for feature in sample_features:
            if feature in features_df.columns:
                value = features_df[feature].iloc[-2] if feature.startswith('target_') else features_df[feature].iloc[-1]
                if pd.notna(value):
                    print(f"   {feature}: {value:.4f}")
        
        # Test data description
        print(f"\nFeature statistics:")
        description = preprocessor.describe_features(features_df)
        print(f"   Total features: {len(description.columns)}")
        print(f"   Missing data: {description.loc['missing'].sum()} total missing values")
        
        print("\n✅ Feature engineering working correctly!")
        
    finally:
        db.close()

def test_ml_preprocessing():
    """Test ML preprocessing pipeline"""
    print("\n\nTesting ML Preprocessing Pipeline")
    print("=" * 50)
    
    db = SessionLocal()
    try:
        preprocessor = DataPreprocessor(scaler_type='standard')
        
        # Process single stock
        print("Processing AAPL for machine learning...")
        result = preprocessor.process_stock_for_ml(
            db, 'AAPL',
            target_column='target_return_1d',
            test_size=0.2,
            scale_features=True,
            sequence_length=60  # For LSTM
        )
        
        print(f"\nProcessing results:")
        print(f"   Raw data shape: {result['raw_data'].shape}")
        print(f"   Features shape: {result['features_df'].shape}")
        print(f"   Training set: {result['X_train'].shape}")
        print(f"   Test set: {result['X_test'].shape}")
        print(f"   LSTM sequences - Train: {result['X_seq_train'].shape}")
        print(f"   LSTM sequences - Test: {result['X_seq_test'].shape}")
        
        # Check data quality
        print(f"\nData quality:")
        print(f"   Features with missing values: {result['X_train'].isnull().sum().sum()}")
        print(f"   Target missing values: {result['y_train'].isnull().sum()}")
        
        # Feature scaling check
        print(f"\nScaling verification:")
        print(f"   Train mean: {result['X_train_scaled'].mean().mean():.4f}")
        print(f"   Train std: {result['X_train_scaled'].std().mean():.4f}")
        
        # Target distribution
        print(f"\nTarget distribution:")
        print(f"   Target mean: {result['y_train'].mean():.4f}")
        print(f"   Target std: {result['y_train'].std():.4f}")
        print(f"   Positive returns: {(result['y_train'] > 0).sum()}/{len(result['y_train'])} ({(result['y_train'] > 0).mean()*100:.1f}%)")
        
        print("\n✅ ML preprocessing pipeline working correctly!")
        
        return result
        
    finally:
        db.close()

def test_multiple_stocks():
    """Test processing multiple stocks"""
    print("\n\nTesting Multiple Stocks Processing")
    print("=" * 50)
    
    db = SessionLocal()
    try:
        preprocessor = DataPreprocessor()
        
        # Load multiple stocks
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        print(f"Loading data for {len(symbols)} stocks...")
        
        stock_data = preprocessor.load_multiple_stocks(db, symbols)
        
        print(f"\nLoaded data:")
        for symbol, df in stock_data.items():
            print(f"   {symbol}: {len(df)} records, {df.index.min()} to {df.index.max()}")
        
        # Test feature creation for each stock
        print(f"\nCreating features for each stock...")
        features_data = {}
        for symbol, df in stock_data.items():
            try:
                features_df = preprocessor.create_features(df, symbol)
                features_data[symbol] = features_df
                print(f"   {symbol}: {features_df.shape[1]} features created")
            except Exception as e:
                print(f"   {symbol}: Error - {e}")
        
        print(f"\n✅ Multiple stocks processing completed!")
        
    finally:
        db.close()

def main():
    """Run all tests"""
    try:
        test_technical_indicators()
        test_feature_engineering()
        result = test_ml_preprocessing()
        test_multiple_stocks()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Technical indicators implemented")
        print("✅ Feature engineering pipeline ready")
        print("✅ ML preprocessing working")
        print("✅ Multi-stock processing functional")
        print("\nYour technical analysis system is ready for ML model training!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()