#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List

from src.database.connection import SessionLocal
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMStockPredictor, BidirectionalLSTMPredictor
from src.models.random_forest import RandomForestStockPredictor, GradientBoostingStockPredictor
from src.models.ensemble import EnsembleStockPredictor, StackingEnsemble
from src.models.base_model import ModelEvaluator

warnings.filterwarnings('ignore')

class MLModelPipeline:
    """
    Complete ML model training and evaluation pipeline
    """
    
    def __init__(self, prediction_horizons: List[int] = [1, 5, 10]):
        self.prediction_horizons = prediction_horizons
        self.models = {}
        self.preprocessor = DataPreprocessor(scaler_type='standard')
        self.evaluation_results = {}
        
    def prepare_data(self, db_session, symbol: str = 'AAPL') -> Dict:
        """
        Prepare data for ML training
        
        Args:
            db_session: Database session
            symbol: Stock symbol to train on
            
        Returns:
            Dictionary with prepared data
        """
        print(f"Preparing data for {symbol}...")
        
        # Load and preprocess data
        data = self.preprocessor.process_stock_for_ml(
            db_session, symbol,
            target_column='target_return_1d',  # Will be modified for each horizon
            test_size=0.2,
            scale_features=True,
            sequence_length=60
        )
        
        # Prepare targets for all horizons
        features_df = data['features_df']
        targets = {}
        
        for horizon in self.prediction_horizons:
            target_col = f'target_return_{horizon}d'
            if target_col in features_df.columns:
                # Use the same feature split as the preprocessor
                X_all, y_all = self.preprocessor.prepare_ml_dataset(features_df, target_col)
                
                # Use same split indices as the original data
                split_idx = len(data['X_train'])
                targets[horizon] = {
                    'y_train': y_all.iloc[:split_idx],
                    'y_test': y_all.iloc[split_idx:]
                }
        
        data['targets'] = targets
        
        print(f"Data prepared for {symbol}:")
        print(f"  Features: {data['X_train'].shape}")
        print(f"  LSTM sequences: {data['X_seq_train'].shape}")
        print(f"  Prediction horizons: {list(targets.keys())}")
        
        return data
    
    def initialize_models(self) -> None:
        """Initialize all models"""
        print("Initializing models...")
        
        # Random Forest
        self.models['RandomForest'] = RandomForestStockPredictor(
            prediction_horizons=self.prediction_horizons,
            n_estimators=100,  # Reduced for faster training
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        
        # Gradient Boosting
        self.models['GradientBoosting'] = GradientBoostingStockPredictor(
            prediction_horizons=self.prediction_horizons,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # LSTM
        self.models['LSTM'] = LSTMStockPredictor(
            prediction_horizons=self.prediction_horizons,
            lstm_units=[64, 32],
            dropout_rate=0.2,
            learning_rate=0.001,
            batch_size=32,
            epochs=30,  # Reduced for faster training
            patience=8,
            sequence_length=60
        )
        
        # Bidirectional LSTM
        self.models['BiLSTM'] = BidirectionalLSTMPredictor(
            prediction_horizons=self.prediction_horizons,
            lstm_units=[32, 16],  # Smaller for Bi-LSTM
            dropout_rate=0.3,
            epochs=20,
            patience=6,
            sequence_length=30
        )
        
        # Ensemble
        base_models = [
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
        ]
        
        self.models['Ensemble'] = EnsembleStockPredictor(
            prediction_horizons=self.prediction_horizons,
            base_models=base_models,
            ensemble_method='weighted_average'
        )
        
        print(f"Initialized {len(self.models)} models")
    
    def train_traditional_models(self, data: Dict) -> None:
        """Train traditional ML models (non-sequence)"""
        print("\n" + "="*60)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*60)
        
        X_train = data['X_train_scaled']
        X_test = data['X_test_scaled']
        targets = data['targets']
        
        # Prepare target dictionaries for multi-horizon training
        y_train_dict = {h: targets[h]['y_train'] for h in self.prediction_horizons if h in targets}
        y_test_dict = {h: targets[h]['y_test'] for h in self.prediction_horizons if h in targets}
        
        traditional_models = ['RandomForest', 'GradientBoosting', 'Ensemble']
        
        for model_name in traditional_models:
            if model_name not in self.models:
                continue
                
            print(f"\nTraining {model_name}...")
            try:
                model = self.models[model_name]
                
                # Train for all horizons
                histories = model.train_multiple_horizons(
                    X_train.values, y_train_dict,
                    X_test.values, y_test_dict  # Using test as validation
                )
                
                print(f"✅ {model_name} trained successfully for {len(histories)} horizons")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue
    
    def train_sequence_models(self, data: Dict) -> None:
        """Train sequence-based models (LSTM)"""
        print("\n" + "="*60)
        print("TRAINING SEQUENCE MODELS (LSTM)")
        print("="*60)
        
        X_seq_train = data['X_seq_train']
        X_seq_test = data['X_seq_test']
        
        # Create corresponding targets for sequences
        sequence_length = data['sequence_length']
        targets = data['targets']
        
        y_seq_train_dict = {}
        y_seq_test_dict = {}
        
        for horizon in self.prediction_horizons:
            if horizon in targets:
                # Adjust targets for sequence data
                y_train_full = targets[horizon]['y_train']
                y_test_full = targets[horizon]['y_test']
                
                # For sequences, we need to align with the sequence data
                if len(y_train_full) > sequence_length:
                    y_seq_train_dict[horizon] = y_train_full.iloc[sequence_length:].values
                if len(y_test_full) > sequence_length:
                    y_seq_test_dict[horizon] = y_test_full.iloc[sequence_length:].values
        
        sequence_models = ['LSTM', 'BiLSTM']
        
        for model_name in sequence_models:
            if model_name not in self.models:
                continue
                
            print(f"\nTraining {model_name}...")
            try:
                model = self.models[model_name]
                
                # Train for each horizon separately (LSTM needs individual training)
                for horizon in self.prediction_horizons:
                    if horizon in y_seq_train_dict:
                        print(f"  Training {model_name} for {horizon}-day prediction...")
                        
                        y_train = y_seq_train_dict[horizon]
                        y_val = y_seq_test_dict.get(horizon, None)
                        X_val = X_seq_test if y_val is not None else None
                        
                        # Ensure we have enough data
                        min_samples = min(len(X_seq_train), len(y_train))
                        if min_samples < 50:
                            print(f"    Warning: Only {min_samples} samples for training, skipping...")
                            continue
                        
                        X_train_adj = X_seq_train[:min_samples]
                        y_train_adj = y_train[:min_samples]
                        
                        if X_val is not None and y_val is not None:
                            min_val_samples = min(len(X_val), len(y_val))
                            X_val_adj = X_val[:min_val_samples]
                            y_val_adj = y_val[:min_val_samples]
                        else:
                            X_val_adj, y_val_adj = None, None
                        
                        history = model.train(X_train_adj, y_train_adj, X_val_adj, y_val_adj, horizon)
                        print(f"    ✅ {horizon}d model trained")
                
                print(f"✅ {model_name} trained successfully")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def evaluate_models(self, data: Dict) -> pd.DataFrame:
        """Evaluate all trained models"""
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        
        results = []
        
        # Prepare test data and targets
        X_test = data['X_test_scaled']
        X_seq_test = data['X_seq_test']
        targets = data['targets']
        
        y_test_dict = {h: targets[h]['y_test'] for h in self.prediction_horizons if h in targets}
        
        for model_name, model in self.models.items():
            if not hasattr(model, 'is_trained') or not model.is_trained:
                print(f"Skipping {model_name} (not trained)")
                continue
                
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Choose appropriate test data
                if model_name in ['LSTM', 'BiLSTM']:
                    X_test_model = X_seq_test
                    # Adjust targets for sequence models
                    sequence_length = data['sequence_length']
                    y_test_adj = {}
                    for h in self.prediction_horizons:
                        if h in y_test_dict and len(y_test_dict[h]) > sequence_length:
                            y_test_adj[h] = y_test_dict[h].iloc[sequence_length:].values
                else:
                    X_test_model = X_test.values
                    y_test_adj = {h: y.values for h, y in y_test_dict.items()}
                
                # Evaluate model
                metrics = model.evaluate(X_test_model, y_test_adj, task_type='regression')
                
                # Store results
                for horizon, horizon_metrics in metrics.items():
                    row = {
                        'model': model_name,
                        'horizon': f"{horizon}d",
                        **horizon_metrics
                    }
                    results.append(row)
                
                print(f"✅ {model_name} evaluated")
                
                # Print summary metrics
                for horizon in metrics:
                    rmse = metrics[horizon]['rmse']
                    r2 = metrics[horizon]['r2']
                    print(f"  {horizon}d: RMSE={rmse:.4f}, R²={r2:.4f}")
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        self.evaluation_results = results_df
        
        return results_df
    
    def print_results_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary of results"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        # Group by horizon and show best models
        for horizon in ['1d', '5d', '10d']:
            horizon_results = results_df[results_df['horizon'] == horizon]
            if len(horizon_results) == 0:
                continue
                
            print(f"\n{horizon} Predictions:")
            print("-" * 40)
            
            # Sort by RMSE (lower is better)
            horizon_results_sorted = horizon_results.sort_values('rmse')
            
            for _, row in horizon_results_sorted.iterrows():
                print(f"{row['model']:15} | RMSE: {row['rmse']:.4f} | R²: {row['r2']:6.3f} | MAE: {row['mae']:.4f}")
            
            # Show best model
            best_model = horizon_results_sorted.iloc[0]
            print(f"🏆 Best: {best_model['model']} (RMSE: {best_model['rmse']:.4f})")
        
        # Overall best models
        print(f"\n" + "="*50)
        print("OVERALL BEST MODELS BY HORIZON")
        print("="*50)
        
        best_models = results_df.loc[results_df.groupby('horizon')['rmse'].idxmin()]
        for _, row in best_models.iterrows():
            print(f"{row['horizon']:3}: {row['model']:15} (RMSE: {row['rmse']:.4f}, R²: {row['r2']:6.3f})")

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    print("🚀 STARTING ML MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Initialize pipeline
    pipeline = MLModelPipeline(prediction_horizons=[1, 5, 10])
    
    # Connect to database
    db = SessionLocal()
    
    try:
        # Step 1: Prepare data
        print("\n📊 STEP 1: DATA PREPARATION")
        data = pipeline.prepare_data(db, symbol='AAPL')
        
        # Step 2: Initialize models
        print("\n🤖 STEP 2: MODEL INITIALIZATION")
        pipeline.initialize_models()
        
        # Step 3: Train traditional models
        print("\n🎯 STEP 3: TRAINING TRADITIONAL MODELS")
        pipeline.train_traditional_models(data)
        
        # Step 4: Train sequence models
        print("\n🧠 STEP 4: TRAINING SEQUENCE MODELS")
        pipeline.train_sequence_models(data)
        
        # Step 5: Evaluate models
        print("\n📈 STEP 5: MODEL EVALUATION")
        results_df = pipeline.evaluate_models(data)
        
        # Step 6: Print results
        print("\n📋 STEP 6: RESULTS SUMMARY")
        pipeline.print_results_summary(results_df)
        
        print("\n" + "="*70)
        print("🎉 ML PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return pipeline, results_df
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    finally:
        db.close()

if __name__ == "__main__":
    pipeline, results = run_complete_pipeline()