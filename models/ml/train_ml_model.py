"""
Machine Learning Training Pipeline for Stock Price Prediction.

This module provides a complete training pipeline using Random Forest Regressor
with time-series data splitting, feature engineering, and model evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import logging
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from feature_engineering.financial_features import FinancialFeatureEngineer
    from data_ingestion.historical_data import HistoricalDataDownloader
except ImportError:
    # Fallback if running from different location
    FinancialFeatureEngineer = None
    HistoricalDataDownloader = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLTrainingPipeline:
    """
    Complete ML training pipeline for stock price prediction.
    
    Features:
    - Time-series data splitting (chronological)
    - Feature engineering
    - Random Forest training
    - Model evaluation with multiple metrics
    - Model persistence
    """
    
    def __init__(
        self,
        model_dir: str = "models/ml",
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the training pipeline.
        
        Args:
            model_dir: Directory to save models. Default: "models/ml"
            test_size: Proportion of data for testing. Default: 0.2
            validation_size: Proportion of training data for validation. Default: 0.1
            random_state: Random state for reproducibility. Default: 42
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_engineer = None
        self.feature_names = None
        self.training_metadata = {}
        
        logger.info(f"MLTrainingPipeline initialized. Model directory: {self.model_dir}")
    
    def load_data(
        self,
        data: Optional[pd.DataFrame] = None,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "2y"
    ) -> pd.DataFrame:
        """
        Load data from DataFrame or download from Yahoo Finance.
        
        Args:
            data: Pre-loaded DataFrame with OHLCV data
            ticker: Stock ticker symbol (if data not provided)
            start_date: Start date for download
            end_date: End date for download
            period: Period for download if dates not provided
        
        Returns:
            DataFrame with OHLCV data
        """
        if data is not None:
            logger.info(f"Using provided data with {len(data)} records")
            return data.copy()
        
        if ticker is None:
            raise ValueError("Either 'data' or 'ticker' must be provided")
        
        if HistoricalDataDownloader is None:
            raise ImportError("HistoricalDataDownloader not available. Install required dependencies.")
        
        logger.info(f"Downloading data for {ticker}...")
        downloader = HistoricalDataDownloader()
        data = downloader.download_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
        
        logger.info(f"Downloaded {len(data)} records")
        return data
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        high_column: Optional[str] = None,
        low_column: Optional[str] = None,
        open_column: Optional[str] = None,
        volume_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Engineer financial features from OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Name of close price column
            high_column: Name of high price column
            low_column: Name of low price column
            open_column: Name of open price column
            volume_column: Name of volume column
        
        Returns:
            DataFrame with engineered features
        """
        if FinancialFeatureEngineer is None:
            raise ImportError("FinancialFeatureEngineer not available. Install required dependencies.")
        
        logger.info("Engineering features...")
        
        # Initialize feature engineer if not already done
        if self.feature_engineer is None:
            self.feature_engineer = FinancialFeatureEngineer()
        
        # Engineer features
        features_df = self.feature_engineer.engineer_features(
            data,
            price_column=price_column,
            high_column=high_column,
            low_column=low_column,
            open_column=open_column,
            volume_column=volume_column,
            include_returns=True,
            include_ma=True,
            include_volatility=True,
            include_trend=True
        )
        
        logger.info(f"Engineered {len(features_df.columns)} features")
        return features_df
    
    def time_series_split(
        self,
        data: pd.DataFrame,
        date_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically for time-series (not random split).
        
        Args:
            data: DataFrame with features
            date_column: Name of date column. If None, uses index
        
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        # Sort by date if date column exists
        if date_column and date_column in data.columns:
            data = data.sort_values(date_column).reset_index(drop=True)
        elif 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        
        n_samples = len(data)
        
        # Calculate split indices
        test_start = int(n_samples * (1 - self.test_size))
        val_start = int(test_start * (1 - self.validation_size))
        
        train_data = data.iloc[:val_start].copy()
        val_data = data.iloc[val_start:test_start].copy()
        test_data = data.iloc[test_start:].copy()
        
        logger.info(f"Time-series split:")
        logger.info(f"  Train: {len(train_data)} samples ({len(train_data)/n_samples*100:.1f}%)")
        logger.info(f"  Validation: {len(val_data)} samples ({len(val_data)/n_samples*100:.1f}%)")
        logger.info(f"  Test: {len(test_data)} samples ({len(test_data)/n_samples*100:.1f}%)")
        
        # Log date ranges if available
        date_col = date_column or ('date' if 'date' in data.columns else None)
        if date_col:
            logger.info(f"  Train period: {train_data[date_col].min()} to {train_data[date_col].max()}")
            logger.info(f"  Validation period: {val_data[date_col].min()} to {val_data[date_col].max()}")
            logger.info(f"  Test period: {test_data[date_col].min()} to {test_data[date_col].max()}")
        
        return train_data, val_data, test_data
    
    def prepare_features_and_target(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        forecast_horizon: int = 1,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for training.
        
        Args:
            data: DataFrame with features
            target_column: Name of target column
            forecast_horizon: Number of periods ahead to predict
            exclude_columns: Columns to exclude from features
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        if exclude_columns is None:
            exclude_columns = ['date', 'ticker', 'timestamp']
        
        # Create target (shifted by forecast_horizon)
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        y = data[target_column].shift(-forecast_horizon).values
        
        # Remove rows with NaN (created by shifting)
        valid_indices = ~np.isnan(y)
        data = data[valid_indices].copy()
        y = y[valid_indices]
        
        # Select feature columns
        feature_columns = [
            col for col in data.columns
            if col != target_column and col not in exclude_columns
        ]
        
        # Keep only numeric columns
        feature_columns = [
            col for col in feature_columns
            if pd.api.types.is_numeric_dtype(data[col])
        ]
        
        X = data[feature_columns].values
        
        logger.info(f"Prepared {len(feature_columns)} features for {len(X)} samples")
        
        return X, y, feature_columns
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[str] = 'sqrt',
        scale_features: bool = True,
        **kwargs
    ) -> Dict:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            n_estimators: Number of trees. Default: 100
            max_depth: Maximum tree depth. Default: 10
            min_samples_split: Minimum samples to split. Default: 2
            min_samples_leaf: Minimum samples in leaf. Default: 1
            max_features: Features to consider. Default: 'sqrt'
            scale_features: Whether to scale features. Default: True
            **kwargs: Additional Random Forest parameters
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training Random Forest model...")
        
        # Scale features
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val if X_val is not None else None
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1,
            **kwargs
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train_scaled)
        train_metrics = self._calculate_metrics(y_train, y_train_pred, prefix="train")
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val_scaled)
            val_metrics = self._calculate_metrics(y_val, y_val_pred, prefix="val")
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Store metadata
        self.training_metadata = {
            'model_type': 'RandomForestRegressor',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'scale_features': scale_features,
            'feature_count': X_train.shape[1],
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'training_date': datetime.now().isoformat(),
            **metrics
        }
        
        logger.info("Training completed!")
        logger.info(f"Train R²: {metrics.get('train_r2', 0):.4f}")
        if 'val_r2' in metrics:
            logger.info(f"Validation R²: {metrics['val_r2']:.4f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names
        
        Returns:
            Dictionary with metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Additional metrics
        mean_error = np.mean(y_pred - y_true)
        std_error = np.std(y_pred - y_true)
        
        prefix_str = f"{prefix}_" if prefix else ""
        
        return {
            f'{prefix_str}mse': float(mse),
            f'{prefix_str}rmse': float(rmse),
            f'{prefix_str}mae': float(mae),
            f'{prefix_str}mape': float(mape),
            f'{prefix_str}r2': float(r2),
            f'{prefix_str}mean_error': float(mean_error),
            f'{prefix_str}std_error': float(std_error)
        }
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scale_features: bool = True
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            scale_features: Whether to scale features. Default: True
        
        Returns:
            Dictionary with test metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating on test set...")
        
        # Scale features if needed
        if scale_features:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        test_metrics = self._calculate_metrics(y_test, y_pred, prefix="test")
        
        logger.info("Test Evaluation Results:")
        logger.info(f"  RMSE: ${test_metrics['test_rmse']:.2f}")
        logger.info(f"  MAE: ${test_metrics['test_mae']:.2f}")
        logger.info(f"  MAPE: {test_metrics['test_mape']:.2f}%")
        logger.info(f"  R²: {test_metrics['test_r2']:.4f}")
        
        return test_metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available.")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(
        self,
        ticker: str,
        model_name: Optional[str] = None,
        save_metadata: bool = True
    ) -> Dict[str, Path]:
        """
        Save trained model to disk.
        
        Args:
            ticker: Stock ticker symbol
            model_name: Custom model name. If None, auto-generated
            save_metadata: Whether to save metadata. Default: True
        
        Returns:
            Dictionary with paths to saved files
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Generate model name
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{ticker}_rf_{timestamp}"
        
        saved_paths = {}
        
        # Save model
        model_path = self.model_dir / f"{model_name}.pkl"
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_metadata': self.training_metadata
        }
        joblib.dump(model_data, model_path)
        saved_paths['model'] = model_path
        logger.info(f"Model saved to: {model_path}")
        
        # Save metadata
        if save_metadata:
            metadata_path = self.model_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.training_metadata, f, indent=2)
            saved_paths['metadata'] = metadata_path
            logger.info(f"Metadata saved to: {metadata_path}")
        
        # Save feature importance
        if self.feature_names is not None:
            importance_df = self.get_feature_importance()
            importance_path = self.model_dir / f"{model_name}_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            saved_paths['importance'] = importance_path
            logger.info(f"Feature importance saved to: {importance_path}")
        
        return saved_paths
    
    def load_model(self, model_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model file
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.training_metadata = model_data.get('training_metadata', {})
        
        logger.info(f"Model loaded from: {model_path}")
    
    def run_full_pipeline(
        self,
        ticker: str,
        data: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "2y",
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        scale_features: bool = True,
        save_model: bool = True
    ) -> Dict:
        """
        Run complete training pipeline from data loading to model saving.
        
        Args:
            ticker: Stock ticker symbol
            data: Pre-loaded data (optional)
            start_date: Start date for data download
            end_date: End date for data download
            period: Period for data download
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            scale_features: Whether to scale features
            save_model: Whether to save model. Default: True
        
        Returns:
            Dictionary with all metrics and results
        """
        logger.info("=" * 60)
        logger.info(f"Starting full training pipeline for {ticker}")
        logger.info("=" * 60)
        
        # Step 1: Load data
        logger.info("\nStep 1: Loading data...")
        raw_data = self.load_data(
            data=data,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            period=period
        )
        
        # Step 2: Engineer features
        logger.info("\nStep 2: Engineering features...")
        features_data = self.engineer_features(raw_data)
        
        # Step 3: Time-series split
        logger.info("\nStep 3: Splitting data (time-series)...")
        train_data, val_data, test_data = self.time_series_split(features_data)
        
        # Step 4: Prepare features and targets
        logger.info("\nStep 4: Preparing features and targets...")
        X_train, y_train, feature_names = self.prepare_features_and_target(train_data)
        X_val, y_val, _ = self.prepare_features_and_target(val_data)
        X_test, y_test, _ = self.prepare_features_and_target(test_data)
        
        self.feature_names = feature_names
        
        # Step 5: Train model
        logger.info("\nStep 5: Training model...")
        train_metrics = self.train(
            X_train, y_train,
            X_val, y_val,
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_features=scale_features
        )
        
        # Step 6: Evaluate on test set
        logger.info("\nStep 6: Evaluating on test set...")
        test_metrics = self.evaluate(X_test, y_test, scale_features=scale_features)
        
        # Step 7: Save model
        if save_model:
            logger.info("\nStep 7: Saving model...")
            saved_paths = self.save_model(ticker)
        else:
            saved_paths = {}
        
        # Compile results
        results = {
            'ticker': ticker,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_count': len(feature_names),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'saved_paths': saved_paths,
            'feature_importance': self.get_feature_importance().to_dict('records')
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 60)
        
        return results


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("ML Training Pipeline - Example Usage")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = MLTrainingPipeline(
        model_dir="models/ml",
        test_size=0.2,
        validation_size=0.1
    )
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        ticker="AAPL",
        period="2y",
        n_estimators=100,
        max_depth=10,
        scale_features=True,
        save_model=True
    )
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Ticker: {results['ticker']}")
    print(f"Features: {results['feature_count']}")
    print(f"Train Samples: {results['train_samples']}")
    print(f"Test Samples: {results['test_samples']}")
    print(f"\nTest Metrics:")
    print(f"  RMSE: ${results['test_metrics']['test_rmse']:.2f}")
    print(f"  MAE: ${results['test_metrics']['test_mae']:.2f}")
    print(f"  MAPE: {results['test_metrics']['test_mape']:.2f}%")
    print(f"  R²: {results['test_metrics']['test_r2']:.4f}")
    
    print("\nTop 10 Features:")
    for i, feat in enumerate(results['feature_importance'][:10], 1):
        print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")













