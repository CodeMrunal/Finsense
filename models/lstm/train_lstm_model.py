"""
LSTM Training Pipeline for Stock Price Forecasting.

This module provides a complete training pipeline using TensorFlow/Keras LSTM models
with sequential time-series data preparation, training, validation, and model saving.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import json
import joblib
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
    FinancialFeatureEngineer = None
    HistoricalDataDownloader = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set TensorFlow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMTrainingPipeline:
    """
    Complete LSTM training pipeline for stock price forecasting.
    
    Features:
    - Sequential time-series data preparation
    - TensorFlow/Keras LSTM model architecture
    - Time-series aware train/validation split
    - Model training with callbacks
    - Comprehensive evaluation metrics
    - Model persistence with scalers
    """
    
    def __init__(
        self,
        model_dir: str = "models/lstm",
        sequence_length: int = 60,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the LSTM training pipeline.
        
        Args:
            model_dir: Directory to save models. Default: "models/lstm"
            sequence_length: Length of input sequences. Default: 60
            test_size: Proportion of data for testing. Default: 0.2
            validation_size: Proportion of training data for validation. Default: 0.1
            random_state: Random state for reproducibility. Default: 42
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        self.model = None
        self.target_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.feature_engineer = None
        self.feature_names = None
        self.training_history = None
        self.training_metadata = {}
        
        logger.info(f"LSTMTrainingPipeline initialized")
        logger.info(f"  Model directory: {self.model_dir}")
        logger.info(f"  Sequence length: {self.sequence_length}")
    
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
    
    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[List[str]] = None,
        scale_features: bool = True,
        scale_target: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequential data for LSTM training.
        
        Args:
            data: DataFrame with features
            target_column: Name of target column
            feature_columns: List of feature columns. If None, auto-selects
            scale_features: Whether to scale features. Default: True
            scale_target: Whether to scale target. Default: True
        
        Returns:
            Tuple of (X_sequences, y_targets, feature_names)
        """
        # Select feature columns
        if feature_columns is None:
            exclude_cols = ['date', 'ticker', 'timestamp', target_column]
            feature_columns = [
                col for col in data.columns
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])
            ]
        
        # Extract features and target
        X_data = data[feature_columns].values
        y_data = data[target_column].values
        
        # Scale features
        if scale_features:
            X_data = self.feature_scaler.fit_transform(X_data)
        
        # Scale target
        if scale_target:
            y_data = self.target_scaler.fit_transform(y_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_data)):
            X_sequences.append(X_data[i - self.sequence_length:i])
            y_sequences.append(y_data[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Created {len(X_sequences)} sequences")
        logger.info(f"  Sequence shape: {X_sequences.shape}")
        logger.info(f"  Target shape: {y_sequences.shape}")
        logger.info(f"  Features: {len(feature_columns)}")
        
        return X_sequences, y_sequences, feature_columns
    
    def build_model(
        self,
        input_shape: Tuple[int, int],
        lstm_units: List[int] = [50, 50],
        dropout_rate: float = 0.2,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        learning_rate: float = 0.001
    ) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            lstm_units: List of units for each LSTM layer. Default: [50, 50]
            dropout_rate: Dropout rate. Default: 0.2
            bidirectional: Whether to use bidirectional LSTM. Default: False
            use_batch_norm: Whether to use batch normalization. Default: True
            learning_rate: Learning rate for optimizer. Default: 0.001
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building LSTM model...")
        logger.info(f"  Input shape: {input_shape}")
        logger.info(f"  LSTM units: {lstm_units}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(f"  Bidirectional: {bidirectional}")
        
        model = Sequential()
        
        # First LSTM layer
        if bidirectional:
            model.add(Bidirectional(
                LSTM(
                    lstm_units[0],
                    return_sequences=len(lstm_units) > 1,
                    input_shape=input_shape
                )
            ))
        else:
            model.add(LSTM(
                lstm_units[0],
                return_sequences=len(lstm_units) > 1,
                input_shape=input_shape
            ))
        
        if use_batch_norm:
            model.add(BatchNormalization())
        
        model.add(Dropout(dropout_rate))
        
        # Additional LSTM layers
        for units in lstm_units[1:]:
            if bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=False)))
            else:
                model.add(LSTM(units, return_sequences=False))
            
            if use_batch_norm:
                model.add(BatchNormalization())
            
            model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Model built successfully")
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        lstm_units: List[int] = [50, 50],
        dropout_rate: float = 0.2,
        bidirectional: bool = False,
        use_batch_norm: bool = True,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        verbose: int = 1
    ) -> Dict:
        """
        Train LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs. Default: 50
            batch_size: Batch size. Default: 32
            lstm_units: List of LSTM units. Default: [50, 50]
            dropout_rate: Dropout rate. Default: 0.2
            bidirectional: Whether to use bidirectional LSTM. Default: False
            use_batch_norm: Whether to use batch normalization. Default: True
            learning_rate: Learning rate. Default: 0.001
            early_stopping_patience: Patience for early stopping. Default: 10
            reduce_lr_patience: Patience for learning rate reduction. Default: 5
            verbose: Verbosity level. Default: 1
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training LSTM model...")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(
            input_shape=input_shape,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            use_batch_norm=use_batch_norm,
            learning_rate=learning_rate
        )
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=0.0001,
                verbose=verbose
            )
        ]
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history = history.history
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train, verbose=0)
        train_metrics = self._calculate_metrics(y_train, y_train_pred, prefix="train")
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val, verbose=0)
            val_metrics = self._calculate_metrics(y_val, y_val_pred, prefix="val")
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        # Store metadata
        self.training_metadata = {
            'model_type': 'LSTM',
            'sequence_length': self.sequence_length,
            'lstm_units': lstm_units,
            'dropout_rate': dropout_rate,
            'bidirectional': bidirectional,
            'use_batch_norm': use_batch_norm,
            'learning_rate': learning_rate,
            'epochs_trained': len(history.history['loss']),
            'feature_count': X_train.shape[2],
            'train_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'batch_size': batch_size,
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
        prefix: str = "",
        inverse_scale: bool = True
    ) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True values (potentially scaled)
            y_pred: Predicted values (potentially scaled)
            prefix: Prefix for metric names
            inverse_scale: Whether to inverse transform before calculating metrics
        
        Returns:
            Dictionary with metrics
        """
        # Inverse transform if scaled
        if inverse_scale and hasattr(self.target_scaler, 'inverse_transform'):
            y_true_orig = self.target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_orig = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_orig = y_true
            y_pred_orig = y_pred
        
        mse = mean_squared_error(y_true_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_orig, y_pred_orig)
        r2 = r2_score(y_true_orig, y_pred_orig)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true_orig - y_pred_orig) / y_true_orig)) * 100
        
        # Additional metrics
        mean_error = np.mean(y_pred_orig - y_true_orig)
        std_error = np.std(y_pred_orig - y_true_orig)
        
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
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
        
        Returns:
            Dictionary with test metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating on test set...")
        
        # Make predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
        test_metrics = self._calculate_metrics(y_test, y_pred, prefix="test")
        
        logger.info("Test Evaluation Results:")
        logger.info(f"  RMSE: ${test_metrics['test_rmse']:.2f}")
        logger.info(f"  MAE: ${test_metrics['test_mae']:.2f}")
        logger.info(f"  MAPE: {test_metrics['test_mape']:.2f}%")
        logger.info(f"  R²: {test_metrics['test_r2']:.4f}")
        
        return test_metrics
    
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
            model_name = f"{ticker}_lstm_{timestamp}"
        
        saved_paths = {}
        
        # Save Keras model
        model_path = self.model_dir / f"{model_name}.h5"
        self.model.save(str(model_path))
        saved_paths['model'] = model_path
        logger.info(f"Model saved to: {model_path}")
        
        # Save scalers
        scalers_path = self.model_dir / f"{model_name}_scalers.pkl"
        scalers_data = {
            'target_scaler': self.target_scaler,
            'feature_scaler': self.feature_scaler,
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names
        }
        joblib.dump(scalers_data, scalers_path)
        saved_paths['scalers'] = scalers_path
        logger.info(f"Scalers saved to: {scalers_path}")
        
        # Save metadata
        if save_metadata:
            metadata_path = self.model_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.training_metadata, f, indent=2)
            saved_paths['metadata'] = metadata_path
            logger.info(f"Metadata saved to: {metadata_path}")
        
        # Save training history
        if self.training_history:
            history_path = self.model_dir / f"{model_name}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            saved_paths['history'] = history_path
            logger.info(f"Training history saved to: {history_path}")
        
        return saved_paths
    
    def load_model(self, model_path: str, scalers_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to saved model file (.h5)
            scalers_path: Path to saved scalers file (.pkl)
        """
        self.model = keras.models.load_model(model_path)
        
        scalers_data = joblib.load(scalers_path)
        self.target_scaler = scalers_data['target_scaler']
        self.feature_scaler = scalers_data['feature_scaler']
        self.sequence_length = scalers_data['sequence_length']
        self.feature_names = scalers_data.get('feature_names', None)
        
        logger.info(f"Model loaded from: {model_path}")
        logger.info(f"Scalers loaded from: {scalers_path}")
    
    def run_full_pipeline(
        self,
        ticker: str,
        data: Optional[pd.DataFrame] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = "2y",
        epochs: int = 50,
        batch_size: int = 32,
        lstm_units: List[int] = [50, 50],
        dropout_rate: float = 0.2,
        bidirectional: bool = False,
        learning_rate: float = 0.001,
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
            epochs: Number of training epochs
            batch_size: Batch size
            lstm_units: List of LSTM units
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            learning_rate: Learning rate
            save_model: Whether to save model. Default: True
        
        Returns:
            Dictionary with all metrics and results
        """
        logger.info("=" * 60)
        logger.info(f"Starting full LSTM training pipeline for {ticker}")
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
        
        # Step 4: Prepare sequences
        logger.info("\nStep 4: Preparing sequences...")
        X_train, y_train, feature_names = self.prepare_sequences(train_data)
        X_val, y_val, _ = self.prepare_sequences(val_data, scale_features=False, scale_target=False)
        X_test, y_test, _ = self.prepare_sequences(test_data, scale_features=False, scale_target=False)
        
        self.feature_names = feature_names
        
        # Step 5: Train model
        logger.info("\nStep 5: Training model...")
        train_metrics = self.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            learning_rate=learning_rate
        )
        
        # Step 6: Evaluate on test set
        logger.info("\nStep 6: Evaluating on test set...")
        test_metrics = self.evaluate(X_test, y_test)
        
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
            'sequence_length': self.sequence_length,
            'feature_count': len(feature_names),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'saved_paths': saved_paths
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("LSTM training pipeline completed successfully!")
        logger.info("=" * 60)
        
        return results


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("LSTM Training Pipeline - Example Usage")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = LSTMTrainingPipeline(
        model_dir="models/lstm",
        sequence_length=60,
        test_size=0.2,
        validation_size=0.1
    )
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        ticker="AAPL",
        period="2y",
        epochs=50,
        batch_size=32,
        lstm_units=[50, 50],
        dropout_rate=0.2,
        bidirectional=False,
        learning_rate=0.001,
        save_model=True
    )
    
    # Print summary
    print("\nTraining Summary:")
    print(f"Ticker: {results['ticker']}")
    print(f"Sequence Length: {results['sequence_length']}")
    print(f"Features: {results['feature_count']}")
    print(f"Train Samples: {results['train_samples']}")
    print(f"Test Samples: {results['test_samples']}")
    print(f"\nTest Metrics:")
    print(f"  RMSE: ${results['test_metrics']['test_rmse']:.2f}")
    print(f"  MAE: ${results['test_metrics']['test_mae']:.2f}")
    print(f"  MAPE: {results['test_metrics']['test_mape']:.2f}%")
    print(f"  R²: {results['test_metrics']['test_r2']:.4f}")













