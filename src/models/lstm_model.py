"""
LSTM (Long Short-Term Memory) model for financial time series forecasting.
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, Optional
from loguru import logger
from config import settings
import os


class LSTMForecaster:
    """LSTM model for time series forecasting."""
    
    def __init__(
        self,
        sequence_length: int = None,
        lstm_units: list = [50, 50],
        dropout_rate: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Length of input sequences
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        self.sequence_length = sequence_length or settings.LSTM_SEQUENCE_LENGTH
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.logger = logger
    
    def create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            data: Feature array
            target: Target array
            sequence_length: Length of sequences
        
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i - sequence_length:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        feature_columns: Optional[list] = None,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Args:
            data: DataFrame with features
            target_column: Column to predict
            feature_columns: List of feature columns to use
            scale_features: Whether to scale features
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        if feature_columns is None:
            exclude_cols = [
                target_column, 'symbol', 'date', 'target',
                'future_close', 'future_return'
            ]
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Extract features and target
        features = data[feature_columns].values
        target = data[target_column].values
        
        # Scale features
        if scale_features:
            features = self.feature_scaler.fit_transform(features)
            target = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(features, target, self.sequence_length)
        
        # Split data (80/20, maintaining temporal order)
        split_idx = int(len(X) * (1 - settings.LSTM_VALIDATION_SPLIT))
        
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        
        # Reshape for LSTM: (samples, timesteps, features)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self, input_shape: Tuple) -> keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        if self.bidirectional:
            model.add(Bidirectional(
                LSTM(self.lstm_units[0], return_sequences=len(self.lstm_units) > 1),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                self.lstm_units[0],
                return_sequences=len(self.lstm_units) > 1,
                input_shape=input_shape
            ))
        
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for units in self.lstm_units[1:]:
            if self.bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=False)))
            else:
                model.add(LSTM(units, return_sequences=False))
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = None,
        batch_size: int = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        
        Returns:
            Dictionary with training history and metrics
        """
        epochs = epochs or settings.LSTM_EPOCHS
        batch_size = batch_size or settings.LSTM_BATCH_SIZE
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Train model
        self.logger.info("Training LSTM model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate
        y_train_pred = self.model.predict(X_train, verbose=0)
        y_test_pred = self.model.predict(X_test, verbose=0)
        
        # Inverse transform if scaled
        if hasattr(self.scaler, 'inverse_transform'):
            y_train = self.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_train_pred = self.scaler.inverse_transform(y_train_pred).flatten()
            y_test_pred = self.scaler.inverse_transform(y_test_pred).flatten()
        
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'history': history.history
        }
        
        self.logger.info(f"Training completed. Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature array
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        
        # Inverse transform if scaled
        if hasattr(self.scaler, 'inverse_transform'):
            predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions
    
    def forecast_future(
        self,
        last_sequence: np.ndarray,
        steps: int = 1
    ) -> np.ndarray:
        """
        Forecast future values.
        
        Args:
            last_sequence: Last sequence of data
            steps: Number of steps to forecast
        
        Returns:
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            if len(current_sequence.shape) == 2:
                pred_input = current_sequence.reshape(
                    (1, current_sequence.shape[0], current_sequence.shape[1])
                )
            else:
                pred_input = current_sequence.reshape(
                    (1, current_sequence.shape[0], 1)
                )
            
            # Predict next value
            pred = self.model.predict(pred_input, verbose=0)
            
            # Inverse transform if scaled
            if hasattr(self.scaler, 'inverse_transform'):
                pred = self.scaler.inverse_transform(pred)[0, 0]
            else:
                pred = pred[0, 0]
            
            forecasts.append(pred)
            
            # Update sequence (shift and append prediction)
            # Note: This is simplified; in practice, you'd need to update features
            current_sequence = np.roll(current_sequence, -1, axis=0)
            if len(current_sequence.shape) == 2:
                current_sequence[-1, 0] = pred  # Assuming first column is price
            else:
                current_sequence[-1] = pred
        
        return np.array(forecasts)
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(filepath)
        
        # Save scalers
        import joblib
        scaler_path = filepath.replace('.h5', '_scalers.pkl')
        joblib.dump({
            'scaler': self.scaler,
            'feature_scaler': self.feature_scaler,
            'sequence_length': self.sequence_length
        }, scaler_path)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        
        # Load scalers
        import joblib
        scaler_path = filepath.replace('.h5', '_scalers.pkl')
        if os.path.exists(scaler_path):
            scalers = joblib.load(scaler_path)
            self.scaler = scalers['scaler']
            self.feature_scaler = scalers['feature_scaler']
            self.sequence_length = scalers['sequence_length']
        
        self.logger.info(f"Model loaded from {filepath}")













