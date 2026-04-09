"""
Traditional Machine Learning models for financial forecasting.
Includes Random Forest, XGBoost, and other ensemble methods.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import joblib
from typing import Dict, Tuple, Optional
from loguru import logger
from config import settings


class AdaptiveModelSelector:
    """
    Adaptive model selector for market regimes.

    Selection rule:
    - If volatility is high -> xgboost
    - If trend is stable -> random_forest
    - Else -> gradient_boosting
    """

    def __init__(
        self,
        high_volatility_threshold: float = 0.30,
        stable_trend_threshold: float = 0.01
    ):
        self.high_volatility_threshold = high_volatility_threshold
        self.stable_trend_threshold = stable_trend_threshold
        self.logger = logger

    def select_model(
        self,
        trend: float,
        volatility: float,
        rsi: float
    ) -> str:
        """
        Select best model based on market conditions.

        Args:
            trend: Trend strength proxy (absolute slope / return proxy)
            volatility: Annualized volatility in decimal (e.g., 0.25 for 25%)
            rsi: RSI value

        Returns:
            Model name compatible with MLForecaster.
        """
        vol = float(volatility) if volatility is not None else 0.0
        tr = abs(float(trend)) if trend is not None else 0.0
        _ = rsi  # Reserved for future regime refinement

        if vol >= self.high_volatility_threshold:
            model = "xgboost"
        elif tr <= self.stable_trend_threshold:
            model = "random_forest"
        else:
            model = "gradient_boosting"

        self.logger.info(
            f"Adaptive selection -> model={model}, trend={tr:.4f}, volatility={vol:.4f}, rsi={rsi}"
        )
        return model


class MLForecaster:
    """Machine Learning models for price forecasting."""
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize ML forecaster.
        
        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'extra_trees', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.logger = logger
        
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "extra_trees":
            self.model = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = 'close',
        forecast_horizon: int = 1,
        feature_columns: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame with features
            target_column: Column to predict
            forecast_horizon: Number of periods ahead to forecast
            feature_columns: List of feature columns to use
        
        Returns:
            Tuple of (X, y) arrays
        """
        if feature_columns is None:
            # Exclude target and metadata columns
            exclude_cols = [
                target_column, 'symbol', 'date', 'target',
                'future_close', 'future_return'
            ]
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Create target (shifted by forecast_horizon)
        if target_column in data.columns:
            y = data[target_column].shift(-forecast_horizon).values
        else:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Select features
        X = data[feature_columns].values
        
        # Remove rows with NaN (created by shifting)
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        self.feature_names = feature_columns
        
        return X, y
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = None,
        scale_features: bool = True,
        hyperparameter_tuning: bool = False
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            scale_features: Whether to scale features
            hyperparameter_tuning: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with training metrics
        """
        if test_size is None:
            test_size = settings.ML_TRAIN_TEST_SPLIT
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        if hyperparameter_tuning:
            self._tune_hyperparameters(X_train, y_train)
        
        # Train model
        self.logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred),
        }
        
        self.logger.info(f"Training completed. Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray):
        """Perform hyperparameter tuning."""
        self.logger.info("Performing hyperparameter tuning...")
        
        if self.model_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == "extra_trees":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 16],
                'min_samples_split': [2, 5]
            }
        elif self.model_type == "gradient_boosting":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [2, 3, 4],
                'learning_rate': [0.03, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
        elif self.model_type == "xgboost":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        else:
            return
        
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
    
    def predict(self, X: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            scale_features: Whether to scale features
        
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if scale_features:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            raise ValueError("Model does not support feature importance.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.logger.info(f"Model loaded from {filepath}")


class MLDirectionClassifier:
    """
    Classification-based directional predictor for financial time series.

    Target:
    y = (close.shift(-forecast_horizon) > close).astype(int)
    """

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.logger = logger

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                random_state=42
            )
        elif model_type == "extra_trees":
            self.model = ExtraTreesClassifier(
                n_estimators=400,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample"
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            )
        else:
            raise ValueError(f"Unknown classifier type: {model_type}")

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "close",
        forecast_horizon: int = 5,
        feature_columns: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if feature_columns is None:
            exclude_cols = {
                target_column, "symbol", "date", "target",
                "future_close", "future_return", "direction_target"
            }
            feature_columns = [c for c in data.columns if c not in exclude_cols]

        y = (data[target_column].shift(-forecast_horizon) > data[target_column]).astype(int).values
        X = data[feature_columns].values
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        self.feature_names = feature_columns
        return X, y

    def _build_param_grid(self):
        if self.model_type == "random_forest":
            return {
                "n_estimators": [200, 300],
                "max_depth": [8, 12],
                "min_samples_split": [2, 5]
            }
        if self.model_type == "extra_trees":
            return {
                "n_estimators": [250, 400],
                "max_depth": [10, 14],
                "min_samples_split": [2, 5]
            }
        if self.model_type == "gradient_boosting":
            return {
                "n_estimators": [150, 250],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [2, 3]
            }
        if self.model_type == "xgboost":
            return {
                "n_estimators": [200, 300],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.03, 0.05, 0.1]
            }
        return {}

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        hyperparameter_tuning: bool = True,
        confidence_threshold: float = 0.55
    ) -> Dict:
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Class balancing for XGBoost
        if self.model_type == "xgboost":
            pos = max(1, int(np.sum(y_train == 1)))
            neg = max(1, int(np.sum(y_train == 0)))
            self.model.set_params(scale_pos_weight=neg / pos)

        if hyperparameter_tuning:
            param_grid = self._build_param_grid()
            if param_grid:
                tscv = TimeSeriesSplit(n_splits=3)
                grid = GridSearchCV(
                    self.model,
                    param_grid=param_grid,
                    cv=tscv,
                    scoring="f1",
                    n_jobs=-1,
                    verbose=0
                )
                grid.fit(X_train, y_train)
                self.model = grid.best_estimator_

        self.model.fit(X_train, y_train)

        proba_up = self.model.predict_proba(X_test)[:, 1]
        y_pred = (proba_up >= 0.5).astype(int)

        confident_mask = (proba_up >= confidence_threshold) | (proba_up <= (1 - confidence_threshold))
        confident_pct = float(np.mean(confident_mask) * 100) if len(confident_mask) else 0.0

        # Threshold-based decision: HOLD if low confidence
        decision = np.where(
            proba_up >= confidence_threshold,
            1,
            np.where(proba_up <= (1 - confidence_threshold), 0, -1)
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "directional_accuracy": float(accuracy_score(y_test, y_pred)),
            "confident_predictions_pct": confident_pct,
            "hold_predictions_pct": float(np.mean(decision == -1) * 100),
            "test_size": int(len(y_test)),
            "positive_class_ratio_train": float(np.mean(y_train)),
        }
        return metrics

    def predict_with_decision(
        self,
        X: np.ndarray,
        confidence_threshold: float = 0.55
    ) -> Dict[str, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        proba_up = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = (proba_up >= 0.5).astype(int)
        decision = np.where(
            proba_up >= confidence_threshold,
            "BUY",
            np.where(proba_up <= (1 - confidence_threshold), "SELL", "HOLD")
        )
        return {
            "proba_up": proba_up,
            "predicted_direction": y_pred,
            "decision": decision
        }

    def get_feature_importance(self) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not expose feature importance.")
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

    def save_model(self, filepath: str):
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type
        }
        joblib.dump(payload, filepath)

    def load_model(self, filepath: str):
        payload = joblib.load(filepath)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_names = payload["feature_names"]
        self.model_type = payload["model_type"]













