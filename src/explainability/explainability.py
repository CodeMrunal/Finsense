"""
Model explainability module using SHAP and feature importance.
Provides interpretable insights for financial predictions.
"""
import numpy as np
import pandas as pd
import shap
from typing import Dict, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class ModelExplainability:
    """Provides explainability for ML models."""
    
    def __init__(self):
        """Initialize explainability module."""
        self.logger = logger
        self.explainer = None
    
    def explain_with_shap(
        self,
        model,
        X: np.ndarray,
        feature_names: Optional[list] = None,
        model_type: str = "tree"
    ) -> Dict:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', 'deep')
        
        Returns:
            Dictionary with SHAP values and explanations
        """
        try:
            # Create SHAP explainer based on model type
            if model_type == "tree":
                explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                explainer = shap.LinearExplainer(model, X)
            elif model_type == "deep":
                explainer = shap.DeepExplainer(model, X)
            else:
                explainer = shap.KernelExplainer(model.predict, X[:100])  # Sample for speed
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate feature importance from SHAP
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(feature_importance))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Summary statistics
            summary = {
                'shap_values': shap_values,
                'feature_importance': importance_df,
                'mean_abs_shap': np.abs(shap_values).mean(),
                'explainer': explainer
            }
            
            self.logger.info("SHAP explanations generated successfully")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanations: {str(e)}")
            return {}
    
    def explain_prediction(
        self,
        model,
        X_instance: np.ndarray,
        feature_names: Optional[list] = None,
        model_type: str = "tree"
    ) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            model: Trained model
            X_instance: Single instance to explain
            feature_names: List of feature names
            model_type: Type of model
        
        Returns:
            Dictionary with prediction explanation
        """
        try:
            # Reshape if needed
            if len(X_instance.shape) == 1:
                X_instance = X_instance.reshape(1, -1)
            
            # Get prediction
            prediction = model.predict(X_instance)[0]
            
            # Generate SHAP explanation
            explainer = self.explain_with_shap(model, X_instance, feature_names, model_type)
            
            if explainer:
                shap_values = explainer['shap_values']
                
                if feature_names is None:
                    feature_names = [f"Feature_{i}" for i in range(len(shap_values[0]))]
                
                # Create feature contribution dataframe
                contributions = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_values[0],
                    'contribution': shap_values[0]
                }).sort_values('contribution', key=abs, ascending=False)
                
                return {
                    'prediction': prediction,
                    'contributions': contributions,
                    'top_positive_features': contributions.nlargest(5, 'contribution'),
                    'top_negative_features': contributions.nsmallest(5, 'contribution')
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {str(e)}")
            return {}
    
    def get_feature_importance(
        self,
        model,
        feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        
        Returns:
            DataFrame with feature importance
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                self.logger.warning("Model does not support feature importance")
                return pd.DataFrame()
            
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(importance))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def explain_lstm_prediction(
        self,
        model,
        X_instance: np.ndarray,
        feature_names: Optional[list] = None
    ) -> Dict:
        """
        Explain LSTM prediction using gradient-based methods.
        
        Args:
            model: Trained LSTM model
            X_instance: Input sequence
            feature_names: List of feature names
        
        Returns:
            Dictionary with explanation
        """
        try:
            import tensorflow as tf
            
            # Reshape if needed
            if len(X_instance.shape) == 2:
                X_instance = X_instance.reshape(1, X_instance.shape[0], X_instance.shape[1])
            
            # Convert to tensor
            X_tensor = tf.constant(X_instance, dtype=tf.float32)
            
            # Calculate gradients
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                prediction = model(X_tensor)
            
            gradients = tape.gradient(prediction, X_tensor)
            
            # Calculate feature importance (mean absolute gradient)
            importance = np.abs(gradients.numpy()).mean(axis=(0, 1))
            
            if feature_names is None:
                feature_names = [f"Feature_{i}" for i in range(len(importance))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return {
                'prediction': prediction.numpy()[0, 0],
                'feature_importance': importance_df,
                'gradients': gradients.numpy()
            }
            
        except Exception as e:
            self.logger.error(f"Error explaining LSTM prediction: {str(e)}")
            return {}
    
    def generate_explanation_report(
        self,
        model,
        X: np.ndarray,
        predictions: np.ndarray,
        feature_names: Optional[list] = None,
        model_type: str = "tree"
    ) -> Dict:
        """
        Generate comprehensive explanation report.
        
        Args:
            model: Trained model
            X: Feature matrix
            predictions: Model predictions
            feature_names: List of feature names
            model_type: Type of model
        
        Returns:
            Dictionary with comprehensive explanations
        """
        report = {
            'predictions': predictions,
            'prediction_stats': {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
        }
        
        # Feature importance
        feature_importance = self.get_feature_importance(model, feature_names)
        if not feature_importance.empty:
            report['feature_importance'] = feature_importance
        
        # SHAP explanations
        shap_explanation = self.explain_with_shap(model, X, feature_names, model_type)
        if shap_explanation:
            report['shap_explanation'] = shap_explanation
        
        # Sample predictions explanation
        sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
        sample_explanations = []
        
        for idx in sample_indices:
            explanation = self.explain_prediction(
                model, X[idx:idx+1], feature_names, model_type
            )
            if explanation:
                sample_explanations.append({
                    'index': int(idx),
                    'prediction': float(predictions[idx]),
                    'explanation': explanation
                })
        
        report['sample_explanations'] = sample_explanations
        
        return report













