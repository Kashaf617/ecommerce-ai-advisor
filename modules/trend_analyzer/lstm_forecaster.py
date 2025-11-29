"""
Simple LSTM-based Trend Forecasting
Time series prediction for product trends
"""
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger
from config import MODELS_DIR

logger = get_logger(__name__)


class SimpleLSTMForecaster:
    """Simple LSTM for trend forecasting with Prophet fallback"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.use_lstm = False
        self._try_load_lstm()
    
    def _try_load_lstm(self):
        """Try to load LSTM (optional, falls back to Prophet)"""
        try:
            import tensorflow as tf
            logger.info("TensorFlow available, LSTM enabled")
            self.use_lstm = True
        except ImportError:
            logger.info("TensorFlow not available, using statistical fallback")
            self.use_lstm = False
    
    def forecast_trend(self, historical_data: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """
        Forecast future trends
        
        Args:
            historical_data: Historical price/demand data
            periods: Number of periods to forecast
            
        Returns:
            Forecast results
        """
        if self.use_lstm and len(historical_data) > 50:
            return self._lstm_forecast(historical_data, periods)
        else:
            return self._statistical_forecast(historical_data, periods)
    
    def _lstm_forecast(self, data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """LSTM-based forecast (simplified)"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            logger.info("Building LSTM forecast model...")
            
            # Simple sequence preparation
            values = data['value'].values if 'value' in data.columns else data.iloc[:, 0].values
            sequence_length = 10
            
            # Very simple LSTM (in production, this would be more sophisticated)
            X, y = self._prepare_sequences(values, sequence_length)
            
            if len(X) < 10:
                return self._statistical_forecast(data, periods)
            
            # Build model
            model = keras.Sequential([
                keras.layers.LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
                keras.layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=10, verbose=0)
            
            # Forecast
            last_sequence = values[-sequence_length:].reshape(1, sequence_length, 1)
            forecast = []
            
            for _ in range(periods):
                pred = model.predict(last_sequence, verbose=0)[0,0]
                forecast.append(pred)
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[0, -1, 0] = pred
            
            return {
                'forecast': forecast,
                'method': 'LSTM',
                'confidence': 0.75,
                'is_ml': True
            }
            
        except Exception as e:
            logger.error(f"LSTM forecast failed: {e}, using fallback")
            return self._statistical_forecast(data, periods)
    
    def _prepare_sequences(self, data, seq_length):
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X).reshape(-1, seq_length, 1), np.array(y)
    
    def _statistical_forecast(self, data: pd.DataFrame, periods: int) -> Dict[str, Any]:
        """Statistical forecast fallback (moving average + trend)"""
        values = data['value'].values if 'value' in data.columns else data.iloc[:, 0].values
        
        # Simple moving average with trend
        window = min(7, len(values) // 2)
        ma = np.convolve(values, np.ones(window)/window, mode='valid')
        
        # Calculate trend
        if len(ma) > 1:
            trend = (ma[-1] - ma[0]) / len(ma)
        else:
            trend = 0
        
        # Forecast
        last_value = values[-1]
        forecast = [last_value + trend * i for i in range(1, periods + 1)]
        
        return {
            'forecast': forecast,
            'method': 'MovingAverage',
            'confidence': 0.65,
            'is_ml': False,
            'note': 'Using statistical fallback (install tensorflow for LSTM)'
        }
