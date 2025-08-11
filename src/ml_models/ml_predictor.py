"""
ML Predictor - Modelos de Machine Learning para predicción de precios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import joblib
import os
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class MLPrediction:
    symbol: str
    prediction: float
    confidence: float
    model_name: str
    features_used: List[str]
    timestamp: datetime
    horizon_hours: int

@dataclass
class ModelPerformance:
    model_name: str
    mae: float
    mse: float
    rmse: float
    accuracy_score: float
    training_samples: int

class MLPredictor:
    """
    Sistema de predicción usando ensemble de modelos ML
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'weight': 0.4
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'weight': 0.4
            },
            'linear_regression': {
                'model': LinearRegression(),
                'weight': 0.2
            }
        }
        
        self.feature_columns = [
            'close', 'volume', 'high', 'low',
            'price_change', 'volume_change',
            'rsi', 'bb_upper', 'bb_lower', 'bb_middle',
            'ema_12', 'ema_26', 'macd', 'macd_signal',
            'volatility', 'price_momentum',
            'volume_sma', 'price_sma'
        ]
        
        self.models_path = Path("data/models")
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features técnicos para el modelo
        """
        try:
            df = df.copy()
            
            # Price features
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            
            # Technical indicators
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['price_sma'] = df['close'].rolling(window=20).mean()
            price_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['price_sma'] + (price_std * 2)
            df['bb_lower'] = df['price_sma'] - (price_std * 2)
            df['bb_middle'] = df['price_sma']
            
            # EMA
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=20).std()
            
            # Momentum
            df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            # Target variable (próximo precio)
            df['target'] = df['close'].shift(-1)
            
            # Limpiar NaN
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            return df
    
    def train_models(self, df: pd.DataFrame, symbol: str) -> Dict[str, ModelPerformance]:
        """
        Entrena todos los modelos con los datos históricos
        """
        try:
            logger.info(f"Entrenando modelos para {symbol}")
            
            # Preparar features
            df_features = self.prepare_features(df)
            
            if len(df_features) < 100:
                logger.warning(f"Pocos datos para entrenar ({len(df_features)} muestras)")
                return {}
            
            # Separar features y target
            available_features = [col for col in self.feature_columns if col in df_features.columns]
            X = df_features[available_features]
            y = df_features['target']
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[symbol] = scaler
            
            performances = {}
            trained_models = {}
            
            # Entrenar cada modelo
            for model_name, config in self.model_configs.items():
                try:
                    model = config['model']
                    
                    # Entrenar
                    model.fit(X_train_scaled, y_train)
                    
                    # Predecir
                    y_pred = model.predict(X_test_scaled)
                    
                    # Métricas
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    # Accuracy (% predicciones dentro del 2%)
                    accuracy = np.mean(np.abs((y_pred - y_test) / y_test) < 0.02) * 100
                    
                    performance = ModelPerformance(
                        model_name=model_name,
                        mae=mae,
                        mse=mse,
                        rmse=rmse,
                        accuracy_score=accuracy,
                        training_samples=len(X_train)
                    )
                    
                    performances[model_name] = performance
                    trained_models[model_name] = model
                    
                    logger.info(f"{model_name}: MAE={mae:.2f}, Accuracy={accuracy:.1f}%")
                    
                except Exception as e:
                    logger.error(f"Error entrenando {model_name}: {e}")
            
            # Guardar modelos entrenados
            self.models[symbol] = trained_models
            self._save_models(symbol)
            
            return performances
            
        except Exception as e:
            logger.error(f"Error en entrenamiento de modelos: {e}")
            return {}
    
    async def predict(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        horizon_hours: int = 1
    ) -> Optional[MLPrediction]:
        """
        Genera predicción usando ensemble de modelos
        """
        try:
            # Cargar modelos si no están en memoria
            if symbol not in self.models:
                self._load_models(symbol)
            
            if symbol not in self.models or not self.models[symbol]:
                logger.warning(f"No hay modelos entrenados para {symbol}")
                return None
            
            # Preparar features
            df_features = self.prepare_features(df)
            
            if len(df_features) == 0:
                logger.warning("No hay datos suficientes para predicción")
                return None
            
            # Usar la última fila para predicción
            available_features = [col for col in self.feature_columns if col in df_features.columns]
            X = df_features[available_features].iloc[-1:].values
            
            # Escalar
            if symbol in self.scalers:
                X_scaled = self.scalers[symbol].transform(X)
            else:
                logger.warning(f"No hay scaler para {symbol}")
                return None
            
            # Predicciones de cada modelo
            predictions = []
            weights = []
            models_used = []
            
            for model_name, model in self.models[symbol].items():
                try:
                    pred = model.predict(X_scaled)[0]
                    weight = self.model_configs[model_name]['weight']
                    
                    predictions.append(pred)
                    weights.append(weight)
                    models_used.append(model_name)
                    
                except Exception as e:
                    logger.warning(f"Error en predicción de {model_name}: {e}")
            
            if not predictions:
                return None
            
            # Ensemble prediction (weighted average)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalizar
            ensemble_prediction = np.average(predictions, weights=weights)
            
            # Calcular confianza basada en concordancia de modelos
            pred_std = np.std(predictions)
            current_price = df['close'].iloc[-1]
            confidence = max(0.1, 1.0 - (pred_std / current_price))
            
            return MLPrediction(
                symbol=symbol,
                prediction=ensemble_prediction,
                confidence=confidence,
                model_name="ensemble",
                features_used=available_features,
                timestamp=datetime.now(),
                horizon_hours=horizon_hours
            )
            
        except Exception as e:
            logger.error(f"Error en predicción ML: {e}")
            return None
    
    def _save_models(self, symbol: str):
        """Guarda modelos entrenados"""
        try:
            symbol_path = self.models_path / symbol.replace('/', '_')
            symbol_path.mkdir(exist_ok=True)
            
            # Guardar modelos
            for model_name, model in self.models[symbol].items():
                model_file = symbol_path / f"{model_name}.joblib"
                joblib.dump(model, model_file)
            
            # Guardar scaler
            if symbol in self.scalers:
                scaler_file = symbol_path / "scaler.joblib"
                joblib.dump(self.scalers[symbol], scaler_file)
            
            logger.info(f"Modelos guardados para {symbol}")
            
        except Exception as e:
            logger.error(f"Error guardando modelos: {e}")
    
    def _load_models(self, symbol: str):
        """Carga modelos entrenados"""
        try:
            symbol_path = self.models_path / symbol.replace('/', '_')
            
            if not symbol_path.exists():
                logger.warning(f"No hay modelos guardados para {symbol}")
                return
            
            models = {}
            
            # Cargar modelos
            for model_name in self.model_configs.keys():
                model_file = symbol_path / f"{model_name}.joblib"
                if model_file.exists():
                    models[model_name] = joblib.load(model_file)
            
            self.models[symbol] = models
            
            # Cargar scaler
            scaler_file = symbol_path / "scaler.joblib"
            if scaler_file.exists():
                self.scalers[symbol] = joblib.load(scaler_file)
            
            logger.info(f"Modelos cargados para {symbol}")
            
        except Exception as e:
            logger.error(f"Error cargando modelos: {e}")
    
    def get_model_info(self, symbol: str) -> Dict[str, Any]:
        """Retorna información sobre los modelos"""
        if symbol not in self.models:
            return {}
        
        info = {
            'symbol': symbol,
            'models_available': list(self.models[symbol].keys()),
            'features_count': len(self.feature_columns),
            'has_scaler': symbol in self.scalers
        }
        
        return info

# Función de testing
async def test_ml_predictor():
    """Función de prueba para el ML predictor"""
    from src.utils.config_manager import ConfigManager
    from src.data_sources.data_aggregator import DataAggregator
    
    config = ConfigManager()
    predictor = MLPredictor(config)
    
    # Simular datos históricos
    dates = pd.date_range(start='2024-01-01', end='2024-08-01', freq='1H')
    np.random.seed(42)
    
    # Crear datos sintéticos que simulan BTC
    price_base = 45000
    prices = []
    volumes = []
    
    for i in range(len(dates)):
        # Simular movimiento de precios con tendencia y ruido
        trend = 0.0001 * i  # Tendencia alcista
        noise = np.random.normal(0, 0.02)  # Volatilidad 2%
        price = price_base * (1 + trend + noise)
        prices.append(price)
        
        # Volume correlacionado con volatilidad
        volume = 1000 + abs(noise) * 50000
        volumes.append(volume)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': volumes
    })
    
    df.set_index('timestamp', inplace=True)
    
    print(f"Datos sintéticos: {len(df)} velas")
    print(f"Precio inicial: ${df['close'].iloc[0]:.2f}")
    print(f"Precio final: ${df['close'].iloc[-1]:.2f}")
    
    # Entrenar modelos
    performances = predictor.train_models(df, "BTC/USDT")
    
    for model_name, perf in performances.items():
        print(f"{model_name}: Accuracy = {perf.accuracy_score:.1f}%")
    
    # Hacer predicción
    prediction = await predictor.predict(df, "BTC/USDT")
    
    if prediction:
        current_price = df['close'].iloc[-1]
        predicted_change = ((prediction.prediction - current_price) / current_price) * 100
        
        print(f"\nPredicción:")
        print(f"Precio actual: ${current_price:.2f}")
        print(f"Precio predicho: ${prediction.prediction:.2f}")
        print(f"Cambio esperado: {predicted_change:.2f}%")
        print(f"Confianza: {prediction.confidence:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ml_predictor())
