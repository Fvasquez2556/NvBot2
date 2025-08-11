"""
Predictor de Price Targets basado en Momentum
Calcula potencial de subida/bajada para cada par analizado
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# Imports locales con fallback
try:
    from src.utils.logger import get_logger
except ImportError:
    try:
        from utils.logger import get_logger
    except ImportError:
        def get_logger(name):
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            return logger

logger = get_logger(__name__)

@dataclass
class PriceTarget:
    """Estructura para targets de precio"""
    symbol: str
    current_price: float
    
    # Targets en %
    conservative_target: float
    moderate_target: float
    aggressive_target: float
    
    # Targets en precio absoluto
    conservative_price: float
    moderate_price: float
    aggressive_price: float
    
    # Métricas de confianza
    confidence: float
    momentum_score: float
    probability_success: float
    
    # Timeframes
    timeframe_days: int
    risk_level: str
    
    # Análisis técnico
    technical_factors: Dict
    fundamental_factors: Dict
    
    timestamp: datetime

class PriceTargetPredictor:
    """
    Predictor avanzado de price targets basado en momentum
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Configuración de predicción
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.max_timeframe_days = self.config.get('max_timeframe_days', 30)
        self.conservative_multiplier = self.config.get('conservative_multiplier', 0.5)
        self.aggressive_multiplier = self.config.get('aggressive_multiplier', 1.8)
        
        # Cache de datos históricos
        self.historical_moves = {}
        self.fibonacci_levels = {}
        self.support_resistance = {}
        
        self.logger.info("✅ PriceTargetPredictor inicializado")
    
    def predict_targets(self, symbol: str, current_price: float, 
                       market_data: pd.DataFrame, momentum_score: float) -> PriceTarget:
        """
        Predice targets de precio basado en múltiples factores
        """
        try:
            # 1. Análisis técnico
            technical_targets = self._calculate_technical_targets(symbol, current_price, market_data)
            
            # 2. Análisis de momentum
            momentum_targets = self._calculate_momentum_targets(symbol, momentum_score, market_data)
            
            # 3. Análisis histórico
            historical_targets = self._calculate_historical_targets(symbol, market_data)
            
            # 4. Machine Learning prediction
            ml_targets = self._calculate_ml_targets(symbol, market_data, momentum_score)
            
            # 5. Combinar todos los análisis
            combined_targets = self._combine_target_analysis(
                technical_targets, momentum_targets, historical_targets, ml_targets
            )
            
            # 6. Calcular confianza global
            confidence = self._calculate_overall_confidence(
                technical_targets, momentum_targets, historical_targets, ml_targets
            )
            
            # 7. Crear objeto PriceTarget
            price_target = PriceTarget(
                symbol=symbol,
                current_price=current_price,
                
                # Targets en %
                conservative_target=combined_targets['conservative'],
                moderate_target=combined_targets['moderate'],
                aggressive_target=combined_targets['aggressive'],
                
                # Targets en precio
                conservative_price=current_price * (1 + combined_targets['conservative'] / 100),
                moderate_price=current_price * (1 + combined_targets['moderate'] / 100),
                aggressive_price=current_price * (1 + combined_targets['aggressive'] / 100),
                
                # Métricas
                confidence=confidence,
                momentum_score=momentum_score,
                probability_success=self._calculate_success_probability(momentum_score, confidence),
                
                # Timeframe
                timeframe_days=self._estimate_timeframe(momentum_score),
                risk_level=self._classify_risk_level(confidence, momentum_score),
                
                # Factores de análisis
                technical_factors=technical_targets,
                fundamental_factors={'momentum_score': momentum_score, 'volume_surge': 0},
                
                timestamp=datetime.now()
            )
            
            self.logger.info(f"🎯 Targets calculados para {symbol}: "
                           f"Conservador: {combined_targets['conservative']:.1f}%, "
                           f"Moderado: {combined_targets['moderate']:.1f}%, "
                           f"Agresivo: {combined_targets['aggressive']:.1f}%")
            
            return price_target
            
        except Exception as e:
            self.logger.error(f"❌ Error calculando targets para {symbol}: {e}")
            return self._create_default_target(symbol, current_price, momentum_score)
    
    def _calculate_technical_targets(self, symbol: str, current_price: float, 
                                   data: pd.DataFrame) -> Dict:
        """Calcula targets basados en análisis técnico"""
        try:
            if len(data) < 20:
                return {'conservative': 2.0, 'moderate': 5.0, 'aggressive': 8.0, 'confidence': 0.3}
            
            # 1. Fibonacci Retracements
            high_20 = data['high'].rolling(20).max().iloc[-1]
            low_20 = data['low'].rolling(20).min().iloc[-1]
            fib_range = high_20 - low_20
            
            fib_targets = {
                '23.6%': current_price + (fib_range * 0.236),
                '38.2%': current_price + (fib_range * 0.382),
                '61.8%': current_price + (fib_range * 0.618),
                '100%': current_price + fib_range
            }
            
            # 2. Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma = data['close'].rolling(bb_period).mean().iloc[-1]
            std = data['close'].rolling(bb_period).std().iloc[-1]
            bb_upper = sma + (bb_std * std)
            
            # 3. Resistance levels
            resistance_levels = self._find_resistance_levels(data)
            next_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            
            # 4. Volume Profile (simplified)
            volume_weighted_price = (data['volume'] * data['close']).sum() / data['volume'].sum()
            
            # Calcular targets técnicos
            technical_conservative = min(
                ((fib_targets['23.6%'] - current_price) / current_price) * 100,
                ((next_resistance - current_price) / current_price) * 100
            )
            
            technical_moderate = ((fib_targets['38.2%'] - current_price) / current_price) * 100
            
            technical_aggressive = max(
                ((bb_upper - current_price) / current_price) * 100,
                ((fib_targets['61.8%'] - current_price) / current_price) * 100
            )
            
            return {
                'conservative': max(1.0, technical_conservative),
                'moderate': max(2.0, technical_moderate),
                'aggressive': max(4.0, technical_aggressive),
                'confidence': 0.7,
                'fibonacci_levels': fib_targets,
                'resistance_levels': resistance_levels,
                'bollinger_upper': bb_upper
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error en análisis técnico para {symbol}: {e}")
            return {'conservative': 2.0, 'moderate': 5.0, 'aggressive': 8.0, 'confidence': 0.3}
    
    def _calculate_momentum_targets(self, symbol: str, momentum_score: float, 
                                  data: pd.DataFrame) -> Dict:
        """Calcula targets basados en fuerza del momentum"""
        try:
            # 1. RSI momentum
            rsi = self._calculate_rsi(data['close'])
            rsi_strength = (rsi - 50) / 50  # Normalizado -1 a 1
            
            # 2. Volume momentum
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            volume_surge = (current_volume / avg_volume) if avg_volume > 0 else 1
            
            # 3. Price acceleration
            price_changes = data['close'].pct_change()
            acceleration = price_changes.rolling(5).mean().iloc[-1]
            
            # 4. Combined momentum factor
            momentum_factor = (momentum_score / 100) * (1 + volume_surge * 0.3) * (1 + acceleration * 10)
            momentum_factor = max(0.1, min(3.0, momentum_factor))  # Clamp entre 0.1 y 3.0
            
            # 5. Historical momentum moves
            historical_moves = self._get_historical_momentum_moves(symbol, momentum_score)
            
            # Calcular targets de momentum
            base_move = historical_moves.get('median_move', 3.0)
            
            momentum_targets = {
                'conservative': base_move * 0.4 * momentum_factor,
                'moderate': base_move * 0.7 * momentum_factor,
                'aggressive': base_move * 1.2 * momentum_factor,
                'confidence': min(0.9, momentum_score / 100),
                'momentum_factor': momentum_factor,
                'volume_surge': volume_surge,
                'rsi_strength': rsi_strength
            }
            
            return momentum_targets
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error en análisis de momentum para {symbol}: {e}")
            return {'conservative': 1.5, 'moderate': 4.0, 'aggressive': 7.0, 'confidence': 0.4}
    
    def _calculate_historical_targets(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Calcula targets basados en movimientos históricos similares"""
        try:
            # Calcular movimientos históricos
            price_changes = data['close'].pct_change()
            
            # Movimientos de diferentes períodos
            moves_1d = price_changes.rolling(1).sum() * 100
            moves_3d = price_changes.rolling(3).sum() * 100
            moves_7d = price_changes.rolling(7).sum() * 100
            
            # Percentiles de movimientos positivos
            positive_moves_1d = moves_1d[moves_1d > 0]
            positive_moves_3d = moves_3d[moves_3d > 0]
            positive_moves_7d = moves_7d[moves_7d > 0]
            
            if len(positive_moves_1d) == 0:
                return {'conservative': 1.0, 'moderate': 3.0, 'aggressive': 6.0, 'confidence': 0.2}
            
            # Calcular percentiles
            p25_1d = positive_moves_1d.quantile(0.25)
            p50_1d = positive_moves_1d.quantile(0.50)
            p75_1d = positive_moves_1d.quantile(0.75)
            
            p25_3d = positive_moves_3d.quantile(0.25) if len(positive_moves_3d) > 0 else p25_1d * 2
            p50_3d = positive_moves_3d.quantile(0.50) if len(positive_moves_3d) > 0 else p50_1d * 2
            p75_3d = positive_moves_3d.quantile(0.75) if len(positive_moves_3d) > 0 else p75_1d * 2
            
            return {
                'conservative': max(0.5, min(p25_1d, p25_3d)),
                'moderate': max(1.0, min(p50_1d, p50_3d)),
                'aggressive': max(2.0, min(p75_1d, p75_3d)),
                'confidence': 0.6,
                'historical_moves': {
                    '1d': {'p25': p25_1d, 'p50': p50_1d, 'p75': p75_1d},
                    '3d': {'p25': p25_3d, 'p50': p50_3d, 'p75': p75_3d}
                }
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error en análisis histórico para {symbol}: {e}")
            return {'conservative': 1.0, 'moderate': 3.0, 'aggressive': 6.0, 'confidence': 0.2}
    
    def _calculate_ml_targets(self, symbol: str, data: pd.DataFrame, 
                            momentum_score: float) -> Dict:
        """Calcula targets usando machine learning (simulado por ahora)"""
        try:
            # Features para ML
            features = self._extract_ml_features(data, momentum_score)
            
            # Simulación de modelo ML (en producción sería un modelo real)
            base_prediction = features['momentum_score'] * 0.08  # 8% por 100 de momentum
            volatility_adjustment = features['volatility'] * 2  # Ajuste por volatilidad
            volume_boost = features['volume_factor'] * 0.02  # Boost por volumen
            
            ml_prediction = base_prediction + volatility_adjustment + volume_boost
            
            # Generar escenarios
            confidence = min(0.8, features['momentum_score'] / 80)
            
            return {
                'conservative': ml_prediction * 0.6,
                'moderate': ml_prediction,
                'aggressive': ml_prediction * 1.5,
                'confidence': confidence,
                'ml_features': features,
                'base_prediction': ml_prediction
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error en predicción ML para {symbol}: {e}")
            return {'conservative': 1.5, 'moderate': 4.0, 'aggressive': 7.5, 'confidence': 0.3}
    
    def _combine_target_analysis(self, technical: Dict, momentum: Dict, 
                               historical: Dict, ml: Dict) -> Dict:
        """Combina todos los análisis en targets finales"""
        # Pesos para cada análisis
        weights = {
            'technical': 0.25,
            'momentum': 0.35,
            'historical': 0.25,
            'ml': 0.15
        }
        
        # Combinar targets
        conservative = (
            technical['conservative'] * weights['technical'] +
            momentum['conservative'] * weights['momentum'] +
            historical['conservative'] * weights['historical'] +
            ml['conservative'] * weights['ml']
        )
        
        moderate = (
            technical['moderate'] * weights['technical'] +
            momentum['moderate'] * weights['momentum'] +
            historical['moderate'] * weights['historical'] +
            ml['moderate'] * weights['ml']
        )
        
        aggressive = (
            technical['aggressive'] * weights['technical'] +
            momentum['aggressive'] * weights['momentum'] +
            historical['aggressive'] * weights['historical'] +
            ml['aggressive'] * weights['ml']
        )
        
        # Asegurar orden lógico
        conservative = max(0.5, conservative)
        moderate = max(conservative + 0.5, moderate)
        aggressive = max(moderate + 1.0, aggressive)
        
        return {
            'conservative': conservative,
            'moderate': moderate,
            'aggressive': aggressive
        }
    
    def _calculate_overall_confidence(self, technical: Dict, momentum: Dict,
                                    historical: Dict, ml: Dict) -> float:
        """Calcula confianza global del análisis"""
        confidences = [
            technical.get('confidence', 0.5),
            momentum.get('confidence', 0.5),
            historical.get('confidence', 0.5),
            ml.get('confidence', 0.5)
        ]
        
        # Promedio ponderado
        weights = [0.25, 0.35, 0.25, 0.15]
        overall_confidence = sum(c * w for c, w in zip(confidences, weights))
        
        return min(0.95, max(0.1, overall_confidence))
    
    def _calculate_success_probability(self, momentum_score: float, confidence: float) -> float:
        """Calcula probabilidad de éxito del target"""
        # Base probability en momentum
        base_prob = min(0.8, momentum_score / 100)
        
        # Ajuste por confianza
        confidence_boost = confidence * 0.3
        
        # Probability final
        probability = base_prob + confidence_boost
        
        return min(0.9, max(0.1, probability))
    
    def _estimate_timeframe(self, momentum_score: float) -> int:
        """Estima timeframe para alcanzar targets"""
        if momentum_score > 80:
            return 1  # 1 día para momentum muy alto
        elif momentum_score > 60:
            return 3  # 3 días para momentum alto
        elif momentum_score > 40:
            return 7  # 1 semana para momentum medio
        else:
            return 14  # 2 semanas para momentum bajo
    
    def _classify_risk_level(self, confidence: float, momentum_score: float) -> str:
        """Clasifica nivel de riesgo"""
        if confidence > 0.7 and momentum_score > 70:
            return "LOW"
        elif confidence > 0.5 and momentum_score > 50:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _create_default_target(self, symbol: str, current_price: float,
                             momentum_score: float) -> PriceTarget:
        """Crea target por defecto en caso de error"""
        return PriceTarget(
            symbol=symbol,
            current_price=current_price,
            conservative_target=1.0,
            moderate_target=3.0,
            aggressive_target=6.0,
            conservative_price=current_price * 1.01,
            moderate_price=current_price * 1.03,
            aggressive_price=current_price * 1.06,
            confidence=0.3,
            momentum_score=momentum_score,
            probability_success=0.4,
            timeframe_days=7,
            risk_level="HIGH",
            technical_factors={},
            fundamental_factors={},
            timestamp=datetime.now()
        )
    
    # Métodos auxiliares
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Encuentra niveles de resistencia"""
        highs = data['high'].rolling(window=5, center=True).max()
        resistance_levels = []
        
        for i in range(2, len(highs) - 2):
            if (highs.iloc[i] == data['high'].iloc[i] and 
                highs.iloc[i] > highs.iloc[i-1] and 
                highs.iloc[i] > highs.iloc[i+1]):
                resistance_levels.append(highs.iloc[i])
        
        return sorted(list(set(resistance_levels)))[-5:]  # Últimos 5 niveles
    
    def _get_historical_momentum_moves(self, symbol: str, momentum_score: float) -> Dict:
        """Obtiene movimientos históricos por momentum"""
        # Simulación - en producción sería una base de datos
        return {
            'median_move': 3.5 + (momentum_score / 100) * 2,
            'avg_move': 4.0 + (momentum_score / 100) * 2.5,
            'max_move': 8.0 + (momentum_score / 100) * 4
        }
    
    def _extract_ml_features(self, data: pd.DataFrame, momentum_score: float) -> Dict:
        """Extrae features para ML"""
        return {
            'momentum_score': momentum_score,
            'volatility': data['close'].pct_change().std() * 100,
            'volume_factor': data['volume'].iloc[-1] / data['volume'].mean(),
            'price_trend': data['close'].pct_change(5).iloc[-1] * 100,
            'rsi': self._calculate_rsi(data['close'])
        }

# Función de prueba
async def test_price_target_predictor():
    """Prueba el predictor de price targets"""
    predictor = PriceTargetPredictor()
    
    # Datos de prueba
    test_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    target = predictor.predict_targets('BTCUSDT', 45000, test_data, 75)
    
    print(f"🎯 Targets para BTCUSDT:")
    print(f"Conservador: {target.conservative_target:.1f}% (${target.conservative_price:.0f})")
    print(f"Moderado: {target.moderate_target:.1f}% (${target.moderate_price:.0f})")
    print(f"Agresivo: {target.aggressive_target:.1f}% (${target.aggressive_price:.0f})")
    print(f"Confianza: {target.confidence:.1%}")
    print(f"Probabilidad éxito: {target.probability_success:.1%}")
    print(f"Timeframe: {target.timeframe_days} días")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_price_target_predictor())
