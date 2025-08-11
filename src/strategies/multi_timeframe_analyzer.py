# @claude: Implementar analyzer sofisticado que:
# 1. Sincronice datos de múltiples timeframes
# 2. Detecte confluencia de momentum patterns
# 3. Calcule weighted scoring por timeframe
# 4. Implemente regime-aware analysis
# 5. Genere unified confidence score

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TimeframeSignal:
    timeframe: str
    momentum_score: float
    volume_confirmation: bool
    technical_score: float
    ml_prediction: float
    confidence: float
    timestamp: datetime
    support_resistance_levels: Dict[str, float]
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    volatility_regime: str  # 'low', 'medium', 'high'

@dataclass
class ConfluenceResult:
    unified_score: float
    confidence_level: float
    dominant_timeframe: str
    signal_strength: str  # 'weak', 'moderate', 'strong', 'very_strong'
    risk_level: str
    entry_recommendation: bool
    stop_loss_level: float
    take_profit_levels: List[float]
    max_position_size: float

class MultiTimeframeAnalyzer:
    """
    Analyzer sofisticado para confluencia multi-timeframe
    """
    
    def __init__(self, config: dict):
        self.timeframes = config.get('timeframes', ['3m', '5m', '15m'])
        self.weights = config.get('timeframe_weights', {
            '3m': 0.3,   # Short-term momentum
            '5m': 0.4,   # Primary timeframe
            '15m': 0.3   # Trend confirmation
        })
        
        # Configuración de thresholds
        self.momentum_threshold = config.get('momentum_threshold', 0.7)
        self.confluence_threshold = config.get('confluence_threshold', 0.75)
        self.min_volume_ratio = config.get('min_volume_ratio', 1.5)
        
        # Configuración de regime detection
        self.volatility_windows = {
            '3m': 20,
            '5m': 12,
            '15m': 4
        }
        
        # Cache para optimizar performance
        self._signal_cache: Dict[str, TimeframeSignal] = {}
        self._cache_expiry = timedelta(minutes=1)
        
        logger.info(f"MultiTimeframeAnalyzer initialized with timeframes: {self.timeframes}")

    async def analyze_confluence(self, symbol: str, current_price: float) -> ConfluenceResult:
        """
        Analiza confluencia entre todos los timeframes
        """
        try:
            # 1. Obtener señales de todos los timeframes
            timeframe_signals = await self._get_all_timeframe_signals(symbol)
            
            # 2. Detectar regime de mercado
            market_regime = self._detect_market_regime(timeframe_signals)
            
            # 3. Calcular confluencia weighted
            confluence_score = self._calculate_weighted_confluence(timeframe_signals)
            
            # 4. Generar unified confidence score
            unified_confidence = self._generate_unified_confidence(
                timeframe_signals, confluence_score, market_regime
            )
            
            # 5. Determinar dominant timeframe
            dominant_tf = self._find_dominant_timeframe(timeframe_signals)
            
            # 6. Calcular levels de riesgo y recompensa
            risk_reward = self._calculate_risk_reward_levels(
                timeframe_signals, current_price, market_regime
            )
            
            # 7. Generar recomendación final
            result = ConfluenceResult(
                unified_score=confluence_score,
                confidence_level=unified_confidence,
                dominant_timeframe=dominant_tf,
                signal_strength=self._classify_signal_strength(confluence_score, unified_confidence),
                risk_level=market_regime['risk_level'],
                entry_recommendation=self._should_enter(confluence_score, unified_confidence, market_regime),
                stop_loss_level=risk_reward['stop_loss'],
                take_profit_levels=risk_reward['take_profits'],
                max_position_size=self._calculate_position_size(unified_confidence, market_regime)
            )
            
            logger.info(f"Confluence analysis for {symbol}: Score={confluence_score:.3f}, "
                       f"Confidence={unified_confidence:.3f}, Recommendation={result.entry_recommendation}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in confluence analysis for {symbol}: {e}")
            return self._create_neutral_result()

    async def _get_all_timeframe_signals(self, symbol: str) -> Dict[str, TimeframeSignal]:
        """
        Obtiene señales de todos los timeframes de forma asíncrona
        """
        tasks = []
        for tf in self.timeframes:
            task = self._get_timeframe_signal(symbol, tf)
            tasks.append(task)
        
        signals = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for i, signal in enumerate(signals):
            if not isinstance(signal, Exception):
                result[self.timeframes[i]] = signal
            else:
                logger.warning(f"Failed to get signal for {symbol} {self.timeframes[i]}: {signal}")
        
        return result

    async def _get_timeframe_signal(self, symbol: str, timeframe: str) -> TimeframeSignal:
        """
        Obtiene señal individual para un timeframe específico
        """
        cache_key = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
        # Check cache first
        if cache_key in self._signal_cache:
            cached_signal = self._signal_cache[cache_key]
            if datetime.now() - cached_signal.timestamp < self._cache_expiry:
                return cached_signal
        
        # Simular obtención de datos (en implementación real conectaría con data sources)
        data = await self._fetch_timeframe_data(symbol, timeframe)
        
        # Calcular indicadores técnicos
        technical_indicators = self._calculate_technical_indicators(data, timeframe)
        
        # Calcular momentum score
        momentum_score = self._calculate_momentum_score(technical_indicators, timeframe)
        
        # Verificar confirmación de volumen
        volume_confirmation = self._check_volume_confirmation(data, timeframe)
        
        # Obtener predicción ML (simulada)
        ml_prediction = self._get_ml_prediction(technical_indicators, timeframe)
        
        # Calcular confidence basado en consistencia de señales
        confidence = self._calculate_signal_confidence(
            momentum_score, volume_confirmation, technical_indicators
        )
        
        # Detectar levels y regimen
        levels = self._identify_support_resistance(data)
        trend = self._identify_trend_direction(technical_indicators)
        volatility = self._classify_volatility_regime(data, timeframe)
        
        signal = TimeframeSignal(
            timeframe=timeframe,
            momentum_score=momentum_score,
            volume_confirmation=volume_confirmation,
            technical_score=technical_indicators['composite_score'],
            ml_prediction=ml_prediction,
            confidence=confidence,
            timestamp=datetime.now(),
            support_resistance_levels=levels,
            trend_direction=trend,
            volatility_regime=volatility
        )
        
        # Cache the signal
        self._signal_cache[cache_key] = signal
        
        return signal

    def _calculate_weighted_confluence(self, signals: Dict[str, TimeframeSignal]) -> float:
        """
        Calcula confluencia weighted entre timeframes
        """
        if not signals:
            return 0.0
        
        weighted_scores = []
        total_weight = 0
        
        for tf, signal in signals.items():
            if tf in self.weights:
                # Combinar momentum, technical y ML scores
                combined_score = (
                    signal.momentum_score * 0.4 +
                    signal.technical_score * 0.3 +
                    signal.ml_prediction * 0.3
                )
                
                # Aplicar confidence weighting
                confidence_adjusted = combined_score * signal.confidence
                
                # Aplicar timeframe weight
                weighted_score = confidence_adjusted * self.weights[tf]
                weighted_scores.append(weighted_score)
                total_weight += self.weights[tf]
        
        if total_weight == 0:
            return 0.0
        
        confluence_score = sum(weighted_scores) / total_weight
        
        # Bonus por agreement entre timeframes
        agreement_bonus = self._calculate_agreement_bonus(signals)
        
        return min(1.0, confluence_score + agreement_bonus)

    def _calculate_agreement_bonus(self, signals: Dict[str, TimeframeSignal]) -> float:
        """
        Calcula bonus por agreement entre timeframes
        """
        if len(signals) < 2:
            return 0.0
        
        # Verificar si todos los timeframes están alineados
        momentum_scores = [s.momentum_score for s in signals.values()]
        trend_directions = [s.trend_direction for s in signals.values()]
        
        # Agreement en momentum (todos por encima del threshold)
        momentum_agreement = all(score > self.momentum_threshold for score in momentum_scores)
        
        # Agreement en dirección de trend
        trend_agreement = len(set(trend_directions)) == 1 and trend_directions[0] == 'bullish'
        
        # Volume confirmation en timeframe dominante
        volume_agreement = any(s.volume_confirmation for s in signals.values())
        
        bonus = 0.0
        if momentum_agreement:
            bonus += 0.05
        if trend_agreement:
            bonus += 0.05
        if volume_agreement:
            bonus += 0.03
        
        return bonus

    def _generate_unified_confidence(self, signals: Dict[str, TimeframeSignal], 
                                   confluence_score: float, market_regime: Dict) -> float:
        """
        Genera unified confidence score
        """
        if not signals:
            return 0.0
        
        # Base confidence desde confluence
        base_confidence = confluence_score
        
        # Ajustar por regime de mercado
        regime_multiplier = {
            'low': 1.1,      # Baja volatilidad = mayor confianza
            'medium': 1.0,   # Volatilidad normal
            'high': 0.8      # Alta volatilidad = menor confianza
        }.get(market_regime.get('volatility', 'medium'), 1.0)
        
        # Ajustar por consistency entre timeframes
        confidence_scores = [s.confidence for s in signals.values()]
        consistency_factor = 1.0 - (np.std(confidence_scores) * 0.5)  # Penalizar inconsistencia
        
        # Ajustar por número de timeframes confirmando
        confirmation_ratio = len([s for s in signals.values() if s.momentum_score > self.momentum_threshold]) / len(signals)
        confirmation_factor = confirmation_ratio ** 0.5  # Raíz cuadrada para suavizar
        
        unified_confidence = (
            base_confidence * 
            regime_multiplier * 
            consistency_factor * 
            confirmation_factor
        )
        
        return min(1.0, max(0.0, unified_confidence))

    def _detect_market_regime(self, signals: Dict[str, TimeframeSignal]) -> Dict:
        """
        Detecta regime de mercado actual
        """
        if not signals:
            return {'volatility': 'medium', 'risk_level': 'medium'}
        
        # Analizar volatilidad across timeframes
        volatility_regimes = [s.volatility_regime for s in signals.values()]
        volatility_counts = {regime: volatility_regimes.count(regime) for regime in set(volatility_regimes)}
        dominant_volatility = max(volatility_counts.keys(), key=lambda k: volatility_counts[k])
        
        # Analizar trend consistency
        trend_directions = [s.trend_direction for s in signals.values()]
        trend_consistency = len(set(trend_directions)) == 1
        
        # Determinar risk level
        risk_factors = []
        if dominant_volatility == 'high':
            risk_factors.append('high_volatility')
        if not trend_consistency:
            risk_factors.append('trend_divergence')
        
        confidence_scores = [s.confidence for s in signals.values()]
        if np.mean(confidence_scores) < 0.6:
            risk_factors.append('low_confidence')
        
        risk_level = 'high' if len(risk_factors) >= 2 else 'medium' if risk_factors else 'low'
        
        return {
            'volatility': dominant_volatility,
            'trend_consistency': trend_consistency,
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }

    # Métodos helper simulados (en implementación real conectarían con sistemas reales)
    async def _fetch_timeframe_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Simula fetch de datos de timeframe"""
        # En implementación real, aquí iría la conexión a la API del exchange
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq=timeframe.replace('m', 'min')),
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(95, 105, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        })

    def _calculate_technical_indicators(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """Calcula indicadores técnicos"""
        return {
            'rsi': np.random.uniform(30, 70),
            'macd': np.random.uniform(-0.5, 0.5),
            'bb_position': np.random.uniform(0.2, 0.8),
            'composite_score': np.random.uniform(0.4, 0.9)
        }

    def _calculate_momentum_score(self, indicators: Dict, timeframe: str) -> float:
        """Calcula momentum score"""
        return np.random.uniform(0.5, 0.95)

    def _check_volume_confirmation(self, data: pd.DataFrame, timeframe: str) -> bool:
        """Verifica confirmación de volumen"""
        return np.random.choice([True, False], p=[0.7, 0.3])

    def _get_ml_prediction(self, indicators: Dict, timeframe: str) -> float:
        """Obtiene predicción ML"""
        return np.random.uniform(0.4, 0.9)

    def _calculate_signal_confidence(self, momentum: float, volume: bool, indicators: Dict) -> float:
        """Calcula confidence de señal"""
        base_confidence = (momentum + indicators['composite_score']) / 2
        volume_bonus = 0.1 if volume else 0
        return min(1.0, base_confidence + volume_bonus)

    def _identify_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Identifica niveles de soporte y resistencia"""
        return {
            'support': np.random.uniform(95, 100),
            'resistance': np.random.uniform(110, 115)
        }

    def _identify_trend_direction(self, indicators: Dict) -> str:
        """Identifica dirección de trend"""
        return np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.5, 0.2, 0.3])

    def _classify_volatility_regime(self, data: pd.DataFrame, timeframe: str) -> str:
        """Clasifica regime de volatilidad"""
        return np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])

    def _find_dominant_timeframe(self, signals: Dict[str, TimeframeSignal]) -> str:
        """Encuentra timeframe dominante"""
        if not signals:
            return self.timeframes[0]
        
        # El timeframe con mayor confidence score weighted
        weighted_scores = {}
        for tf, signal in signals.items():
            if tf in self.weights:
                weighted_scores[tf] = signal.confidence * self.weights[tf]
        
        return max(weighted_scores.keys(), key=lambda k: weighted_scores[k]) if weighted_scores else self.timeframes[0]

    def _calculate_risk_reward_levels(self, signals: Dict[str, TimeframeSignal], 
                                    current_price: float, market_regime: Dict) -> Dict:
        """Calcula levels de riesgo y recompensa"""
        # Simplified calculation - en implementación real sería más sofisticado
        volatility_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.3}.get(market_regime['volatility'], 1.0)
        
        stop_loss_distance = current_price * 0.02 * volatility_multiplier  # 2% base stop loss
        
        return {
            'stop_loss': current_price - stop_loss_distance,
            'take_profits': [
                current_price * 1.03,  # 3% target
                current_price * 1.06,  # 6% target
                current_price * 1.10   # 10% target
            ]
        }

    def _classify_signal_strength(self, confluence_score: float, confidence: float) -> str:
        """Clasifica strength de señal"""
        combined_score = (confluence_score + confidence) / 2
        
        if combined_score >= 0.85:
            return 'very_strong'
        elif combined_score >= 0.75:
            return 'strong'
        elif combined_score >= 0.60:
            return 'moderate'
        else:
            return 'weak'

    def _should_enter(self, confluence_score: float, confidence: float, market_regime: Dict) -> bool:
        """Determina si se debe entrar en posición"""
        min_threshold = 0.75 if market_regime['risk_level'] == 'high' else 0.65
        return confluence_score >= min_threshold and confidence >= 0.70

    def _calculate_position_size(self, confidence: float, market_regime: Dict) -> float:
        """Calcula tamaño máximo de posición"""
        base_size = 0.05  # 5% base position
        confidence_multiplier = confidence
        risk_multiplier = {'low': 1.2, 'medium': 1.0, 'high': 0.6}.get(market_regime['risk_level'], 1.0)
        
        return base_size * confidence_multiplier * risk_multiplier

    def _create_neutral_result(self) -> ConfluenceResult:
        """Crea resultado neutral en caso de error"""
        return ConfluenceResult(
            unified_score=0.0,
            confidence_level=0.0,
            dominant_timeframe=self.timeframes[0],
            signal_strength='weak',
            risk_level='high',
            entry_recommendation=False,
            stop_loss_level=0.0,
            take_profit_levels=[],
            max_position_size=0.0
        )

    async def get_timeframe_health_status(self) -> Dict[str, Dict]:
        """
        Obtiene status de salud de todos los timeframes
        """
        health_status = {}
        
        for tf in self.timeframes:
            try:
                # Verificar conectividad y calidad de datos
                test_data = await self._fetch_timeframe_data("BTCUSDT", tf)
                data_quality = len(test_data) > 50  # Verificar que hay suficientes datos
                
                health_status[tf] = {
                    'status': 'healthy' if data_quality else 'degraded',
                    'data_points': len(test_data),
                    'last_update': datetime.now(),
                    'weight': self.weights.get(tf, 0)
                }
            except Exception as e:
                health_status[tf] = {
                    'status': 'error',
                    'error': str(e),
                    'last_update': datetime.now(),
                    'weight': self.weights.get(tf, 0)
                }
        
        return health_status
