"""
Sistema de Señales de Trading Intradía con Alertas de Precaución
Incluye precio de entrada óptimo y warnings sobre posibles caídas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

# Imports locales con fallback
try:
    from src.utils.logger import get_logger
    from src.ml_models.price_target_predictor import PriceTargetPredictor, PriceTarget
except ImportError:
    try:
        from utils.logger import get_logger
        from ml_models.price_target_predictor import PriceTargetPredictor, PriceTarget
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

class SignalType(Enum):
    """Tipos de señales de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WATCH = "WATCH"
    AVOID = "AVOID"

class TimeFrame(Enum):
    """Timeframes de trading"""
    SCALPING = "1m-5m"      # 1-5 minutos
    QUICK = "15m-1h"        # 15 minutos - 1 hora
    INTRADAY = "1h-4h"      # 1-4 horas
    SWING = "4h-1d"         # 4 horas - 1 día
    POSITION = "1d+"        # Más de 1 día

class RiskLevel(Enum):
    """Niveles de riesgo"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class TradingAlert:
    """Alerta de trading con detalles"""
    type: str  # "WARNING", "CAUTION", "INFO", "DANGER"
    message: str
    severity: int  # 1-10 (10 = máxima severidad)
    action_required: str
    timeframe: str

@dataclass
class EntryStrategy:
    """Estrategia de entrada"""
    optimal_price: float
    entry_range_min: float
    entry_range_max: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    position_size_recommended: float
    timeframe: TimeFrame
    
@dataclass
class TradingSignal:
    """Señal completa de trading intradía"""
    symbol: str
    signal_type: SignalType
    current_price: float
    timestamp: datetime
    
    # Precios y estrategia
    entry_strategy: EntryStrategy
    price_targets: PriceTarget
    
    # Análisis de riesgo
    risk_level: RiskLevel
    confidence: float
    probability_success: float
    
    # Alertas y precauciones
    alerts: List[TradingAlert]
    precautions: List[str]
    market_conditions: Dict
    
    # Trading intradía
    intraday_signals: Dict
    scalping_opportunities: List[Dict]
    
    # Validación
    is_valid: bool
    expiry_time: datetime
    signal_strength: int  # 1-10

class IntradayTradingSignals:
    """
    Generador de señales de trading intradía con alertas de precaución
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Inicializar predictor de targets
        self.price_predictor = PriceTargetPredictor(config)
        
        # Configuración de trading
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.max_risk_tolerance = self.config.get('max_risk_tolerance', 'MEDIUM')
        self.intraday_timeframes = ['1m', '5m', '15m', '1h', '4h']
        
        # Configuración de alertas
        self.enable_precaution_alerts = self.config.get('enable_precaution_alerts', True)
        self.enable_entry_optimization = self.config.get('enable_entry_optimization', True)
        
        self.logger.info("✅ IntradayTradingSignals inicializado")
    
    def generate_trading_signal(self, symbol: str, current_price: float, 
                              market_data: pd.DataFrame, momentum_score: float) -> TradingSignal:
        """
        Genera señal completa de trading con alertas y precio de entrada
        """
        try:
            # 1. Generar price targets
            price_targets = self.price_predictor.predict_targets(
                symbol, current_price, market_data, momentum_score
            )
            
            # 2. Determinar tipo de señal
            signal_type = self._determine_signal_type(momentum_score, price_targets)
            
            # 3. Calcular precio de entrada óptimo
            entry_strategy = self._calculate_optimal_entry(
                symbol, current_price, market_data, price_targets
            )
            
            # 4. Analizar riesgos
            risk_level = self._analyze_risk_level(market_data, momentum_score, price_targets)
            
            # 5. Generar alertas de precaución
            alerts = self._generate_precaution_alerts(
                symbol, current_price, market_data, momentum_score
            )
            
            # 6. Generar señales intradía
            intraday_signals = self._generate_intraday_signals(
                symbol, market_data, momentum_score
            )
            
            # 7. Detectar oportunidades de scalping
            scalping_opportunities = self._detect_scalping_opportunities(market_data)
            
            # 8. Crear señal completa
            trading_signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                current_price=current_price,
                timestamp=datetime.now(),
                
                entry_strategy=entry_strategy,
                price_targets=price_targets,
                
                risk_level=risk_level,
                confidence=price_targets.confidence,
                probability_success=price_targets.probability_success,
                
                alerts=alerts,
                precautions=self._generate_precautions(market_data, momentum_score),
                market_conditions=self._analyze_market_conditions(market_data),
                
                intraday_signals=intraday_signals,
                scalping_opportunities=scalping_opportunities,
                
                is_valid=self._validate_signal(price_targets, risk_level),
                expiry_time=datetime.now() + timedelta(hours=4),
                signal_strength=self._calculate_signal_strength(momentum_score, price_targets)
            )
            
            self.logger.info(f"🎯 Señal generada para {symbol}: {signal_type.value} "
                           f"(Entrada: ${entry_strategy.optimal_price:.4f})")
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"❌ Error generando señal para {symbol}: {e}")
            return self._create_default_signal(symbol, current_price)
    
    def _determine_signal_type(self, momentum_score: float, price_targets: PriceTarget) -> SignalType:
        """Determina el tipo de señal basado en análisis"""
        if momentum_score > 75 and price_targets.confidence > 0.7:
            return SignalType.BUY
        elif momentum_score > 60 and price_targets.confidence > 0.6:
            return SignalType.WATCH
        elif momentum_score < 30:
            return SignalType.AVOID
        elif momentum_score < 45:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _calculate_optimal_entry(self, symbol: str, current_price: float, 
                               market_data: pd.DataFrame, price_targets: PriceTarget) -> EntryStrategy:
        """Calcula precio de entrada óptimo"""
        try:
            # 1. Análisis de soporte dinámico
            support_levels = self._find_support_levels(market_data)
            nearest_support = max([s for s in support_levels if s < current_price], 
                                default=current_price * 0.98)
            
            # 2. Análisis de retroceso esperado
            volatility = market_data['close'].pct_change().std()
            expected_pullback = volatility * 2  # 2 desviaciones estándar
            
            # 3. Precio óptimo de entrada
            pullback_entry = current_price * (1 - expected_pullback)
            support_entry = nearest_support
            
            # Elegir mejor entrada
            optimal_price = max(pullback_entry, support_entry)
            optimal_price = min(optimal_price, current_price * 0.995)  # Max 0.5% debajo
            
            # 4. Rango de entrada
            entry_range_min = optimal_price * 0.998
            entry_range_max = optimal_price * 1.002
            
            # 5. Stop Loss dinámico
            atr = self._calculate_atr(market_data)
            stop_loss = optimal_price - (atr * 1.5)
            
            # 6. Take Profits escalonados
            tp1 = current_price * (1 + price_targets.conservative_target / 100)
            tp2 = current_price * (1 + price_targets.moderate_target / 100)
            tp3 = current_price * (1 + price_targets.aggressive_target / 100)
            
            # 7. Position size recomendado
            risk_per_trade = 0.02  # 2% del capital
            distance_to_stop = abs(optimal_price - stop_loss)
            position_size = risk_per_trade / (distance_to_stop / optimal_price)
            position_size = min(position_size, 0.1)  # Máximo 10%
            
            # 8. Timeframe recomendado
            if price_targets.momentum_score > 80:
                timeframe = TimeFrame.SCALPING
            elif price_targets.momentum_score > 60:
                timeframe = TimeFrame.QUICK
            else:
                timeframe = TimeFrame.INTRADAY
            
            return EntryStrategy(
                optimal_price=optimal_price,
                entry_range_min=entry_range_min,
                entry_range_max=entry_range_max,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                position_size_recommended=position_size,
                timeframe=timeframe
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculando entrada óptima: {e}")
            return self._create_default_entry_strategy(current_price, price_targets)
    
    def _generate_precaution_alerts(self, symbol: str, current_price: float,
                                  market_data: pd.DataFrame, momentum_score: float) -> List[TradingAlert]:
        """Genera alertas de precaución"""
        alerts = []
        
        try:
            # 1. Alerta de volatilidad extrema
            volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            if volatility > 0.05:  # 5% diario
                alerts.append(TradingAlert(
                    type="WARNING",
                    message=f"⚠️ VOLATILIDAD EXTREMA: {volatility*100:.1f}% - Posible caída brusca antes de subir",
                    severity=8,
                    action_required="Reducir tamaño de posición y usar stop loss ajustado",
                    timeframe="Próximas 1-2 horas"
                ))
            
            # 2. Alerta de volumen anómalo
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
            if current_volume > avg_volume * 3:
                alerts.append(TradingAlert(
                    type="CAUTION",
                    message=f"📊 VOLUMEN ANÓMALO: {current_volume/avg_volume:.1f}x promedio - Observar movimiento",
                    severity=6,
                    action_required="Monitorear de cerca los próximos movimientos",
                    timeframe="Próximos 15-30 minutos"
                ))
            
            # 3. Alerta de RSI extremo
            rsi = self._calculate_rsi(market_data['close'])
            if rsi > 80:
                alerts.append(TradingAlert(
                    type="WARNING",
                    message=f"📈 RSI SOBRECOMPRADO: {rsi:.1f} - Probable corrección antes de continuar",
                    severity=7,
                    action_required="Esperar retroceso para entrada o reducir exposición",
                    timeframe="Próximas horas"
                ))
            elif rsi < 20:
                alerts.append(TradingAlert(
                    type="INFO",
                    message=f"📉 RSI SOBREVENDIDO: {rsi:.1f} - Posible rebote técnico",
                    severity=4,
                    action_required="Considerar entrada en rebote",
                    timeframe="Próximas horas"
                ))
            
            # 4. Alerta de momentum falso
            if momentum_score > 70:
                recent_changes = market_data['close'].pct_change().tail(5)
                if recent_changes.std() > 0.03:  # Alta volatilidad reciente
                    alerts.append(TradingAlert(
                        type="CAUTION",
                        message="🎭 POSIBLE MOMENTUM FALSO: Alta volatilidad reciente puede indicar trampa",
                        severity=6,
                        action_required="Confirmar señal con múltiples timeframes",
                        timeframe="Próximos 30 minutos"
                    ))
            
            # 5. Alerta de resistencia cercana
            resistance_levels = self._find_resistance_levels(market_data)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], 
                                   default=current_price * 1.1)
            
            distance_to_resistance = (nearest_resistance - current_price) / current_price
            if distance_to_resistance < 0.02:  # Menos de 2%
                alerts.append(TradingAlert(
                    type="WARNING",
                    message=f"🚧 RESISTENCIA CERCANA: ${nearest_resistance:.4f} ({distance_to_resistance*100:.1f}%)",
                    severity=7,
                    action_required="Considerar tomar parciales cerca de resistencia",
                    timeframe="Inmediato"
                ))
            
            # 6. Alerta de condiciones de mercado
            market_trend = self._analyze_market_trend(market_data)
            if market_trend == "BEARISH" and momentum_score > 60:
                alerts.append(TradingAlert(
                    type="CAUTION",
                    message="🐻 MERCADO BAJISTA: Momentum positivo en tendencia bajista - Mayor riesgo",
                    severity=6,
                    action_required="Reducir timeframe y ser más conservador",
                    timeframe="Sesión actual"
                ))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generando alertas: {e}")
        
        return alerts
    
    def _generate_intraday_signals(self, symbol: str, market_data: pd.DataFrame, 
                                 momentum_score: float) -> Dict:
        """Genera señales específicas para trading intradía"""
        signals = {}
        
        try:
            # 1. Señales de scalping (1-5 minutos)
            if momentum_score > 70:
                signals['scalping'] = {
                    'active': True,
                    'timeframe': '1m-5m',
                    'strategy': 'Momentum breakout',
                    'entry_triggers': ['Volume spike', 'Price acceleration'],
                    'exit_targets': ['+0.5%', '+1.0%', '+1.5%'],
                    'max_hold_time': '15 minutes'
                }
            
            # 2. Señales de swing intradía (15 minutos - 1 hora)
            signals['swing_intraday'] = {
                'active': momentum_score > 50,
                'timeframe': '15m-1h',
                'strategy': 'Support/Resistance bounce',
                'entry_triggers': ['Support bounce', 'Breakout confirmation'],
                'exit_targets': ['+1.5%', '+3.0%', '+5.0%'],
                'max_hold_time': '2-4 hours'
            }
            
            # 3. Señales de position trading (4 horas - 1 día)
            signals['position_intraday'] = {
                'active': momentum_score > 40,
                'timeframe': '4h-1d',
                'strategy': 'Trend following',
                'entry_triggers': ['Trend confirmation', 'Volume confirmation'],
                'exit_targets': ['+3.0%', '+6.0%', '+10.0%'],
                'max_hold_time': '1-3 days'
            }
            
            # 4. Alertas de timing
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 11:  # Horario de alta volatilidad
                signals['timing_alert'] = "🕘 HORARIO DE ALTA ACTIVIDAD - Mayor volatilidad esperada"
            elif 21 <= current_hour <= 23:  # Horario asiático
                signals['timing_alert'] = "🕘 HORARIO ASIÁTICO - Volumen reducido en pares occidentales"
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generando señales intradía: {e}")
        
        return signals
    
    def _detect_scalping_opportunities(self, market_data: pd.DataFrame) -> List[Dict]:
        """Detecta oportunidades específicas de scalping"""
        opportunities = []
        
        try:
            # 1. Breakout de microcanal
            recent_high = market_data['high'].tail(10).max()
            recent_low = market_data['low'].tail(10).min()
            current_price = market_data['close'].iloc[-1]
            
            channel_width = (recent_high - recent_low) / recent_low
            if channel_width < 0.005:  # Canal muy estrecho (0.5%)
                opportunities.append({
                    'type': 'Microcanal Breakout',
                    'description': f'Canal estrecho de {channel_width*100:.2f}% - Probable breakout',
                    'entry_above': recent_high,
                    'entry_below': recent_low,
                    'expected_move': channel_width * 2,
                    'timeframe': '1-5 minutes'
                })
            
            # 2. Squeeze de Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(market_data['close'])
            bb_width = (bb_upper - bb_lower) / bb_lower
            if bb_width < 0.01:  # Bands muy estrechas
                opportunities.append({
                    'type': 'Bollinger Squeeze',
                    'description': f'Bandas comprimidas {bb_width*100:.2f}% - Explosión inminente',
                    'entry_above': bb_upper,
                    'entry_below': bb_lower,
                    'expected_move': bb_width * 3,
                    'timeframe': '5-15 minutes'
                })
            
            # 3. Divergencia de momentum
            prices = market_data['close'].tail(20)
            volumes = market_data['volume'].tail(20)
            
            if len(prices) >= 20:
                price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
                volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
                
                if price_trend > 0 and volume_trend < 0:  # Precio sube, volumen baja
                    opportunities.append({
                        'type': 'Divergencia Bajista',
                        'description': 'Precio sube pero volumen disminuye - Debilitamiento',
                        'action': 'Preparar para reversión',
                        'expected_move': 'Corrección 1-2%',
                        'timeframe': '10-30 minutes'
                    })
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error detectando scalping: {e}")
        
        return opportunities
    
    def _generate_precautions(self, market_data: pd.DataFrame, momentum_score: float) -> List[str]:
        """Genera lista de precauciones específicas"""
        precautions = []
        
        try:
            # Precauciones generales
            if momentum_score > 80:
                precautions.append("⚠️ Momentum extremo: Posible corrección técnica antes de continuar subida")
                precautions.append("📊 Considerar tomar parciales en niveles de resistencia")
            
            if momentum_score < 40:
                precautions.append("⚠️ Momentum débil: Confirmar señal con análisis adicional")
                precautions.append("🔍 Observar reacción en soportes clave antes de entrar")
            
            # Precauciones de volatilidad
            volatility = market_data['close'].pct_change().std()
            if volatility > 0.03:
                precautions.append("🌊 Alta volatilidad: Usar stop loss más ajustado")
                precautions.append("⏰ Movimientos pueden ser bruscos - Monitorear de cerca")
            
            # Precauciones de volumen
            volume_ratio = market_data['volume'].iloc[-1] / market_data['volume'].mean()
            if volume_ratio > 2.5:
                precautions.append("📈 Volumen anómalo: Posible manipulación o noticia importante")
                precautions.append("🔍 Verificar noticias antes de tomar posición grande")
            
            # Precauciones de timing
            current_hour = datetime.now().hour
            if current_hour < 6 or current_hour > 22:
                precautions.append("🕐 Horario de bajo volumen: Spreads más amplios y mayor volatilidad")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generando precauciones: {e}")
        
        return precautions
    
    # Métodos auxiliares
    def _analyze_risk_level(self, market_data: pd.DataFrame, momentum_score: float, 
                          price_targets: PriceTarget) -> RiskLevel:
        """Analiza nivel de riesgo"""
        risk_factors = 0
        
        # Factor momentum
        if momentum_score < 40:
            risk_factors += 2
        elif momentum_score > 80:
            risk_factors += 1
        
        # Factor confianza
        if price_targets.confidence < 0.5:
            risk_factors += 2
        elif price_targets.confidence < 0.7:
            risk_factors += 1
        
        # Factor volatilidad
        volatility = market_data['close'].pct_change().std()
        if volatility > 0.05:
            risk_factors += 2
        elif volatility > 0.03:
            risk_factors += 1
        
        # Determinar nivel
        if risk_factors >= 5:
            return RiskLevel.EXTREME
        elif risk_factors >= 4:
            return RiskLevel.HIGH
        elif risk_factors >= 2:
            return RiskLevel.MEDIUM
        elif risk_factors >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calcula Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else data['close'].iloc[-1] * 0.02
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Encuentra niveles de soporte"""
        lows = data['low'].rolling(window=5, center=True).min()
        support_levels = []
        
        for i in range(2, len(lows) - 2):
            if (lows.iloc[i] == data['low'].iloc[i] and 
                lows.iloc[i] < lows.iloc[i-1] and 
                lows.iloc[i] < lows.iloc[i+1]):
                support_levels.append(lows.iloc[i])
        
        return sorted(list(set(support_levels)))[-5:]
    
    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Encuentra niveles de resistencia"""
        highs = data['high'].rolling(window=5, center=True).max()
        resistance_levels = []
        
        for i in range(2, len(highs) - 2):
            if (highs.iloc[i] == data['high'].iloc[i] and 
                highs.iloc[i] > highs.iloc[i-1] and 
                highs.iloc[i] > highs.iloc[i+1]):
                resistance_levels.append(highs.iloc[i])
        
        return sorted(list(set(resistance_levels)))[-5:]
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[float, float]:
        """Calcula Bollinger Bands"""
        sma = prices.rolling(period).mean().iloc[-1]
        std_dev = prices.rolling(period).std().iloc[-1]
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def _analyze_market_trend(self, data: pd.DataFrame) -> str:
        """Analiza tendencia general del mercado"""
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
        current_price = data['close'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            return "BULLISH"
        elif current_price < sma_20 < sma_50:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _analyze_market_conditions(self, data: pd.DataFrame) -> Dict:
        """Analiza condiciones generales del mercado"""
        return {
            'trend': self._analyze_market_trend(data),
            'volatility': data['close'].pct_change().std(),
            'volume_trend': 'HIGH' if data['volume'].iloc[-1] > data['volume'].mean() * 1.5 else 'NORMAL',
            'rsi': self._calculate_rsi(data['close']),
            'market_hours': 'ACTIVE' if 9 <= datetime.now().hour <= 17 else 'QUIET'
        }
    
    def _validate_signal(self, price_targets: PriceTarget, risk_level: RiskLevel) -> bool:
        """Valida si la señal es confiable"""
        return (price_targets.confidence > self.min_confidence and 
                risk_level.value != 'EXTREME')
    
    def _calculate_signal_strength(self, momentum_score: float, price_targets: PriceTarget) -> int:
        """Calcula fuerza de la señal (1-10)"""
        strength = 0
        
        # Factor momentum
        strength += min(4, momentum_score / 20)
        
        # Factor confianza
        strength += min(3, price_targets.confidence * 3)
        
        # Factor targets
        if price_targets.aggressive_target > 5:
            strength += 2
        elif price_targets.moderate_target > 3:
            strength += 1
        
        # Factor probabilidad
        strength += min(1, price_targets.probability_success)
        
        return min(10, max(1, int(strength)))
    
    def _create_default_signal(self, symbol: str, current_price: float) -> TradingSignal:
        """Crea señal por defecto en caso de error"""
        from ml_models.price_target_predictor import PriceTarget
        
        default_target = PriceTarget(
            symbol=symbol, current_price=current_price,
            conservative_target=1.0, moderate_target=2.0, aggressive_target=3.0,
            conservative_price=current_price*1.01, moderate_price=current_price*1.02, 
            aggressive_price=current_price*1.03,
            confidence=0.3, momentum_score=50, probability_success=0.4,
            timeframe_days=7, risk_level="HIGH",
            technical_factors={}, fundamental_factors={}, timestamp=datetime.now()
        )
        
        default_entry = EntryStrategy(
            optimal_price=current_price*0.995, entry_range_min=current_price*0.994,
            entry_range_max=current_price*0.996, stop_loss=current_price*0.97,
            take_profit_1=current_price*1.01, take_profit_2=current_price*1.02,
            take_profit_3=current_price*1.03, position_size_recommended=0.02,
            timeframe=TimeFrame.INTRADAY
        )
        
        return TradingSignal(
            symbol=symbol, signal_type=SignalType.HOLD, current_price=current_price,
            timestamp=datetime.now(), entry_strategy=default_entry, price_targets=default_target,
            risk_level=RiskLevel.HIGH, confidence=0.3, probability_success=0.4,
            alerts=[], precautions=["⚠️ Señal de emergencia - Datos insuficientes"],
            market_conditions={}, intraday_signals={}, scalping_opportunities=[],
            is_valid=False, expiry_time=datetime.now() + timedelta(hours=1),
            signal_strength=3
        )
    
    def _create_default_entry_strategy(self, current_price: float, price_targets: PriceTarget) -> EntryStrategy:
        """Crea estrategia de entrada por defecto"""
        return EntryStrategy(
            optimal_price=current_price * 0.995,
            entry_range_min=current_price * 0.994,
            entry_range_max=current_price * 0.996,
            stop_loss=current_price * 0.97,
            take_profit_1=current_price * (1 + price_targets.conservative_target / 100),
            take_profit_2=current_price * (1 + price_targets.moderate_target / 100),
            take_profit_3=current_price * (1 + price_targets.aggressive_target / 100),
            position_size_recommended=0.02,
            timeframe=TimeFrame.INTRADAY
        )
