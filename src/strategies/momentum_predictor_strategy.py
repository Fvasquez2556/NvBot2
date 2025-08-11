# @claude: Crear estrategia maestra que integre:
# 1. Detección de momentum (Fase 2)
# 2. Predicción ML (Fase 3) 
# 3. Análisis multi-timeframe (Fase 4)
# 4. Risk management dinámico
# 5. Position sizing basado en confianza
# 6. Stop-loss y take-profit adaptativos

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass

# Imports de componentes internos
from src.strategies.base_strategy import BaseStrategy
from src.strategies.momentum_detector import MomentumDetector
from src.strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer, ConfluenceResult
from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class TradeSignal:
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    position_size: float
    reasoning: str
    timestamp: datetime
    risk_level: str
    expected_return: float
    max_holding_time: timedelta

@dataclass
class RiskMetrics:
    portfolio_risk: float
    symbol_risk: float
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    total_risk_score: float

class MomentumPredictorStrategy(BaseStrategy):
    """
    Estrategia maestra que integra todos los componentes del sistema
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config or {})
        
        # Configuración principal
        self.config = ConfigManager().get_strategy_config('momentum_predictor')
        
        # Inicializar componentes principales
        self.momentum_detector = MomentumDetector(self.config.get('momentum', {}))
        self.multi_tf_analyzer = MultiTimeframeAnalyzer(self.config.get('multi_timeframe', {}))
        
        # Configuración de risk management
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.15)  # 15% max drawdown
        self.max_position_risk = self.config.get('max_position_risk', 0.05)   # 5% per position
        self.max_correlation = self.config.get('max_correlation', 0.7)        # Max 70% correlation
        self.max_positions = self.config.get('max_positions', 8)              # Max 8 positions
        
        # Configuración de timing
        self.min_confidence = self.config.get('min_confidence', 0.75)
        self.min_confluence_score = self.config.get('min_confluence_score', 0.70)
        self.max_holding_hours = self.config.get('max_holding_hours', 48)
        
        # Estado interno
        self.current_positions: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        logger.info("MomentumPredictorStrategy initialized successfully")

    async def analyze_symbol(self, symbol: str, current_price: float) -> Optional[TradeSignal]:
        """
        Análisis completo de un símbolo para generar señal de trading
        """
        try:
            # 1. Reset daily counter si es necesario
            self._reset_daily_counters()
            
            # 2. Verificar si podemos operar este símbolo
            if not self._can_trade_symbol(symbol):
                return None
            
            # 3. Análisis de momentum base
            momentum_result = await self.momentum_detector.detect_momentum(symbol, current_price)
            if not momentum_result or momentum_result.confidence < 0.6:
                logger.debug(f"Momentum insufficient for {symbol}: {momentum_result.confidence if momentum_result else 'None'}")
                return None
            
            # 4. Análisis multi-timeframe
            confluence_result = await self.multi_tf_analyzer.analyze_confluence(symbol, current_price)
            if confluence_result.unified_score < self.min_confluence_score:
                logger.debug(f"Confluence insufficient for {symbol}: {confluence_result.unified_score}")
                return None
            
            # 5. Evaluación de riesgo integral
            risk_metrics = await self._evaluate_risk(symbol, confluence_result)
            if risk_metrics.total_risk_score > 0.8:  # Too risky
                logger.warning(f"High risk detected for {symbol}: {risk_metrics.total_risk_score}")
                return None
            
            # 6. Calcular position sizing dinámico
            position_size = self._calculate_dynamic_position_size(
                confluence_result, risk_metrics, current_price
            )
            
            if position_size <= 0:
                logger.debug(f"Position size too small for {symbol}")
                return None
            
            # 7. Optimizar niveles de entrada y salida
            optimized_levels = self._optimize_entry_exit_levels(
                confluence_result, momentum_result, current_price, risk_metrics
            )
            
            # 8. Generar señal final
            signal = TradeSignal(
                symbol=symbol,
                action='buy',  # Por ahora solo long positions
                confidence=confluence_result.confidence_level,
                entry_price=optimized_levels['entry'],
                stop_loss=optimized_levels['stop_loss'],
                take_profit_levels=optimized_levels['take_profits'],
                position_size=position_size,
                reasoning=self._generate_reasoning(momentum_result, confluence_result, risk_metrics),
                timestamp=datetime.now(),
                risk_level=confluence_result.risk_level,
                expected_return=optimized_levels['expected_return'],
                max_holding_time=timedelta(hours=self.max_holding_hours)
            )
            
            logger.info(f"Signal generated for {symbol}: {signal.action} at {signal.entry_price:.4f}, "
                       f"confidence={signal.confidence:.3f}, size={signal.position_size:.4f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def should_long(self, symbol: str = None) -> bool:
        """
        Determina si debe abrir posición long
        """
        # Esta función es llamada por el framework de trading
        # En nuestro caso, la lógica está en analyze_symbol
        if symbol and symbol in self.pending_orders:
            return True
        return False

    def should_short(self, symbol: str = None) -> bool:
        """
        Determina si debe abrir posición short
        Por ahora no implementamos shorts
        """
        return False

    def should_cancel(self, symbol: str = None) -> bool:
        """
        Determina si debe cancelar órdenes pendientes
        """
        if not symbol:
            return False
            
        # Cancelar si el símbolo ya no cumple criterios
        if symbol in self.pending_orders:
            order_time = self.pending_orders[symbol]['timestamp']
            if datetime.now() - order_time > timedelta(minutes=10):  # Timeout después de 10 min
                logger.info(f"Canceling stale order for {symbol}")
                return True
                
        return False

    async def go_long(self, symbol: str, current_price: float, **kwargs):
        """
        Ejecuta entrada en posición long
        """
        try:
            # Verificar que tenemos señal válida
            if symbol not in self.pending_orders:
                logger.warning(f"No pending order found for {symbol}")
                return
            
            order_info = self.pending_orders[symbol]
            
            # Verificar que el precio sigue siendo favorable
            price_diff = abs(current_price - order_info['entry_price']) / order_info['entry_price']
            if price_diff > 0.01:  # 1% slippage máximo
                logger.warning(f"Price moved too much for {symbol}: {price_diff:.3f}")
                del self.pending_orders[symbol]
                return
            
            # Calcular cantidad exacta basada en position size
            quantity = self._calculate_quantity(symbol, order_info['position_size'], current_price)
            
            # Registrar posición
            position_info = {
                'symbol': symbol,
                'entry_price': current_price,
                'quantity': quantity,
                'stop_loss': order_info['stop_loss'],
                'take_profit_levels': order_info['take_profit_levels'],
                'entry_time': datetime.now(),
                'confidence': order_info['confidence'],
                'risk_level': order_info['risk_level']
            }
            
            self.current_positions[symbol] = position_info
            
            # Limpiar orden pendiente
            del self.pending_orders[symbol]
            
            # Incrementar contador
            self.daily_trades += 1
            self.performance_metrics['total_trades'] += 1
            
            # Setup de stop loss y take profit dinámicos
            await self._setup_dynamic_exits(symbol, position_info)
            
            logger.info(f"Opened long position for {symbol}: quantity={quantity:.6f}, "
                       f"entry={current_price:.4f}, stop_loss={position_info['stop_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"Error opening long position for {symbol}: {e}")

    async def go_short(self, symbol: str, current_price: float, **kwargs):
        """
        Por ahora no implementamos short positions
        """
        logger.info(f"Short positions not implemented for {symbol}")
        pass

    async def update_positions(self):
        """
        Actualiza posiciones existentes con lógica adaptativa
        """
        for symbol, position in self.current_positions.copy().items():
            try:
                current_price = await self._get_current_price(symbol)
                
                # Verificar stop loss
                if current_price <= position['stop_loss']:
                    await self._close_position(symbol, current_price, 'stop_loss')
                    continue
                
                # Verificar take profit levels
                for i, tp_level in enumerate(position['take_profit_levels']):
                    if current_price >= tp_level:
                        # Cerrar parcialmente o completamente
                        await self._handle_take_profit(symbol, current_price, i)
                        break
                
                # Verificar time-based exit
                holding_time = datetime.now() - position['entry_time']
                if holding_time > timedelta(hours=self.max_holding_hours):
                    await self._close_position(symbol, current_price, 'time_limit')
                    continue
                
                # Actualizar stop loss dinámico (trailing stop)
                await self._update_trailing_stop(symbol, current_price, position)
                
            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {e}")

    async def _evaluate_risk(self, symbol: str, confluence_result: ConfluenceResult) -> RiskMetrics:
        """
        Evaluación integral de riesgo
        """
        # 1. Portfolio risk - exposición total
        total_exposure = sum(pos['quantity'] * pos['entry_price'] for pos in self.current_positions.values())
        portfolio_balance = await self._get_portfolio_balance()
        portfolio_risk = total_exposure / portfolio_balance if portfolio_balance > 0 else 0
        
        # 2. Symbol-specific risk
        symbol_risk = self._calculate_symbol_risk(symbol, confluence_result)
        
        # 3. Correlation risk - diversificación
        correlation_risk = await self._calculate_correlation_risk(symbol)
        
        # 4. Liquidity risk
        liquidity_risk = await self._calculate_liquidity_risk(symbol)
        
        # 5. Concentration risk
        concentration_risk = len(self.current_positions) / self.max_positions
        
        # Combinar riesgos
        total_risk_score = (
            portfolio_risk * 0.3 +
            symbol_risk * 0.25 +
            correlation_risk * 0.2 +
            liquidity_risk * 0.15 +
            concentration_risk * 0.1
        )
        
        return RiskMetrics(
            portfolio_risk=portfolio_risk,
            symbol_risk=symbol_risk,
            correlation_risk=correlation_risk,
            liquidity_risk=liquidity_risk,
            concentration_risk=concentration_risk,
            total_risk_score=total_risk_score
        )

    async def _calculate_dynamic_position_size(self, confluence_result: ConfluenceResult, 
                                       risk_metrics: RiskMetrics, current_price: float) -> float:
        """
        Calcula position size dinámico basado en confianza y riesgo
        """
        # Base size desde confluence
        base_size = confluence_result.max_position_size
        
        # Ajustar por confianza
        confidence_multiplier = confluence_result.confidence_level ** 0.5  # Suavizar
        
        # Ajustar por riesgo total
        risk_multiplier = max(0.1, 1.0 - risk_metrics.total_risk_score)
        
        # Ajustar por portfolio utilization
        utilization = len(self.current_positions) / self.max_positions
        utilization_multiplier = max(0.5, 1.0 - utilization)
        
        # Ajustar por volatilidad del símbolo
        volatility_multiplier = self._get_volatility_multiplier(confluence_result.risk_level)
        
        dynamic_size = (
            base_size * 
            confidence_multiplier * 
            risk_multiplier * 
            utilization_multiplier * 
            volatility_multiplier
        )
        
        # Aplicar límites
        portfolio_balance = await self._get_portfolio_balance()
        max_position_value = portfolio_balance * self.max_position_risk
        max_size_by_balance = max_position_value / current_price
        
        return min(dynamic_size, max_size_by_balance)

    def _optimize_entry_exit_levels(self, confluence_result: ConfluenceResult, 
                                   momentum_result, current_price: float, 
                                   risk_metrics: RiskMetrics) -> Dict:
        """
        Optimiza niveles de entrada y salida
        """
        # Entry price con micro-optimización
        entry_price = current_price * 0.999  # Slight discount for better entry
        
        # Stop loss adaptativo
        base_stop_distance = confluence_result.stop_loss_level
        risk_adjusted_stop = base_stop_distance * (1 + risk_metrics.total_risk_score * 0.3)
        
        # Take profit levels optimizados
        base_tp_levels = confluence_result.take_profit_levels
        confidence_bonus = confluence_result.confidence_level * 0.2
        
        optimized_tp_levels = [
            level * (1 + confidence_bonus) for level in base_tp_levels
        ]
        
        # Expected return calculation
        expected_return = (optimized_tp_levels[0] - entry_price) / entry_price
        
        return {
            'entry': entry_price,
            'stop_loss': risk_adjusted_stop,
            'take_profits': optimized_tp_levels,
            'expected_return': expected_return
        }

    def _generate_reasoning(self, momentum_result, confluence_result: ConfluenceResult, 
                          risk_metrics: RiskMetrics) -> str:
        """
        Genera explicación de la decisión de trading
        """
        reasoning_parts = [
            f"Momentum: {momentum_result.confidence:.2f}" if momentum_result else "Momentum: N/A",
            f"Confluence: {confluence_result.unified_score:.2f}",
            f"Confidence: {confluence_result.confidence_level:.2f}",
            f"Dominant TF: {confluence_result.dominant_timeframe}",
            f"Signal: {confluence_result.signal_strength}",
            f"Risk: {risk_metrics.total_risk_score:.2f}"
        ]
        
        return " | ".join(reasoning_parts)

    # Métodos helper para operaciones
    def _can_trade_symbol(self, symbol: str) -> bool:
        """Verifica si podemos operar el símbolo"""
        # Verificar límites diarios
        if self.daily_trades >= self.config.get('max_daily_trades', 10):
            return False
        
        # Verificar si ya tenemos posición
        if symbol in self.current_positions:
            return False
        
        # Verificar si ya tenemos orden pendiente
        if symbol in self.pending_orders:
            return False
        
        return True

    def _reset_daily_counters(self):
        """Reset contadores diarios"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = current_date
            logger.info("Daily counters reset")

    def _calculate_symbol_risk(self, symbol: str, confluence_result: ConfluenceResult) -> float:
        """Calcula riesgo específico del símbolo"""
        base_risk = 0.5  # Risk neutral
        
        # Ajustar por volatilidad
        volatility_adjustment = {
            'low': -0.1,
            'medium': 0.0,
            'high': 0.3
        }.get(confluence_result.risk_level, 0.0)
        
        # Ajustar por strength de señal
        signal_adjustment = {
            'very_strong': -0.2,
            'strong': -0.1,
            'moderate': 0.0,
            'weak': 0.3
        }.get(confluence_result.signal_strength, 0.0)
        
        return max(0.0, min(1.0, base_risk + volatility_adjustment + signal_adjustment))

    async def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calcula riesgo de correlación con posiciones existentes"""
        if not self.current_positions:
            return 0.0
        
        # Simplified correlation - en implementación real usaría correlaciones históricas
        similar_assets = sum(1 for pos_symbol in self.current_positions.keys() 
                           if pos_symbol[:3] == symbol[:3])  # Same base asset
        
        return min(1.0, similar_assets * 0.3)

    async def _calculate_liquidity_risk(self, symbol: str) -> float:
        """Calcula riesgo de liquidez"""
        # Simplified - en implementación real analizaría volume y spreads
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        return 0.1 if symbol in major_pairs else 0.4

    def _get_volatility_multiplier(self, risk_level: str) -> float:
        """Obtiene multiplicador por volatilidad"""
        return {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.7
        }.get(risk_level, 1.0)

    def _calculate_quantity(self, symbol: str, position_size: float, price: float) -> float:
        """Calcula cantidad exacta a comprar"""
        # Simplified calculation
        return position_size / price

    async def _setup_dynamic_exits(self, symbol: str, position_info: Dict):
        """Configura exits dinámicos"""
        # En implementación real, configuraría órdenes en el exchange
        logger.info(f"Dynamic exits configured for {symbol}")

    async def _get_current_price(self, symbol: str) -> float:
        """Obtiene precio actual del símbolo"""
        # Simulated - en implementación real conectaría con exchange
        return np.random.uniform(100, 110)

    async def _get_portfolio_balance(self) -> float:
        """Obtiene balance del portfolio"""
        # Simulated
        return 10000.0

    async def _close_position(self, symbol: str, price: float, reason: str):
        """Cierra posición"""
        if symbol in self.current_positions:
            position = self.current_positions[symbol]
            pnl = (price - position['entry_price']) * position['quantity']
            
            # Update performance metrics
            self.performance_metrics['total_pnl'] += pnl
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            
            del self.current_positions[symbol]
            logger.info(f"Closed position for {symbol} at {price:.4f}, reason: {reason}, PnL: {pnl:.2f}")

    async def _handle_take_profit(self, symbol: str, price: float, tp_level: int):
        """Maneja take profit parcial"""
        if symbol in self.current_positions:
            position = self.current_positions[symbol]
            
            # Cerrar 1/3 en cada nivel
            close_fraction = 1.0 / len(position['take_profit_levels'])
            close_quantity = position['quantity'] * close_fraction
            
            pnl = (price - position['entry_price']) * close_quantity
            self.performance_metrics['total_pnl'] += pnl
            
            # Actualizar posición
            position['quantity'] -= close_quantity
            
            if position['quantity'] <= 0:
                del self.current_positions[symbol]
            
            logger.info(f"Partial close for {symbol} at TP{tp_level+1}: {price:.4f}, PnL: {pnl:.2f}")

    async def _update_trailing_stop(self, symbol: str, current_price: float, position: Dict):
        """Actualiza trailing stop"""
        entry_price = position['entry_price']
        current_stop = position['stop_loss']
        
        # Activar trailing cuando ganancia > 3%
        if current_price > entry_price * 1.03:
            new_stop = current_price * 0.98  # 2% trailing
            if new_stop > current_stop:
                position['stop_loss'] = new_stop
                logger.debug(f"Updated trailing stop for {symbol}: {new_stop:.4f}")

    async def get_strategy_status(self) -> Dict:
        """
        Obtiene status completo de la estrategia
        """
        return {
            'positions': len(self.current_positions),
            'pending_orders': len(self.pending_orders),
            'daily_trades': self.daily_trades,
            'performance': self.performance_metrics,
            'risk_utilization': len(self.current_positions) / self.max_positions,
            'components_status': {
                'momentum_detector': 'active',
                'multi_tf_analyzer': 'active'
            }
        }
