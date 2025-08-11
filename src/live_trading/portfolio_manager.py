"""
Portfolio Manager - Gestiona el portfolio y ejecuta operaciones de trading
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
import ccxt.async_support as ccxt

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager
from src.strategies.momentum_predictor_strategy import TradeSignal

logger = get_logger(__name__)

@dataclass
class Position:
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0

@dataclass
class Portfolio:
    total_balance: float
    available_balance: float
    positions: Dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    trades_today: int = 0

@dataclass
class OrderResult:
    success: bool
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    error_message: str = ""

class PortfolioManager:
    """
    Gestiona el portfolio, posiciones y ejecuta señales de trading
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.exchange = None
        self.portfolio = Portfolio(
            total_balance=10000.0,  # Demo balance
            available_balance=10000.0
        )
        self.risk_limits = {
            'max_position_size': config.get('risk_management.max_position_size', 0.1),
            'max_daily_trades': config.get('risk_management.max_daily_trades', 10),
            'max_portfolio_risk': config.get('risk_management.max_portfolio_risk', 0.3),
            'stop_loss_pct': config.get('risk_management.stop_loss_pct', 0.05),
            'take_profit_pct': config.get('risk_management.take_profit_pct', 0.10)
        }
        
    async def initialize(self):
        """Inicializa la conexión al exchange"""
        try:
            exchange_config = self.config.get('exchanges.binance', {})
            
            if exchange_config.get('enabled', False):
                self.exchange = ccxt.binance({
                    'apiKey': exchange_config.get('api_key', ''),
                    'secret': exchange_config.get('secret_key', ''),
                    'sandbox': exchange_config.get('sandbox', True),
                    'enableRateLimit': True,
                })
                
                # Obtener balance real si está configurado
                if exchange_config.get('api_key'):
                    try:
                        balance = await self.exchange.fetch_balance()
                        self.portfolio.total_balance = float(balance['USDT']['total'])
                        self.portfolio.available_balance = float(balance['USDT']['free'])
                        logger.info(f"Balance actual: ${self.portfolio.total_balance:.2f} USDT")
                    except Exception as e:
                        logger.warning(f"No se pudo obtener balance real: {e}")
                        
            logger.info("Portfolio Manager inicializado")
            
        except Exception as e:
            logger.error(f"Error inicializando Portfolio Manager: {e}")
            raise
    
    async def execute_signals(self, signals: List[TradeSignal]) -> List[OrderResult]:
        """
        Ejecuta una lista de señales de trading
        """
        results = []
        
        for signal in signals:
            try:
                # Verificar límites de riesgo
                if not self._can_execute_trade(signal):
                    logger.warning(f"Trade rechazado por límites de riesgo: {signal.symbol}")
                    continue
                    
                # Calcular tamaño de posición
                position_size = self._calculate_position_size(signal)
                
                if position_size <= 0:
                    logger.warning(f"Tamaño de posición inválido para {signal.symbol}")
                    continue
                
                # Ejecutar orden
                if signal.action == 'buy':
                    result = await self._execute_buy_order(signal, position_size)
                elif signal.action == 'sell':
                    result = await self._execute_sell_order(signal, position_size)
                else:
                    continue
                    
                results.append(result)
                
                # Actualizar contador de trades
                if result.success:
                    self.portfolio.trades_today += 1
                    logger.info(f"Trade ejecutado: {signal.action} {position_size} {signal.symbol} @ ${signal.entry_price}")
                
            except Exception as e:
                logger.error(f"Error ejecutando señal para {signal.symbol}: {e}")
                results.append(OrderResult(
                    success=False,
                    order_id="",
                    symbol=signal.symbol,
                    side=signal.action,
                    amount=0,
                    price=0,
                    error_message=str(e)
                ))
        
        return results
    
    async def _execute_buy_order(self, signal: TradeSignal, size: float) -> OrderResult:
        """Ejecuta orden de compra"""
        try:
            if self.exchange and not self.config.get('trading.paper_trading', True):
                # Trading real
                order = await self.exchange.create_market_buy_order(
                    signal.symbol, 
                    size
                )
                
                order_result = OrderResult(
                    success=True,
                    order_id=order['id'],
                    symbol=signal.symbol,
                    side='buy',
                    amount=size,
                    price=float(order['average'] or signal.entry_price)
                )
            else:
                # Paper trading
                order_result = OrderResult(
                    success=True,
                    order_id=f"paper_{datetime.now().timestamp()}",
                    symbol=signal.symbol,
                    side='buy',
                    amount=size,
                    price=signal.entry_price
                )
            
            # Crear posición
            if order_result.success:
                position = Position(
                    symbol=signal.symbol,
                    side='long',
                    size=size,
                    entry_price=order_result.price,
                    current_price=order_result.price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit_levels[0] if signal.take_profit_levels else 0,
                    timestamp=datetime.now()
                )
                
                self.portfolio.positions[signal.symbol] = position
                self.portfolio.available_balance -= (size * order_result.price)
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error ejecutando compra de {signal.symbol}: {e}")
            return OrderResult(
                success=False,
                order_id="",
                symbol=signal.symbol,
                side='buy',
                amount=0,
                price=0,
                error_message=str(e)
            )
    
    async def _execute_sell_order(self, signal: TradeSignal, size: float) -> OrderResult:
        """Ejecuta orden de venta"""
        try:
            # Verificar si tenemos posición para vender
            if signal.symbol not in self.portfolio.positions:
                logger.warning(f"No hay posición para vender en {signal.symbol}")
                return OrderResult(
                    success=False,
                    order_id="",
                    symbol=signal.symbol,
                    side='sell',
                    amount=0,
                    price=0,
                    error_message="No position to sell"
                )
            
            position = self.portfolio.positions[signal.symbol]
            sell_size = min(size, position.size)
            
            if self.exchange and not self.config.get('trading.paper_trading', True):
                # Trading real
                order = await self.exchange.create_market_sell_order(
                    signal.symbol, 
                    sell_size
                )
                
                order_result = OrderResult(
                    success=True,
                    order_id=order['id'],
                    symbol=signal.symbol,
                    side='sell',
                    amount=sell_size,
                    price=float(order['average'] or signal.entry_price)
                )
            else:
                # Paper trading
                order_result = OrderResult(
                    success=True,
                    order_id=f"paper_{datetime.now().timestamp()}",
                    symbol=signal.symbol,
                    side='sell',
                    amount=sell_size,
                    price=signal.entry_price
                )
            
            # Actualizar posición
            if order_result.success:
                pnl = (order_result.price - position.entry_price) * sell_size
                self.portfolio.total_pnl += pnl
                self.portfolio.daily_pnl += pnl
                self.portfolio.available_balance += (sell_size * order_result.price)
                
                # Cerrar posición si se vendió todo
                if sell_size >= position.size:
                    del self.portfolio.positions[signal.symbol]
                else:
                    position.size -= sell_size
            
            return order_result
            
        except Exception as e:
            logger.error(f"Error ejecutando venta de {signal.symbol}: {e}")
            return OrderResult(
                success=False,
                order_id="",
                symbol=signal.symbol,
                side='sell',
                amount=0,
                price=0,
                error_message=str(e)
            )
    
    def _can_execute_trade(self, signal: TradeSignal) -> bool:
        """Verifica si se puede ejecutar el trade según límites de riesgo"""
        
        # Verificar límite de trades diarios
        if self.portfolio.trades_today >= self.risk_limits['max_daily_trades']:
            return False
        
        # Verificar balance disponible
        required_balance = signal.position_size * signal.entry_price
        if required_balance > self.portfolio.available_balance:
            return False
        
        # Verificar riesgo del portfolio
        total_risk = sum(
            pos.size * pos.current_price 
            for pos in self.portfolio.positions.values()
        )
        
        if (total_risk + required_balance) / self.portfolio.total_balance > self.risk_limits['max_portfolio_risk']:
            return False
        
        return True
    
    def _calculate_position_size(self, signal: TradeSignal) -> float:
        """Calcula el tamaño de posición basado en la confianza de la señal"""
        
        # Tamaño base según configuración
        max_position_value = self.portfolio.total_balance * self.risk_limits['max_position_size']
        
        # Ajustar según confianza de la señal
        confidence_multiplier = min(signal.confidence, 1.0)
        adjusted_position_value = max_position_value * confidence_multiplier
        
        # Calcular tamaño en unidades
        position_size = adjusted_position_value / signal.entry_price
        
        # Verificar balance disponible
        max_affordable = self.portfolio.available_balance / signal.entry_price
        position_size = min(position_size, max_affordable)
        
        return max(0, position_size)
    
    async def update_positions(self, current_prices: Dict[str, float]):
        """Actualiza precios y PnL de posiciones actuales"""
        
        for symbol, position in self.portfolio.positions.items():
            if symbol in current_prices:
                position.current_price = current_prices[symbol]
                
                # Calcular PnL
                if position.side == 'long':
                    position.pnl = (position.current_price - position.entry_price) * position.size
                else:  # short
                    position.pnl = (position.entry_price - position.current_price) * position.size
                
                position.pnl_pct = (position.pnl / (position.entry_price * position.size)) * 100
                
                # Verificar stop loss y take profit
                await self._check_exit_conditions(symbol, position)
    
    async def _check_exit_conditions(self, symbol: str, position: Position):
        """Verifica condiciones de salida (stop loss / take profit)"""
        
        should_exit = False
        exit_reason = ""
        
        if position.side == 'long':
            # Stop loss
            if position.current_price <= position.stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
            
            # Take profit
            elif position.current_price >= position.take_profit:
                should_exit = True
                exit_reason = "Take Profit"
        
        else:  # short position
            # Stop loss
            if position.current_price >= position.stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
            
            # Take profit
            elif position.current_price <= position.take_profit:
                should_exit = True
                exit_reason = "Take Profit"
        
        if should_exit:
            logger.info(f"Cerrando posición {symbol}: {exit_reason}")
            
            # Crear señal de salida
            exit_signal = TradeSignal(
                symbol=symbol,
                action='sell' if position.side == 'long' else 'buy',
                confidence=1.0,
                entry_price=position.current_price,
                stop_loss=0,
                take_profit_levels=[],
                position_size=position.size,
                reasoning=exit_reason,
                timestamp=datetime.now(),
                risk_level='low',
                expected_return=0,
                max_holding_time=timedelta(hours=1)
            )
            
            await self.execute_signals([exit_signal])
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Retorna resumen del portfolio"""
        
        total_position_value = sum(
            pos.size * pos.current_price 
            for pos in self.portfolio.positions.values()
        )
        
        return {
            'total_balance': self.portfolio.total_balance,
            'available_balance': self.portfolio.available_balance,
            'invested_amount': total_position_value,
            'total_pnl': self.portfolio.total_pnl,
            'daily_pnl': self.portfolio.daily_pnl,
            'positions_count': len(self.portfolio.positions),
            'trades_today': self.portfolio.trades_today,
            'utilization_pct': (total_position_value / self.portfolio.total_balance) * 100
        }
    
    async def close(self):
        """Cierra conexiones"""
        if self.exchange:
            await self.exchange.close()

# Función de testing
async def test_portfolio_manager():
    """Función de prueba para el portfolio manager"""
    from src.utils.config_manager import ConfigManager
    
    config = ConfigManager()
    manager = PortfolioManager(config)
    
    await manager.initialize()
    
    # Crear señal de prueba
    test_signal = TradeSignal(
        symbol="BTC/USDT",
        action="buy",
        confidence=0.8,
        entry_price=45000.0,
        stop_loss=43000.0,
        take_profit_levels=[47000.0],
        position_size=0.01,
        reasoning="Test signal",
        timestamp=datetime.now(),
        risk_level="medium",
        expected_return=0.04,
        max_holding_time=timedelta(hours=24)
    )
    
    # Ejecutar señal
    results = await manager.execute_signals([test_signal])
    print(f"Resultado: {results[0].success}")
    
    # Ver resumen
    summary = manager.get_portfolio_summary()
    print(f"Portfolio: {summary}")
    
    await manager.close()

if __name__ == "__main__":
    asyncio.run(test_portfolio_manager())
