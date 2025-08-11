#!/usr/bin/env python3
"""
Bot principal de predicción de momentum - Versión Avanzada
Soporte para TODOS los pares USDT de Binance usando WebSockets
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent))

from utils.logger import get_logger
from utils.config_manager import ConfigManager
from utils.notification_system import NotificationSystem
from strategies.momentum_predictor_strategy import MomentumPredictorStrategy
from strategies.momentum_detector import MomentumDetector
from strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from data_sources.binance_websocket_aggregator import BinanceWebSocketAggregator, TickerData, KlineData
from live_trading.portfolio_manager import PortfolioManager
from ml_models.ml_predictor import MLPredictor

logger = get_logger(__name__)

class AdvancedMomentumBot:
    """
    Bot avanzado que analiza TODOS los pares USDT de Binance
    Usa WebSockets para datos en tiempo real y evitar rate limits
    """
    
    def __init__(self):
        self.config = ConfigManager()
        self.is_running = False
        self.components = {}
        
        # Estado del bot
        self.analyze_all_pairs = self.config.get('bot.analyze_all_usdt_pairs', False)
        self.active_pairs = set()
        self.market_data = {}
        self.signals_cache = {}
        
        # Performance metrics
        self.processed_pairs = 0
        self.signals_generated = 0
        self.last_cycle_time = 0
        self.cycle_count = 0
        
        # Configurar manejo de señales del sistema
        self._setup_signal_handlers()
        
        logger.info(f"🚀 Bot inicializado - Modo: {'TODOS los pares USDT' if self.analyze_all_pairs else 'pares específicos'}")
        
    def _setup_signal_handlers(self):
        """Configurar manejadores de señales del sistema"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Señal {signum} recibida, iniciando cierre...")
            self.is_running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    async def initialize(self):
        """Inicializa todos los componentes del bot"""
        try:
            logger.info("🔧 Inicializando componentes avanzados...")
            
            # 1. Inicializar WebSocket aggregator
            await self._initialize_websocket_aggregator()
            
            # 2. Inicializar componentes de trading
            await self._initialize_trading_components()
            
            # 3. Configurar callbacks de datos
            self._setup_data_callbacks()
            
            # 4. Entrenar modelos ML si es necesario
            await self._train_ml_models_if_needed()
            
            logger.info("✅ Todos los componentes inicializados correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando bot: {e}")
            raise
    
    async def _initialize_websocket_aggregator(self):
        """Inicializar el agregador WebSocket avanzado"""
        aggregator_config = self.config.get('data_aggregator', {})
        
        self.components['websocket_aggregator'] = BinanceWebSocketAggregator(
            max_workers=aggregator_config.get('max_workers', 20),
            chunk_size=aggregator_config.get('chunk_size', 50),
            min_volume_24h=aggregator_config.get('min_volume_24h', 1000000),
            max_pairs=aggregator_config.get('max_pairs', 200),
            ping_interval=aggregator_config.get('ping_interval', 20),
            max_reconnections=aggregator_config.get('max_reconnections', 5)
        )
        
        # Configurar pares a analizar
        if not self.analyze_all_pairs:
            specific_pairs = self.config.get('bot.trading_pairs', [])
            binance_pairs = [pair.replace('/', '') for pair in specific_pairs]
            await self.components['websocket_aggregator'].set_custom_pairs(binance_pairs)
            logger.info(f"📋 Configurado para pares específicos: {binance_pairs}")
        else:
            logger.info("🌍 Configurado para TODOS los pares USDT de Binance")
        
        logger.info("✅ WebSocket aggregator inicializado")
    
    async def _initialize_trading_components(self):
        """Inicializar componentes de trading y análisis"""
        # Portfolio Manager
        self.components['portfolio_manager'] = PortfolioManager(self.config)
        await self.components['portfolio_manager'].initialize()
        
        # ML Predictor
        self.components['ml_predictor'] = MLPredictor(self.config)
        
        # Momentum Detector
        self.components['momentum_detector'] = MomentumDetector(self.config)
        
        # Multi-timeframe Analyzer
        self.components['timeframe_analyzer'] = MultiTimeframeAnalyzer(self.config)
        
        # Main Strategy
        self.components['main_strategy'] = MomentumPredictorStrategy(self.config)
        
        # Notification System
        self.components['notification_system'] = NotificationSystem(self.config)
        await self.components['notification_system'].initialize()
        
        logger.info("✅ Componentes de trading inicializados")
    
    def _setup_data_callbacks(self):
        """Configurar callbacks para datos en tiempo real"""
        aggregator = self.components['websocket_aggregator']
        
        # Callback para datos de ticker
        async def on_ticker_update(symbol: str, ticker: TickerData):
            await self._process_ticker_data(symbol, ticker)
        
        # Callback para datos de klines
        async def on_kline_update(symbol: str, kline: KlineData):
            await self._process_kline_data(symbol, kline)
        
        # Callback para errores de conexión
        async def on_connection_error(symbol: str, error: Exception):
            logger.warning(f"⚠️ Error de conexión para {symbol}: {error}")
        
        # Registrar callbacks
        aggregator.register_ticker_callback(on_ticker_update)
        aggregator.register_kline_callback(on_kline_update)
        aggregator.register_error_callback(on_connection_error)
        
        logger.info("✅ Callbacks de datos configurados")
    
    async def _process_ticker_data(self, symbol: str, ticker: TickerData):
        """Procesar datos de ticker en tiempo real"""
        try:
            self.active_pairs.add(symbol)
            
            # Inicializar estructura de datos si no existe
            if symbol not in self.market_data:
                self.market_data[symbol] = {
                    'ticker': None,
                    'klines': [],
                    'last_analysis': None,
                    'signals': []
                }
            
            # Actualizar ticker data
            self.market_data[symbol]['ticker'] = ticker
            
            # Analizar si hay suficientes datos
            if self._has_enough_data(symbol):
                await self._analyze_symbol(symbol)
                
        except Exception as e:
            logger.error(f"❌ Error procesando ticker para {symbol}: {e}")
    
    async def _process_kline_data(self, symbol: str, kline: KlineData):
        """Procesar datos de klines en tiempo real"""
        try:
            if symbol not in self.market_data:
                self.market_data[symbol] = {
                    'ticker': None,
                    'klines': [],
                    'last_analysis': None,
                    'signals': []
                }
            
            # Agregar nueva kline
            klines = self.market_data[symbol]['klines']
            klines.append(kline)
            
            # Mantener solo las últimas N klines
            max_klines = self.config.get('data_aggregator.klines_buffer_size', 1000)
            if len(klines) > max_klines:
                self.market_data[symbol]['klines'] = klines[-max_klines:]
            
        except Exception as e:
            logger.error(f"❌ Error procesando kline para {symbol}: {e}")
    
    def _has_enough_data(self, symbol: str) -> bool:
        """Verificar si hay suficientes datos para análisis"""
        if symbol not in self.market_data:
            return False
        
        data = self.market_data[symbol]
        min_klines = 50  # Mínimo para análisis técnico
        
        return (
            data['ticker'] is not None and
            len(data['klines']) >= min_klines
        )
    
    async def _analyze_symbol(self, symbol: str):
        """Analizar un símbolo específico"""
        try:
            # Verificar cooldown de análisis
            cooldown = self.config.get('momentum.signal_cooldown', 300)
            if self._is_analysis_in_cooldown(symbol, cooldown):
                return
            
            # Obtener datos del símbolo
            data = self.market_data[symbol]
            ticker = data['ticker']
            klines = data['klines']
            
            # Convertir klines a DataFrame
            df = self._convert_klines_to_dataframe(klines)
            if df.empty or len(df) < 20:
                return
            
            # Ejecutar estrategia principal
            current_price = ticker.last_price
            signals = await self.components['main_strategy'].analyze_symbol(
                symbol, df, current_price
            )
            
            if signals:
                # Filtrar señales por confianza
                min_confidence = self.config.get('momentum.min_confidence_threshold', 0.6)
                high_confidence_signals = [
                    s for s in signals 
                    if getattr(s, 'confidence', 0) > min_confidence
                ]
                
                if high_confidence_signals:
                    await self._handle_trading_signals(symbol, high_confidence_signals)
            
            # Actualizar timestamp de último análisis
            self.market_data[symbol]['last_analysis'] = datetime.now()
            self.processed_pairs += 1
            
        except Exception as e:
            logger.error(f"❌ Error analizando {symbol}: {e}")
    
    def _convert_klines_to_dataframe(self, klines: List[KlineData]) -> pd.DataFrame:
        """Convertir lista de klines a DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        data = []
        for kline in klines:
            data.append({
                'timestamp': kline.close_time,
                'open': kline.open_price,
                'high': kline.high_price,
                'low': kline.low_price,
                'close': kline.close_price,
                'volume': kline.volume
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
        
        return df
    
    def _is_analysis_in_cooldown(self, symbol: str, cooldown_seconds: int) -> bool:
        """Verificar si el análisis está en cooldown"""
        if symbol not in self.market_data:
            return False
        
        last_analysis = self.market_data[symbol].get('last_analysis')
        if not last_analysis:
            return False
        
        time_diff = (datetime.now() - last_analysis).total_seconds()
        return time_diff < cooldown_seconds
    
    async def _handle_trading_signals(self, symbol: str, signals: List):
        """Manejar señales de trading"""
        try:
            # Ejecutar señales
            results = await self.components['portfolio_manager'].execute_signals(signals)
            
            # Contar señales exitosas
            successful_trades = [r for r in results if r.success]
            self.signals_generated += len(successful_trades)
            
            # Notificar trades exitosos
            if successful_trades:
                await self._notify_trades(successful_trades)
                logger.info(f"🚀 {len(successful_trades)} señales ejecutadas para {symbol}")
            
            # Almacenar señales en cache
            self.signals_cache[symbol] = {
                'signals': signals,
                'timestamp': datetime.now(),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"❌ Error manejando señales para {symbol}: {e}")
    
    async def _notify_trades(self, successful_trades):
        """Notificar trades ejecutados"""
        for trade in successful_trades:
            await self.components['notification_system'].send_alert({
                'type': 'trade',
                'title': f'Trade Ejecutado: {trade.symbol}',
                'message': f'{trade.side.upper()} {trade.amount:.6f} @ ${trade.price:.2f}',
                'data': {
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'amount': trade.amount,
                    'price': trade.price,
                    'order_id': getattr(trade, 'order_id', 'demo')
                }
            })
    
    async def _train_ml_models_if_needed(self):
        """Entrenar modelos ML para los pares principales"""
        try:
            # Solo entrenar para los pares más importantes si analizamos todos
            if self.analyze_all_pairs:
                train_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
            else:
                symbols = self.config.get('bot.trading_pairs', ['BTC/USDT'])
                train_pairs = [s.replace('/', '') for s in symbols]
            
            logger.info(f"🤖 Entrenando modelos ML para {len(train_pairs)} pares")
            
            # Este es un placeholder - en producción usaríamos datos históricos reales
            for symbol in train_pairs:
                logger.info(f"📊 Modelo preparado para {symbol}")
                
        except Exception as e:
            logger.error(f"❌ Error entrenando modelos ML: {e}")
    
    async def run(self):
        """Bucle principal del bot"""
        self.is_running = True
        
        logger.info("🔄 Iniciando bucle principal del bot avanzado...")
        
        # Notificación de inicio
        await self.components['notification_system'].send_alert({
            'type': 'info',
            'title': 'Bot Avanzado Iniciado',
            'message': f"Análisis {'de TODOS los pares' if self.analyze_all_pairs else 'específico'} iniciado",
            'data': {
                'timestamp': datetime.now().isoformat(),
                'mode': 'all_pairs' if self.analyze_all_pairs else 'specific_pairs',
                'version': self.config.get('bot.version', '2.0.0')
            }
        })
        
        # Iniciar WebSocket aggregator
        await self.components['websocket_aggregator'].start()
        
        try:
            while self.is_running:
                cycle_start = datetime.now()
                self.cycle_count += 1
                
                # Log de estadísticas cada 10 ciclos
                if self.cycle_count % 10 == 0:
                    await self._log_statistics()
                
                # Resumen periódico cada hora
                if self.cycle_count % 3600 == 0:  # Asumiendo ciclo de 1 segundo
                    await self._send_periodic_summary()
                
                # Gestión de portfolio
                await self._manage_portfolio()
                
                # Calcular tiempo de ciclo
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                self.last_cycle_time = cycle_time
                
                # Esperar próximo ciclo
                cycle_interval = self.config.get('bot.cycle_interval', 60)
                await asyncio.sleep(max(1, cycle_interval - cycle_time))
                
        except KeyboardInterrupt:
            logger.info("⏹️ Bot detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error crítico en bot: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _log_statistics(self):
        """Log de estadísticas del bot"""
        stats = {
            'pares_activos': len(self.active_pairs),
            'pares_procesados': self.processed_pairs,
            'señales_generadas': self.signals_generated,
            'tiempo_ultimo_ciclo': f"{self.last_cycle_time:.2f}s",
            'ciclo': self.cycle_count,
            'modo': 'TODOS' if self.analyze_all_pairs else 'específicos'
        }
        
        logger.info(f"📊 Estadísticas del bot: {stats}")
        
        # Resetear contadores
        self.processed_pairs = 0
    
    async def _manage_portfolio(self):
        """Gestión del portfolio"""
        try:
            # Actualizar precios actuales
            current_prices = {}
            for symbol, data in self.market_data.items():
                if data.get('ticker'):
                    current_prices[symbol] = data['ticker'].last_price
            
            if current_prices:
                await self.components['portfolio_manager'].update_positions(current_prices)
                
        except Exception as e:
            logger.error(f"❌ Error en gestión de portfolio: {e}")
    
    async def _send_periodic_summary(self):
        """Enviar resumen periódico del portfolio"""
        try:
            summary = self.components['portfolio_manager'].get_portfolio_summary()
            
            await self.components['notification_system'].send_alert({
                'type': 'summary',
                'title': 'Resumen del Portfolio Avanzado',
                'message': f"Balance: ${summary.get('total_balance', 0):.2f} | "
                          f"PnL: ${summary.get('total_pnl', 0):.2f} | "
                          f"Pares: {len(self.active_pairs)}",
                'data': {
                    **summary,
                    'active_pairs': len(self.active_pairs),
                    'signals_generated': self.signals_generated
                }
            })
            
        except Exception as e:
            logger.error(f"❌ Error enviando resumen: {e}")
    
    async def shutdown(self):
        """Cierra todos los componentes del bot"""
        logger.info("🛑 Cerrando bot avanzado...")
        
        self.is_running = False
        
        # Detener WebSocket aggregator primero
        if 'websocket_aggregator' in self.components:
            try:
                await self.components['websocket_aggregator'].stop()
                logger.info("✅ WebSocket aggregator cerrado")
            except Exception as e:
                logger.error(f"❌ Error cerrando WebSocket aggregator: {e}")
        
        # Cerrar otros componentes
        for name, component in self.components.items():
            if name == 'websocket_aggregator':
                continue  # Ya cerrado
                
            try:
                if hasattr(component, 'close'):
                    await component.close()
                logger.info(f"✅ {name} cerrado")
            except Exception as e:
                logger.error(f"❌ Error cerrando {name}: {e}")
        
        # Notificación de cierre
        if 'notification_system' in self.components:
            try:
                await self.components['notification_system'].send_alert({
                    'type': 'info',
                    'title': 'Bot Avanzado Detenido',
                    'message': 'El bot ha finalizado operaciones',
                    'data': {
                        'timestamp': datetime.now().isoformat(),
                        'total_signals': self.signals_generated,
                        'total_pairs': len(self.active_pairs)
                    }
                })
            except:
                pass
        
        logger.info("✅ Bot avanzado cerrado correctamente")

async def main():
    """Función principal del bot avanzado"""
    try:
        # Configurar política de eventos para Windows
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Crear e inicializar bot
        bot = AdvancedMomentumBot()
        await bot.initialize()
        
        # Ejecutar bot
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot interrumpido por usuario")
    except Exception as e:
        logger.error(f"❌ Error crítico: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
