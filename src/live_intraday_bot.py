"""
Bot de Trading Intradía con WebSocket en Tiempo Real
Integra detección de momentum, precio de entrada y alertas de precaución
"""

import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import signal
import sys
import os

# Agregar path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports con fallback
try:
    from src.data_sources.binance_websocket_aggregator import BinanceWebSocketAggregator
    from src.strategies.intraday_trading_signals import IntradayTradingSignals, SignalType, RiskLevel
    from src.strategies.momentum_detector import MomentumDetector
    from src.utils.logger import get_logger
except ImportError:
    try:
        from data_sources.binance_websocket_aggregator import BinanceWebSocketAggregator
        from strategies.intraday_trading_signals import IntradayTradingSignals, SignalType, RiskLevel
        from strategies.momentum_detector import MomentumDetector
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

class LiveIntradayTradingBot:
    """
    Bot de trading intradía en tiempo real con WebSocket
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Configuración del bot
        self.min_momentum_threshold = self.config.get('min_momentum_threshold', 60)
        self.max_pairs_to_monitor = self.config.get('max_pairs_to_monitor', 50)
        self.signal_cooldown_minutes = self.config.get('signal_cooldown_minutes', 30)
        
        # Componentes del bot
        self.websocket_aggregator = None
        self.trading_signals = IntradayTradingSignals(config)
        self.momentum_detector = None
        
        # Estado del bot
        self.market_data = {}
        self.recent_signals = {}  # Para evitar spam
        self.is_running = False
        self.active_signals = {}
        
        # Estadísticas
        self.signals_generated = 0
        self.high_confidence_signals = 0
        self.alerts_generated = 0
        
        self.logger.info("🤖 LiveIntradayTradingBot inicializado")
    
    async def start(self):
        """Inicia el bot de trading en tiempo real"""
        try:
            self.logger.info("🚀 Iniciando bot de trading intradía...")
            self.is_running = True
            
            # Configurar manejo de señales para parada elegante
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Inicializar componentes
            await self._initialize_components()
            
            # Iniciar WebSocket agregador
            await self._start_websocket_monitoring()
            
            # Bucle principal de procesamiento
            await self._main_processing_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Error iniciando bot: {e}")
            await self.stop()
    
    async def stop(self):
        """Detiene el bot de manera elegante"""
        self.logger.info("🛑 Deteniendo bot de trading...")
        self.is_running = False
        
        if self.websocket_aggregator:
            await self.websocket_aggregator.stop()
        
        self._print_final_statistics()
        self.logger.info("✅ Bot detenido correctamente")
    
    async def _initialize_components(self):
        """Inicializa todos los componentes necesarios"""
        try:
            # Configuración para WebSocket
            ws_config = {
                'max_pairs': self.max_pairs_to_monitor,
                'min_volume_usdt': 10000000,  # 10M USDT mínimo
                'update_interval': 1.0,
                'enable_price_alerts': True
            }
            
            # Inicializar WebSocket aggregator
            self.websocket_aggregator = BinanceWebSocketAggregator(ws_config)
            
            # Inicializar momentum detector
            momentum_config = {
                'window_size': 20,
                'volume_threshold': 1.5,
                'price_change_threshold': 0.02
            }
            
            try:
                self.momentum_detector = MomentumDetector(momentum_config)
            except:
                self.logger.warning("⚠️ MomentumDetector no disponible, usando análisis básico")
                self.momentum_detector = None
            
            self.logger.info("✅ Componentes inicializados correctamente")
            
        except Exception as e:
            self.logger.error(f"❌ Error inicializando componentes: {e}")
            raise
    
    async def _start_websocket_monitoring(self):
        """Inicia el monitoreo WebSocket"""
        try:
            self.logger.info("📡 Iniciando monitoreo WebSocket...")
            
            # Configurar callback para datos de mercado
            await self.websocket_aggregator.start(
                data_callback=self._process_market_data
            )
            
            # Esperar a que se establezcan las conexiones
            await asyncio.sleep(5)
            
            pairs_count = len(self.websocket_aggregator.active_pairs)
            self.logger.info(f"✅ Monitoreando {pairs_count} pares de trading")
            
        except Exception as e:
            self.logger.error(f"❌ Error iniciando WebSocket: {e}")
            raise
    
    async def _process_market_data(self, symbol: str, data: Dict):
        """Procesa datos de mercado en tiempo real"""
        try:
            # Actualizar datos de mercado
            if symbol not in self.market_data:
                self.market_data[symbol] = []
            
            # Agregar nuevo punto de datos
            market_point = {
                'timestamp': datetime.now(),
                'price': float(data.get('price', 0)),
                'volume': float(data.get('volume', 0)),
                'change_24h': float(data.get('change_24h', 0))
            }
            
            self.market_data[symbol].append(market_point)
            
            # Mantener solo los últimos 100 puntos
            if len(self.market_data[symbol]) > 100:
                self.market_data[symbol] = self.market_data[symbol][-100:]
            
            # Procesar señales si tenemos suficientes datos
            if len(self.market_data[symbol]) >= 20:
                await self._analyze_trading_opportunity(symbol, data)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error procesando datos para {symbol}: {e}")
    
    async def _analyze_trading_opportunity(self, symbol: str, current_data: Dict):
        """Analiza oportunidades de trading para un símbolo"""
        try:
            # Verificar cooldown
            if self._is_in_cooldown(symbol):
                return
            
            # Crear DataFrame con datos históricos
            df_data = []
            for point in self.market_data[symbol]:
                df_data.append({
                    'timestamp': point['timestamp'],
                    'close': point['price'],
                    'volume': point['volume'],
                    'high': point['price'] * 1.002,  # Aproximación
                    'low': point['price'] * 0.998,   # Aproximación
                    'open': point['price']
                })
            
            market_df = pd.DataFrame(df_data)
            market_df.set_index('timestamp', inplace=True)
            
            # Calcular momentum
            momentum_score = self._calculate_momentum_score(market_df)
            
            # Filtrar por threshold mínimo
            if momentum_score < self.min_momentum_threshold:
                return
            
            # Generar señal de trading
            current_price = float(current_data.get('price', 0))
            signal = self.trading_signals.generate_trading_signal(
                symbol, current_price, market_df, momentum_score
            )
            
            # Procesar señal generada
            await self._process_trading_signal(symbol, signal, momentum_score)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error analizando {symbol}: {e}")
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """Calcula score de momentum básico"""
        try:
            if self.momentum_detector:
                # Usar detector avanzado si está disponible
                return self.momentum_detector.calculate_momentum(df)
            else:
                # Cálculo básico de momentum
                price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
                volume_ratio = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-20:].mean()
                
                # Score básico (0-100)
                momentum = max(0, min(100, 
                    (price_change * 100 * 2) +  # Factor precio
                    (volume_ratio - 1) * 30 +    # Factor volumen
                    50                            # Base
                ))
                
                return momentum
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculando momentum: {e}")
            return 50  # Momentum neutral
    
    async def _process_trading_signal(self, symbol: str, signal, momentum_score: float):
        """Procesa y muestra señal de trading"""
        try:
            # Actualizar estadísticas
            self.signals_generated += 1
            if signal.confidence > 0.7:
                self.high_confidence_signals += 1
            if signal.alerts:
                self.alerts_generated += len(signal.alerts)
            
            # Mostrar señal solo si es relevante
            if (signal.signal_type in [SignalType.BUY, SignalType.WATCH] and 
                signal.confidence > 0.6):
                
                await self._display_trading_signal(symbol, signal, momentum_score)
                
                # Registrar en signals activas
                self.active_signals[symbol] = {
                    'signal': signal,
                    'timestamp': datetime.now(),
                    'momentum': momentum_score
                }
                
                # Actualizar cooldown
                self.recent_signals[symbol] = datetime.now()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error procesando señal para {symbol}: {e}")
    
    async def _display_trading_signal(self, symbol: str, signal, momentum_score: float):
        """Muestra señal de trading formateada"""
        try:
            print("\n" + "="*80)
            print(f"🎯 SEÑAL DE TRADING EN VIVO - {symbol}")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')} | 💰 ${signal.current_price:.4f} | 📊 Momentum: {momentum_score:.0f}")
            print("="*80)
            
            # Tipo de señal con emoji
            signal_emoji = {
                SignalType.BUY: "🟢 COMPRAR",
                SignalType.WATCH: "🟡 OBSERVAR", 
                SignalType.SELL: "🔴 VENDER",
                SignalType.HOLD: "⚪ MANTENER",
                SignalType.AVOID: "⚫ EVITAR"
            }
            
            print(f"\n📊 SEÑAL: {signal_emoji.get(signal.signal_type, signal.signal_type.value)}")
            print(f"💪 Confianza: {signal.confidence*100:.1f}% | 🎯 Éxito: {signal.probability_success*100:.1f}%")
            print(f"⭐ Fuerza: {signal.signal_strength}/10 | 🛡️ Riesgo: {signal.risk_level.value}")
            
            # Estrategia de entrada
            print(f"\n💎 PRECIO DE ENTRADA ÓPTIMO")
            print(f"🎯 Entrada: ${signal.entry_strategy.optimal_price:.4f}")
            print(f"🛡️  Stop Loss: ${signal.entry_strategy.stop_loss:.4f}")
            print(f"🎪 Take Profit 1: ${signal.entry_strategy.take_profit_1:.4f}")
            print(f"🎪 Take Profit 2: ${signal.entry_strategy.take_profit_2:.4f}")
            print(f"📏 Tamaño: {signal.entry_strategy.position_size_recommended*100:.1f}% del capital")
            
            # Alertas de precaución
            if signal.alerts:
                print(f"\n⚠️ ALERTAS DE PRECAUCIÓN:")
                for alert in signal.alerts:
                    severity = "🔴" if alert.severity >= 8 else "🟠" if alert.severity >= 6 else "🟡"
                    print(f"   {severity} {alert.message}")
                    print(f"      💡 {alert.action_required}")
            
            # Observaciones importantes
            if signal.precautions:
                print(f"\n🔍 OBSERVAR EL MOVIMIENTO:")
                for precaution in signal.precautions[:3]:  # Solo las 3 más importantes
                    print(f"   • {precaution}")
            
            # Señales intradía
            if signal.intraday_signals:
                active_signals = [k for k, v in signal.intraday_signals.items() 
                                if isinstance(v, dict) and v.get('active')]
                if active_signals:
                    print(f"\n📈 TRADING INTRADÍA ACTIVO:")
                    for sig in active_signals:
                        details = signal.intraday_signals[sig]
                        timeframe = details.get('timeframe', 'N/A')
                        targets = details.get('exit_targets', [])
                        print(f"   🎯 {sig.replace('_', ' ').title()}: {timeframe} | Targets: {', '.join(targets)}")
            
            print("-" * 80)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error mostrando señal: {e}")
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Verifica si un símbolo está en período de cooldown"""
        if symbol not in self.recent_signals:
            return False
        
        last_signal = self.recent_signals[symbol]
        cooldown_period = timedelta(minutes=self.signal_cooldown_minutes)
        
        return datetime.now() - last_signal < cooldown_period
    
    async def _main_processing_loop(self):
        """Bucle principal de procesamiento"""
        self.logger.info("🔄 Iniciando bucle principal de análisis...")
        
        while self.is_running:
            try:
                # Mostrar estadísticas cada 5 minutos
                if self.signals_generated > 0 and self.signals_generated % 20 == 0:
                    self._print_statistics()
                
                # Limpiar señales expiradas
                await self._cleanup_expired_signals()
                
                # Esperar antes del siguiente ciclo
                await asyncio.sleep(30)  # Revisar cada 30 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error en bucle principal: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_expired_signals(self):
        """Limpia señales expiradas"""
        try:
            current_time = datetime.now()
            expired_symbols = []
            
            for symbol, signal_data in self.active_signals.items():
                if current_time > signal_data['signal'].expiry_time:
                    expired_symbols.append(symbol)
            
            for symbol in expired_symbols:
                del self.active_signals[symbol]
                self.logger.debug(f"🗑️ Señal expirada eliminada: {symbol}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error limpiando señales: {e}")
    
    def _print_statistics(self):
        """Imprime estadísticas del bot"""
        try:
            print(f"\n📊 ESTADÍSTICAS DEL BOT")
            print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
            print(f"🎯 Señales generadas: {self.signals_generated}")
            print(f"⭐ Alta confianza: {self.high_confidence_signals}")
            print(f"⚠️ Alertas emitidas: {self.alerts_generated}")
            print(f"📈 Señales activas: {len(self.active_signals)}")
            print(f"🔍 Pares monitoreados: {len(self.market_data)}")
            print("-" * 50)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error mostrando estadísticas: {e}")
    
    def _print_final_statistics(self):
        """Imprime estadísticas finales"""
        print(f"\n📊 ESTADÍSTICAS FINALES DEL BOT")
        print("="*50)
        print(f"🎯 Total señales generadas: {self.signals_generated}")
        print(f"⭐ Señales alta confianza: {self.high_confidence_signals}")
        print(f"⚠️ Total alertas emitidas: {self.alerts_generated}")
        print(f"📈 Pares únicos analizados: {len(self.market_data)}")
        
        if self.signals_generated > 0:
            confidence_rate = (self.high_confidence_signals / self.signals_generated) * 100
            print(f"💪 Tasa de confianza: {confidence_rate:.1f}%")
        
        print("="*50)
    
    def _signal_handler(self, signum, frame):
        """Maneja señales del sistema para parada elegante"""
        self.logger.info(f"🛑 Señal {signum} recibida, deteniendo bot...")
        self.is_running = False

async def main():
    """Función principal"""
    print("🤖 LIVE INTRADAY TRADING BOT")
    print("💡 Trading intradía con WebSocket en tiempo real")
    print("✅ Precio de entrada óptimo")
    print("⚠️ Alertas de precaución")
    print("📊 Señales intradía")
    print("=" * 60)
    
    # Configuración del bot
    config = {
        'min_momentum_threshold': 65,        # Momentum mínimo para señal
        'max_pairs_to_monitor': 30,         # Máximo 30 pares
        'signal_cooldown_minutes': 15,      # 15 min entre señales del mismo par
        'min_confidence': 0.6,              # Confianza mínima 60%
        'enable_precaution_alerts': True,   # Activar alertas
        'enable_entry_optimization': True    # Optimizar entradas
    }
    
    # Crear y ejecutar bot
    bot = LiveIntradayTradingBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\n👋 Bot detenido por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando bot: {e}")
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
