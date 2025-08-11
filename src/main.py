#!/usr/bin/env python3
"""
Archivo principal del bot de predicción de momentum - NvBot2
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent))

from utils.logger import get_logger
from utils.config_manager import ConfigManager
from utils.notification_system import NotificationSystem
from strategies.momentum_predictor_strategy import MomentumPredictorStrategy
from strategies.momentum_detector import MomentumDetector
from strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from data_sources.data_aggregator import DataAggregator
from live_trading.portfolio_manager import PortfolioManager
from ml_models.ml_predictor import MLPredictor

logger = get_logger(__name__)

class NvBot2:
    """
    Bot principal que integra todos los componentes
    """
    
    def __init__(self):
        self.config = ConfigManager()
        self.is_running = False
        self.components = {}
        
    async def initialize(self):
        """Inicializa todos los componentes del bot"""
        try:
            logger.info("🚀 Inicializando NvBot2...")
            
            # Inicializar componentes principales
            self.components['data_aggregator'] = DataAggregator(self.config)
            await self.components['data_aggregator'].initialize()
            
            self.components['portfolio_manager'] = PortfolioManager(self.config)
            await self.components['portfolio_manager'].initialize()
            
            self.components['ml_predictor'] = MLPredictor(self.config)
            
            self.components['momentum_detector'] = MomentumDetector(self.config)
            
            self.components['timeframe_analyzer'] = MultiTimeframeAnalyzer(self.config)
            
            self.components['main_strategy'] = MomentumPredictorStrategy(self.config)
            
            self.components['notification_system'] = NotificationSystem(self.config)
            await self.components['notification_system'].initialize()
            
            # Entrenar modelos ML si es necesario
            await self._train_ml_models_if_needed()
            
            logger.info("✅ Todos los componentes inicializados correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando bot: {e}")
            raise
    
    async def _train_ml_models_if_needed(self):
        """Entrena modelos ML si no existen o son muy antiguos"""
        try:
            symbols = self.config.get('bot.trading_pairs', ['BTC/USDT'])
            
            for symbol in symbols:
                logger.info(f"📊 Verificando modelos ML para {symbol}")
                
                # Obtener datos históricos
                historical_data = await self.components['data_aggregator'].get_historical_data(
                    symbol, '1h', 2000
                )
                
                if len(historical_data) > 500:  # Suficientes datos
                    # Entrenar modelos
                    performances = self.components['ml_predictor'].train_models(
                        historical_data, symbol
                    )
                    
                    if performances:
                        best_model = max(performances.keys(), 
                                       key=lambda x: performances[x].accuracy_score)
                        best_accuracy = performances[best_model].accuracy_score
                        
                        logger.info(f"🎯 Modelo entrenado para {symbol}: "
                                  f"Mejor = {best_model} ({best_accuracy:.1f}% accuracy)")
                    else:
                        logger.warning(f"⚠️ No se pudieron entrenar modelos para {symbol}")
                else:
                    logger.warning(f"⚠️ Pocos datos históricos para {symbol} ({len(historical_data)} velas)")
                    
        except Exception as e:
            logger.error(f"❌ Error entrenando modelos ML: {e}")

    async def run(self):
        """Bucle principal del bot"""
        self.is_running = True
        cycle_count = 0
        
        logger.info("🔄 Iniciando bucle principal del bot...")
        
        # Notificación de inicio
        await self.components['notification_system'].send_alert({
            'type': 'info',
            'title': 'NvBot2 Iniciado',
            'message': 'El bot ha comenzado operaciones',
            'data': {
                'timestamp': datetime.now().isoformat(),
                'version': self.config.get('bot.version', '1.0.0')
            }
        })
        
        try:
            while self.is_running:
                cycle_start = datetime.now()
                cycle_count += 1
                
                logger.info(f"🔄 Ciclo #{cycle_count} - {cycle_start.strftime('%H:%M:%S')}")
                
                try:
                    # 1. Obtener datos de mercado
                    market_data = await self.components['data_aggregator'].get_latest_data()
                    ticker_data = await self.components['data_aggregator'].get_ticker_data()
                    
                    if not market_data:
                        logger.warning("⚠️ No se obtuvieron datos de mercado")
                        await asyncio.sleep(30)
                        continue
                    
                    # 2. Actualizar precios en portfolio
                    current_prices = {
                        symbol: ticker.last_price 
                        for symbol, ticker in ticker_data.items()
                    }
                    await self.components['portfolio_manager'].update_positions(current_prices)
                    
                    # 3. Generar señales de trading
                    all_signals = []
                    
                    for symbol, data in market_data.items():
                        if len(data) < 10:  # Datos insuficientes
                            continue
                            
                        try:
                            # Convertir a DataFrame para análisis
                            df = self._convert_to_dataframe(data)
                            
                            # Ejecutar estrategia principal
                            signals = await self.components['main_strategy'].analyze_symbol(
                                symbol, df, current_prices.get(symbol, 0)
                            )
                            
                            if signals:
                                all_signals.extend(signals)
                                
                        except Exception as e:
                            logger.error(f"❌ Error analizando {symbol}: {e}")
                    
                    # 4. Ejecutar señales
                    if all_signals:
                        logger.info(f"📈 Ejecutando {len(all_signals)} señales")
                        
                        results = await self.components['portfolio_manager'].execute_signals(all_signals)
                        
                        # Notificar trades ejecutados
                        successful_trades = [r for r in results if r.success]
                        if successful_trades:
                            await self._notify_trades(successful_trades)
                    
                    # 5. Enviar resumen periódico
                    if cycle_count % 60 == 0:  # Cada hora (si ciclo = 1 min)
                        await self._send_periodic_summary()
                    
                    # 6. Esperar próximo ciclo
                    cycle_interval = self.config.get("bot.cycle_interval", 60)
                    await asyncio.sleep(cycle_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Error en ciclo del bot: {e}")
                    await asyncio.sleep(30)  # Esperar menos tiempo en caso de error
                    
        except KeyboardInterrupt:
            logger.info("⏹️ Bot detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error crítico en bot: {e}")
            raise
        finally:
            await self.shutdown()
    
    def _convert_to_dataframe(self, market_data: List) -> pd.DataFrame:
        """Convierte datos de mercado a DataFrame"""
        data = []
        for candle in market_data:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()
        
        return df
    
    async def _notify_trades(self, successful_trades):
        """Notifica trades ejecutados"""
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
                    'order_id': trade.order_id
                }
            })
    
    async def _send_periodic_summary(self):
        """Envía resumen periódico del portfolio"""
        try:
            summary = self.components['portfolio_manager'].get_portfolio_summary()
            
            await self.components['notification_system'].send_alert({
                'type': 'summary',
                'title': 'Resumen del Portfolio',
                'message': f"Balance: ${summary['total_balance']:.2f} | PnL: ${summary['total_pnl']:.2f}",
                'data': summary
            })
            
        except Exception as e:
            logger.error(f"Error enviando resumen: {e}")
    
    async def shutdown(self):
        """Cierra todos los componentes del bot"""
        logger.info("🛑 Cerrando NvBot2...")
        
        self.is_running = False
        
        # Cerrar componentes
        for name, component in self.components.items():
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
                    'title': 'NvBot2 Detenido',
                    'message': 'El bot ha finalizado operaciones',
                    'data': {'timestamp': datetime.now().isoformat()}
                })
            except:
                pass
        
        logger.info("✅ NvBot2 cerrado correctamente")

async def main():
    """Función principal del bot"""
    try:
        # Crear e inicializar bot
        bot = NvBot2()
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
