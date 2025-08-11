#!/usr/bin/env python3
"""
🤖 BOT DE TRADING INTRADÍA EN VIVO - VERSIÓN CORREGIDA
======================================================

Bot completo con:
- WebSocket en tiempo real 
- Señales de trading intradía
- Price targets
- Alertas de precaución
- Precios de entrada optimizados
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Añadir src al path
sys.path.append(str(Path(__file__).parent))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from data_sources.binance_websocket_aggregator import BinanceWebSocketAggregator
from strategies.intraday_trading_signals import IntradayTradingSignals
from ml_models.price_target_predictor import PriceTargetPredictor

# Mock config para el bot
class BotConfig:
    def get(self, key, default=None):
        configs = {
            'data_aggregator.min_volume_24h': 5000000,  # $5M+ volumen
            'data_aggregator.max_pairs': 10,  # Top 10 pares
            'data_aggregator.update_interval': 1
        }
        return configs.get(key, default)

class LiveIntradayBot:
    """Bot de trading intradía en tiempo real corregido"""
    
    def __init__(self):
        self.config = BotConfig()
        self.aggregator = None
        self.signals = IntradayTradingSignals()
        self.predictor = PriceTargetPredictor()
        self.is_running = False
        
        # Estadísticas
        self.processed_count = 0
        self.signals_generated = 0
        self.start_time = None
        
        # Cache de datos recientes
        self.price_cache = {}
        
    def display_banner(self):
        """Banner de inicio"""
        print("\n" + "="*70)
        print("🤖 LIVE INTRADAY TRADING BOT - CORREGIDO")
        print("="*70)
        print("📊 Trading intradía con datos en vivo")
        print("⚡ WebSocket en tiempo real")
        print("🎯 Señales de entrada optimizadas") 
        print("⚠️  Alertas de precaución integradas")
        print("💰 Price targets automáticos")
        print("="*70 + "\n")
        
    async def on_price_update(self, symbol, ticker_data):
        """Procesa actualizaciones de precio en tiempo real"""
        try:
            self.processed_count += 1
            
            # Extraer datos del ticker
            price = float(ticker_data.price)
            volume = float(ticker_data.volume_24h)
            change = float(ticker_data.change_24h)
            high = float(ticker_data.high_24h)
            low = float(ticker_data.low_24h)
            
            # Actualizar cache
            self.price_cache[symbol] = {
                'price': price,
                'volume': volume,
                'change': change,
                'high': high,
                'low': low,
                'timestamp': datetime.now()
            }
            
            # Mostrar precios cada 20 actualizaciones
            if self.processed_count % 20 == 0:
                self.display_price_update(symbol, price, change, volume)
            
            # Generar señal si hay suficiente volumen
            if volume > 5000000:  # $5M+
                await self.analyze_trading_opportunity(symbol, ticker_data)
                
        except Exception as e:
            logging.error(f"Error procesando {symbol}: {e}")
            
    def display_price_update(self, symbol, price, change, volume):
        """Muestra actualización de precio formateada"""
        change_color = "📈" if change >= 0 else "📉"
        print(f"{change_color} {symbol}: ${price:,.4f} ({change:+.2f}%) Vol: ${volume:,.0f}")
        
    async def analyze_trading_opportunity(self, symbol, ticker_data):
        """Analiza oportunidad de trading"""
        try:
            # Preparar datos para análisis
            market_data = {
                'symbol': symbol,
                'current_price': float(ticker_data.price),
                'volume_24h': float(ticker_data.volume_24h),
                'price_change_24h': float(ticker_data.change_24h),
                'high_24h': float(ticker_data.high_24h),
                'low_24h': float(ticker_data.low_24h),
                'timestamp': datetime.now()
            }
            
            # Generar señal de trading
            signal = await self.signals.generate_signal(market_data)
            
            if signal and signal.get('signal_strength', 0) > 65:
                self.signals_generated += 1
                await self.display_trading_signal(symbol, signal, market_data)
                
                # Calcular price target
                target = await self.predictor.calculate_price_target(market_data)
                if target:
                    self.display_price_target(symbol, target)
                    
        except Exception as e:
            logging.error(f"Error analizando {symbol}: {e}")
            
    async def display_trading_signal(self, symbol, signal, market_data):
        """Muestra señal de trading formateada"""
        print(f"\n🚨 SEÑAL DE TRADING #{self.signals_generated}")
        print(f"{'='*50}")
        print(f"📊 Par: {symbol}")
        print(f"💰 Precio actual: ${market_data['current_price']:,.4f}")
        print(f"🎯 Tipo: {signal.get('signal_type', 'N/A')}")
        print(f"💪 Fuerza: {signal.get('signal_strength', 0):.1f}%")
        print(f"⏰ Timeframe: {signal.get('timeframe', 'N/A')}")
        
        # Precios de entrada y salida
        if signal.get('entry_price'):
            print(f"🚪 Entrada: ${signal['entry_price']:,.4f}")
        if signal.get('stop_loss'):
            print(f"🛑 Stop Loss: ${signal['stop_loss']:,.4f}")
        if signal.get('take_profit_1'):
            print(f"🎯 Take Profit 1: ${signal['take_profit_1']:,.4f}")
        if signal.get('take_profit_2'):
            print(f"🎯 Take Profit 2: ${signal['take_profit_2']:,.4f}")
            
        # Alertas de precaución
        if signal.get('precaution_alerts'):
            print(f"⚠️  PRECAUCIÓN:")
            for alert in signal['precaution_alerts'][:2]:  # Mostrar solo las primeras 2
                print(f"   • {alert}")
                
        # Información adicional
        if signal.get('position_size_recommended'):
            print(f"📏 Tamaño posición: {signal['position_size_recommended']:.2%}")
        if signal.get('risk_level'):
            print(f"⚖️  Riesgo: {signal['risk_level']}")
            
        print(f"{'='*50}\n")
        
    def display_price_target(self, symbol, target):
        """Muestra price target"""
        print(f"🎯 PRICE TARGET - {symbol}")
        print(f"   📈 Target: ${target.get('target_price', 0):,.4f}")
        print(f"   📊 Upside: +{target.get('upside_percentage', 0):.1f}%")
        print(f"   🕐 Timeframe: {target.get('timeframe', 'N/A')}")
        print(f"   💯 Confianza: {target.get('confidence_score', 0):.1f}%\n")
        
    def display_statistics(self):
        """Muestra estadísticas en tiempo real"""
        if not self.start_time:
            return
            
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        print(f"\n📊 ESTADÍSTICAS ({elapsed:.0f}s)")
        print(f"   📨 Actualizaciones procesadas: {self.processed_count}")
        print(f"   🚨 Señales generadas: {self.signals_generated}")
        print(f"   📈 Pares activos: {len(self.price_cache)}")
        print(f"   ⚡ Rate: {self.processed_count/elapsed:.1f} updates/seg")
        
        # Mostrar top 3 pares por volumen
        if self.price_cache:
            sorted_pairs = sorted(self.price_cache.items(), 
                                key=lambda x: x[1]['volume'], reverse=True)[:3]
            print(f"   🔥 Top pares por volumen:")
            for symbol, data in sorted_pairs:
                print(f"      {symbol}: ${data['volume']:,.0f}")
        print()
        
    async def run(self, duration=300):  # 5 minutos por defecto
        """Ejecuta el bot por un tiempo determinado"""
        self.display_banner()
        
        print(f"🕐 Ejecutando por {duration} segundos...")
        print("📡 Conectando a datos en tiempo real...")
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            
            # Inicializar aggregator
            self.aggregator = BinanceWebSocketAggregator(self.config)
            self.aggregator.register_callback('ticker_update', self.on_price_update)
            
            # Conectar
            await self.aggregator.initialize()
            print("✅ Conectado! Monitoreando mercado...\n")
            
            # Loop principal con estadísticas
            for i in range(duration):
                await asyncio.sleep(1)
                
                # Mostrar estadísticas cada 30 segundos
                if i > 0 and i % 30 == 0:
                    self.display_statistics()
                    
        except KeyboardInterrupt:
            print("\n⏹️  Bot detenido por usuario")
        except Exception as e:
            print(f"\n❌ Error en bot: {e}")
            logging.error(f"Error en bot: {e}")
        finally:
            await self.stop()
            
    async def stop(self):
        """Detiene el bot"""
        self.is_running = False
        
        if self.aggregator:
            try:
                await self.aggregator.close()
                print("✅ WebSocket cerrado")
            except Exception as e:
                logging.error(f"Error cerrando aggregator: {e}")
                
        # Mostrar estadísticas finales
        self.display_final_summary()
        
    def display_final_summary(self):
        """Muestra resumen final"""
        print("\n" + "="*70)
        print("📋 RESUMEN DE SESIÓN")
        print("="*70)
        
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            print(f"⏰ Duración: {elapsed:.0f} segundos")
            print(f"📊 Total actualizaciones: {self.processed_count}")
            print(f"🚨 Señales generadas: {self.signals_generated}")
            print(f"📈 Rate promedio: {self.processed_count/elapsed:.1f} updates/seg")
            
        print(f"✅ Bot ejecutado exitosamente")
        print("="*70 + "\n")

async def main():
    """Función principal"""
    bot = LiveIntradayBot()
    
    print("🤖 ¿Cuánto tiempo quieres ejecutar el bot?")
    print("1. Demo rápida (60 segundos)")
    print("2. Sesión corta (5 minutos)")
    print("3. Sesión media (15 minutos)")
    print("4. Sesión larga (30 minutos)")
    
    try:
        choice = input("\nElige una opción (1-4) o presiona Enter para demo: ").strip()
        
        duration_map = {
            '1': 60,
            '2': 300,
            '3': 900,
            '4': 1800
        }
        
        duration = duration_map.get(choice, 60)  # Por defecto 1 minuto
        print(f"\n🚀 Iniciando sesión de {duration} segundos...")
        
        await bot.run(duration)
        
    except KeyboardInterrupt:
        print("\n👋 Bot cancelado por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Hasta luego!")
