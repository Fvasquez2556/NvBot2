#!/usr/bin/env python3
"""
Demo script para probar el agregador WebSocket avanzado
Muestra datos en tiempo real de TODOS los pares USDT de Binance
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_sources.binance_websocket_aggregator import BinanceWebSocketAggregator, TickerData, KlineData

class WebSocketDemo:
    """Demo del agregador WebSocket"""
    
    def __init__(self):
        self.aggregator = None
        self.received_tickers = 0
        self.received_klines = 0
        self.active_pairs = set()
        
    async def initialize(self):
        """Inicializar el agregador"""
        print("🚀 Inicializando demo WebSocket...")
        
        # Importar ConfigManager
        from utils.config_manager import ConfigManager
        
        # Crear configuración temporal para demo
        config = ConfigManager()
        
        # Crear agregador con configuración demo
        self.aggregator = BinanceWebSocketAggregator(config)
        
        # Configurar callbacks
        self.aggregator.register_callback('ticker', self.on_ticker_update)
        self.aggregator.register_callback('kline', self.on_kline_update)
        self.aggregator.register_callback('error', self.on_error)
        
        print("✅ Demo inicializado")
    
    async def on_ticker_update(self, data):
        """Callback para datos de ticker"""
        self.received_tickers += 1
        self.active_pairs.add("ticker_symbol")  # Placeholder
        
        # Mostrar cada 100 tickers
        if self.received_tickers % 100 == 0:
            print(f"📊 Ticker #{self.received_tickers} recibido")
    
    async def on_kline_update(self, data):
        """Callback para datos de klines"""
        self.received_klines += 1
        
        # Mostrar cada 50 klines
        if self.received_klines % 50 == 0:
            print(f"📈 Kline #{self.received_klines} recibido")
    
    async def on_error(self, error):
        """Callback para errores"""
        print(f"❌ Error: {error}")
    
    async def run_demo(self, duration_seconds: int = 300):
        """Ejecutar demo por un tiempo determinado"""
        print(f"🔄 Iniciando demo por {duration_seconds} segundos...")
        
        # Inicializar agregador
        await self.aggregator.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            while True:
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                # Mostrar estadísticas cada 30 segundos
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    await self.show_statistics(elapsed)
                
                # Terminar después del tiempo especificado
                if elapsed > duration_seconds:
                    break
                
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrumpido por usuario")
        
        finally:
            print("🛑 Cerrando demo...")
            await self.aggregator.close()
            print("✅ Demo finalizado")
    
    async def show_statistics(self, elapsed_time: float):
        """Mostrar estadísticas del demo"""
        
        print(f"\n📊 === ESTADÍSTICAS DEMO ({elapsed_time:.0f}s) ===")
        print(f"Pares descubiertos: {len(self.aggregator.usdt_pairs)}")
        print(f"Tickers recibidos: {self.received_tickers}")
        print(f"Klines recibidas: {self.received_klines}")
        print(f"Websockets activos: {self.aggregator.stats.get('websockets_connected', 0)}")
        
        # Mostrar algunos pares ejemplo si están disponibles
        if self.aggregator.usdt_pairs:
            sample_pairs = list(self.aggregator.usdt_pairs)[:5]
            print(f"🏆 Pares ejemplos: {', '.join(sample_pairs)}")
        
        print("=" * 50)

async def main():
    """Función principal del demo"""
    try:
        # Configurar política de eventos para Windows
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        print("🌟 Demo WebSocket Aggregator - Binance USDT Pairs")
        print("Este demo muestra datos en tiempo real de Binance")
        print("Presiona Ctrl+C para detener\n")
        
        # Crear y ejecutar demo
        demo = WebSocketDemo()
        await demo.initialize()
        
        # Ejecutar por 5 minutos por defecto
        await demo.run_demo(300)
        
    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
