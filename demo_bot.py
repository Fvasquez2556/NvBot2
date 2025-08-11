#!/usr/bin/env python3
"""
Versión demo del bot - Solo componentes básicos sin dependencias externas
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager
from src.utils.notification_simple import SimpleNotificationSystem
from src.strategies.momentum_detector import MomentumDetector
from src.strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer

logger = get_logger(__name__)

class NvBot2Demo:
    """
    Versión demo del bot que funciona sin dependencias externas
    """
    
    def __init__(self):
        self.config = ConfigManager()
        self.is_running = False
        self.components = {}
        
    async def initialize(self):
        """Inicializa componentes básicos"""
        try:
            logger.info("🚀 Inicializando NvBot2 Demo...")
            
            # Componentes básicos que funcionan
            self.components['momentum_detector'] = MomentumDetector(self.config)
            self.components['timeframe_analyzer'] = MultiTimeframeAnalyzer(self.config)
            self.components['notification_system'] = SimpleNotificationSystem(self.config)
            
            await self.components['notification_system'].initialize()
            
            logger.info("✅ Componentes demo inicializados correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando demo: {e}")
            raise
    
    async def run_demo(self):
        """Ejecuta demo del bot con datos simulados"""
        self.is_running = True
        cycle_count = 0
        
        logger.info("🔄 Iniciando demo del bot...")
        
        # Notificación de inicio
        await self.components['notification_system'].send_alert({
            'type': 'info',
            'title': 'NvBot2 Demo Iniciado',
            'message': 'Demo ejecutándose con datos simulados',
            'data': {
                'timestamp': datetime.now().isoformat(),
                'mode': 'DEMO'
            }
        })
        
        try:
            while self.is_running and cycle_count < 10:  # Solo 10 ciclos para demo
                cycle_start = datetime.now()
                cycle_count += 1
                
                logger.info(f"🔄 Demo Ciclo #{cycle_count} - {cycle_start.strftime('%H:%M:%S')}")
                
                try:
                    # Generar datos simulados para BTC
                    btc_data = self._generate_demo_data()
                    
                    # Analizar momentum
                    momentum_result = await self.components['momentum_detector'].detect_momentum(btc_data)
                    
                    if momentum_result:
                        logger.info(f"📊 Momentum detectado: {momentum_result.direction} "
                                  f"(confianza: {momentum_result.confidence:.2f})")
                        
                        # Simular análisis multi-timeframe
                        logger.info("⏰ Ejecutando análisis multi-timeframe...")
                        
                        # Simular señal de trading
                        if momentum_result.confidence > 0.7:
                            await self._simulate_trade_signal(momentum_result)
                    
                    # Simular portfolio update
                    if cycle_count % 3 == 0:  # Cada 3 ciclos
                        await self._simulate_portfolio_summary()
                    
                    # Esperar 5 segundos entre ciclos de demo
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    logger.error(f"❌ Error en ciclo demo: {e}")
                    await asyncio.sleep(2)
                    
        except KeyboardInterrupt:
            logger.info("⏹️ Demo detenido por usuario")
        except Exception as e:
            logger.error(f"❌ Error crítico en demo: {e}")
        finally:
            await self.shutdown()
    
    def _generate_demo_data(self) -> pd.DataFrame:
        """Genera datos de mercado simulados para demo"""
        # Crear datos sintéticos que simulan BTC
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        
        # Generar 100 velas de 1 minuto
        dates = pd.date_range(start=datetime.now() - pd.Timedelta(minutes=100), 
                             end=datetime.now(), freq='1T')
        
        # Precio base que fluctúa
        base_price = 45000 + np.random.normal(0, 1000)
        prices = []
        volumes = []
        
        current_price = base_price
        for i in range(len(dates)):
            # Simular movimiento de precios
            change = np.random.normal(0, 0.005)  # 0.5% volatilidad
            current_price = current_price * (1 + change)
            
            # Simular OHLC
            open_price = current_price
            high_price = current_price * (1 + abs(np.random.normal(0, 0.002)))
            low_price = current_price * (1 - abs(np.random.normal(0, 0.002)))
            close_price = current_price
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            })
            
            # Volumen correlacionado con volatilidad
            volume = 1000 + abs(change) * 50000
            volumes.append(volume)
        
        df = pd.DataFrame(prices, index=dates)
        df['volume'] = volumes
        
        return df
    
    async def _simulate_trade_signal(self, momentum_result):
        """Simula una señal de trading"""
        action = 'BUY' if momentum_result.direction == 'bullish' else 'SELL'
        price = 45000 + np.random.normal(0, 500)  # Precio simulado
        amount = 0.01  # Cantidad simulada
        
        await self.components['notification_system'].send_trade_notification({
            'symbol': 'BTC/USDT',
            'side': action.lower(),
            'amount': amount,
            'price': price,
            'order_id': f'demo_{datetime.now().timestamp():.0f}',
            'confidence': momentum_result.confidence,
            'signal_strength': momentum_result.signal_strength
        })
    
    async def _simulate_portfolio_summary(self):
        """Simula resumen del portfolio"""
        # Generar datos de portfolio aleatorios para demo
        total_balance = 10000 + np.random.normal(0, 200)
        pnl = np.random.normal(0, 50)
        
        await self.components['notification_system'].send_portfolio_summary({
            'total_balance': total_balance,
            'available_balance': total_balance * 0.8,
            'total_pnl': pnl,
            'daily_pnl': pnl * 0.3,
            'positions_count': np.random.randint(0, 4),
            'trades_today': np.random.randint(0, 8)
        })
    
    async def shutdown(self):
        """Cierra el demo"""
        logger.info("🛑 Cerrando Demo...")
        
        self.is_running = False
        
        # Notificación de cierre
        await self.components['notification_system'].send_alert({
            'type': 'info',
            'title': 'Demo Finalizado',
            'message': 'La demostración ha terminado',
            'data': {'timestamp': datetime.now().isoformat()}
        })
        
        # Cerrar componentes
        for name, component in self.components.items():
            try:
                if hasattr(component, 'close'):
                    await component.close()
            except:
                pass
        
        logger.info("✅ Demo cerrado correctamente")

async def main():
    """Función principal del demo"""
    try:
        print("=" * 60)
        print("🎯 NVBOT2 DEMO - Trading Bot Demonstration")
        print("=" * 60)
        print("🚀 Iniciando demostración del bot de trading...")
        print("📊 Usando datos simulados (sin conexión a exchanges)")
        print("⚠️  MODO DEMO - No se ejecutan trades reales")
        print("=" * 60)
        
        # Crear y ejecutar demo
        bot = NvBot2Demo()
        await bot.initialize()
        await bot.run_demo()
        
        print("=" * 60)
        print("✅ Demo completado exitosamente!")
        print("📋 Para usar el bot completo:")
        print("   1. Instalar dependencias: pip install -r requirements.txt")
        print("   2. Configurar API keys en config/secrets.env")
        print("   3. Ejecutar: python src/main.py")
        print("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("⏹️ Demo interrumpido por usuario")
    except Exception as e:
        logger.error(f"❌ Error en demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
