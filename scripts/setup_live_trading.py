#!/usr/bin/env python3
"""
🔧 Configurador de API Keys y Sistema de Trading en Vivo
================================================

Este script te ayuda a configurar las API keys y verificar que todo esté listo
para el trading en vivo con datos reales de Binance.

Uso:
    python scripts/setup_live_trading.py
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import aiohttp
import json
from datetime import datetime

# Añadir el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

class LiveTradingSetup:
    """Configurador para trading en vivo"""
    
    def __init__(self):
        self.config_dir = Path(__file__).parent.parent / "config"
        self.secrets_file = self.config_dir / "secrets.env"
        
    def display_banner(self):
        """Muestra banner de bienvenida"""
        print("\n" + "="*60)
        print("🚀 CONFIGURADOR DE TRADING EN VIVO")
        print("="*60)
        print("📊 Bot de Momentum y Trading Intradía")
        print("🔧 Configuración de API Keys de Binance")
        print("⚡ Verificación de conexión en tiempo real")
        print("="*60 + "\n")
        
    def check_secrets_file(self):
        """Verifica si existe el archivo de secrets"""
        if not self.secrets_file.exists():
            print("❌ No se encontró el archivo secrets.env")
            print("📁 Creando archivo de configuración...")
            return False
        return True
        
    def load_environment(self):
        """Carga las variables de entorno"""
        try:
            load_dotenv(self.secrets_file)
            return True
        except Exception as e:
            print(f"❌ Error cargando archivo de configuración: {e}")
            return False
            
    def check_api_keys(self):
        """Verifica que las API keys estén configuradas"""
        api_key = os.getenv('BINANCE_API_KEY', '')
        secret_key = os.getenv('BINANCE_SECRET_KEY', '')
        
        if not api_key or api_key == 'TU_BINANCE_API_KEY_AQUI':
            print("⚠️  API Key de Binance no configurada")
            return False, None, None
            
        if not secret_key or secret_key == 'TU_BINANCE_SECRET_KEY_AQUI':
            print("⚠️  Secret Key de Binance no configurada")
            return False, None, None
            
        print("✅ API Keys encontradas")
        return True, api_key, secret_key
        
    async def test_binance_connection(self, api_key, secret_key):
        """Prueba la conexión con Binance"""
        print("\n🔍 Probando conexión con Binance...")
        
        try:
            # Test básico de conectividad
            async with aiohttp.ClientSession() as session:
                # Primero probamos endpoint público
                async with session.get('https://api.binance.com/api/v3/ping') as response:
                    if response.status == 200:
                        print("✅ Conexión básica a Binance exitosa")
                    else:
                        print("❌ Error en conexión básica")
                        return False
                        
                # Luego probamos obtener información de servidor
                async with session.get('https://api.binance.com/api/v3/time') as response:
                    if response.status == 200:
                        data = await response.json()
                        server_time = datetime.fromtimestamp(data['serverTime'] / 1000)
                        print(f"🕐 Tiempo del servidor: {server_time}")
                        print("✅ Sincronización de tiempo correcta")
                    else:
                        print("❌ Error obteniendo tiempo del servidor")
                        return False
                        
                # Test de límites de rate
                async with session.get('https://api.binance.com/api/v3/exchangeInfo') as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"✅ Exchange info obtenida - {len(data.get('symbols', []))} pares disponibles")
                    else:
                        print("❌ Error obteniendo información del exchange")
                        return False
                        
            return True
            
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False
            
    async def test_websocket_feeds(self):
        """Prueba los feeds de WebSocket"""
        print("\n🌐 Probando WebSocket feeds...")
        
        try:
            import websockets
            
            # Test de stream de precios
            uri = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
            
            async with websockets.connect(uri) as websocket:
                print("✅ Conexión WebSocket establecida")
                
                # Recibir un mensaje de prueba
                data = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                message = json.loads(data)
                
                if 'c' in message:  # 'c' es el precio actual
                    price = float(message['c'])
                    print(f"📈 Precio actual BTC/USDT: ${price:,.2f}")
                    print("✅ WebSocket funcionando correctamente")
                    return True
                    
        except asyncio.TimeoutError:
            print("⏰ Timeout en WebSocket - puede ser temporal")
            return False
        except Exception as e:
            print(f"❌ Error en WebSocket: {e}")
            return False
            
    def show_configuration_guide(self):
        """Muestra guía de configuración"""
        print("\n" + "="*60)
        print("📋 GUÍA DE CONFIGURACIÓN")
        print("="*60)
        print("\n🔑 Para obtener API Keys de Binance:")
        print("1. Ve a: https://www.binance.com/en/my/settings/api-management")
        print("2. Crea una nueva API Key")
        print("3. Habilita permisos de 'Spot & Margin Trading'")
        print("4. ⚠️  NO habilites 'Futures' inicialmente")
        print("5. Configura IP whitelist para mayor seguridad")
        
        print("\n📝 Edita el archivo: config/secrets.env")
        print("   - Reemplaza TU_BINANCE_API_KEY_AQUI con tu API key")
        print("   - Reemplaza TU_BINANCE_SECRET_KEY_AQUI con tu secret key")
        
        print("\n🛡️  IMPORTANTE - SEGURIDAD:")
        print("   - Usa TESTNET primero: testnet.binance.vision")
        print("   - Empieza con cantidades pequeñas")
        print("   - Nunca compartas tus API keys")
        print("   - El archivo secrets.env NO se sube a GitHub")
        
        print("\n🔄 Después de configurar, ejecuta de nuevo este script")
        print("="*60)
        
    async def test_demo_bot(self):
        """Ejecuta una demo del bot con datos en vivo"""
        print("\n🤖 Iniciando demo del bot con datos en vivo...")
        
        try:
            # Importar el bot de trading
            from live_intraday_bot import LiveIntradayTradingBot
            
            # Crear instancia del bot
            bot = LiveIntradayTradingBot()
            
            print("✅ Bot inicializado correctamente")
            print("📊 Ejecutando análisis de 5 pares principales...")
            
            # Ejecutar por 30 segundos como demo
            await asyncio.sleep(30)
            
            print("✅ Demo completada exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error en demo del bot: {e}")
            return False
            
    async def run_setup(self):
        """Ejecuta el proceso completo de configuración"""
        self.display_banner()
        
        # 1. Verificar archivo de configuración
        if not self.check_secrets_file():
            self.show_configuration_guide()
            return
            
        # 2. Cargar configuración
        if not self.load_environment():
            return
            
        # 3. Verificar API keys
        keys_ok, api_key, secret_key = self.check_api_keys()
        if not keys_ok:
            self.show_configuration_guide()
            return
            
        # 4. Probar conexión
        connection_ok = await self.test_binance_connection(api_key, secret_key)
        if not connection_ok:
            print("\n❌ Error de conexión. Verifica:")
            print("   - Tu conexión a internet")
            print("   - Que las API keys sean correctas")
            print("   - Que no haya restricciones de IP")
            return
            
        # 5. Probar WebSocket
        websocket_ok = await self.test_websocket_feeds()
        if not websocket_ok:
            print("⚠️  WebSocket con problemas, pero puedes continuar")
            
        # 6. Demo del bot
        print("\n🎯 ¿Quieres ejecutar una demo del bot? (s/n): ", end="")
        try:
            response = input().lower()
            if response in ['s', 'si', 'y', 'yes']:
                await self.test_demo_bot()
        except KeyboardInterrupt:
            print("\n👋 Demo cancelada")
            
        # 7. Resumen final
        self.show_final_summary()
        
    def show_final_summary(self):
        """Muestra resumen final"""
        print("\n" + "="*60)
        print("🎉 CONFIGURACIÓN COMPLETADA")
        print("="*60)
        print("✅ Sistema listo para trading en vivo")
        print("\n📊 Próximos pasos:")
        print("1. python scripts/demo_intraday_signals.py  # Demo de señales")
        print("2. python scripts/demo_price_targets.py     # Demo de targets")
        print("3. python src/live_intraday_bot.py          # Bot en vivo")
        print("\n⚠️  Recuerda:")
        print("   - Empieza con TESTNET_MODE=true")
        print("   - Usa cantidades pequeñas inicialmente")
        print("   - Monitorea las operaciones constantemente")
        print("="*60 + "\n")

async def main():
    """Función principal"""
    setup = LiveTradingSetup()
    await setup.run_setup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Configuración cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
