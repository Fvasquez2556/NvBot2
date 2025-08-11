#!/usr/bin/env python3
"""
Script de prueba rápida del sistema WebSocket
Verifica que todos los componentes funcionen correctamente
"""

import asyncio
import sys
from pathlib import Path

# Agregar directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

async def test_imports():
    """Probar que todas las importaciones funcionen"""
    print("🔍 Probando importaciones...")
    
    try:
        from data_sources.binance_websocket_aggregator import BinanceWebSocketAggregator, TickerData, KlineData
        print("✅ BinanceWebSocketAggregator importado")
        
        from utils.config_manager import ConfigManager
        print("✅ ConfigManager importado")
        
        from utils.logger import get_logger
        print("✅ Logger importado")
        
        print("✅ Todas las importaciones exitosas")
        return True
        
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        return False

async def test_config():
    """Probar configuración"""
    print("\n📋 Probando configuración...")
    
    try:
        from utils.config_manager import ConfigManager
        config = ConfigManager()
        
        # Probar configuraciones clave
        analyze_all = config.get('bot.analyze_all_usdt_pairs', False)
        min_volume = config.get('data_aggregator.min_volume_24h', 1000000)
        max_pairs = config.get('data_aggregator.max_pairs', 200)
        
        print(f"✅ Analizar todos los pares: {analyze_all}")
        print(f"✅ Volumen mínimo: ${min_volume:,}")
        print(f"✅ Máximo pares: {max_pairs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

async def test_websocket_basic():
    """Prueba básica del WebSocket aggregator"""
    print("\n🌐 Probando WebSocket aggregator...")
    
    try:
        from data_sources.binance_websocket_aggregator import BinanceWebSocketAggregator
        from utils.config_manager import ConfigManager
        
        # Crear config para la prueba
        config = ConfigManager()
        
        # Crear aggregator con configuración
        aggregator = BinanceWebSocketAggregator(config)
        
        print("✅ Agregador creado")
        
        # Configurar callbacks de prueba
        received_data = {'tickers': 0, 'klines': 0}
        
        def test_ticker_callback(data):
            received_data['tickers'] += 1
            if received_data['tickers'] <= 3:  # Solo mostrar los primeros 3
                print(f"📊 Ticker recibido #{received_data['tickers']}")
        
        def test_kline_callback(data):
            received_data['klines'] += 1
            if received_data['klines'] <= 3:  # Solo mostrar las primeras 3
                print(f"📈 Kline recibido #{received_data['klines']}")
        
        def test_error_callback(error):
            print(f"⚠️ Error: {error}")
        
        # Registrar callbacks
        aggregator.register_callback('ticker', test_ticker_callback)
        aggregator.register_callback('kline', test_kline_callback)
        aggregator.register_callback('error', test_error_callback)
        
        print("✅ Callbacks registrados")
        
        # Inicializar aggregator
        print("🔧 Inicializando agregador...")
        await aggregator.initialize()
        
        print("✅ Agregador inicializado")
        
        # Mostrar estadísticas básicas
        print(f"📊 Pares USDT descubiertos: {len(aggregator.usdt_pairs)}")
        if aggregator.usdt_pairs:
            sample_pairs = list(aggregator.usdt_pairs)[:5]
            print(f"� Ejemplos: {', '.join(sample_pairs)}")
        
        print("✅ Prueba WebSocket exitosa")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba WebSocket: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal de prueba"""
    print("🧪 === PRUEBA RÁPIDA DEL SISTEMA WEBSOCKET ===\n")
    
    # Configurar política de eventos para Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    tests_passed = 0
    total_tests = 3
    
    # Ejecutar pruebas
    if await test_imports():
        tests_passed += 1
    
    if await test_config():
        tests_passed += 1
    
    if await test_websocket_basic():
        tests_passed += 1
    
    # Resultado final
    print(f"\n🎯 === RESULTADO FINAL ===")
    print(f"Pruebas pasadas: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        print("✅ El sistema WebSocket está listo para usar")
        print("\n📚 Próximos pasos:")
        print("1. Ejecutar demo completo: python scripts/websocket_demo.py")
        print("2. Iniciar bot avanzado: python src/main_advanced.py")
        return 0
    else:
        print("❌ Algunas pruebas fallaron")
        print("🔧 Revisa los errores anteriores y la configuración")
        return 1

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(result)
    except KeyboardInterrupt:
        print("\n⏹️ Prueba interrumpida por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        sys.exit(1)
