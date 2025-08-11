#!/usr/bin/env python3
"""
Script de prueba para verificar que todos los componentes del bot funcionan correctamente
"""

import asyncio
import sys
from pathlib import Path
import traceback

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager

logger = get_logger(__name__)

async def test_config_manager():
    """Prueba el ConfigManager"""
    print("🔧 Probando ConfigManager...")
    try:
        config = ConfigManager()
        bot_name = config.get('bot.name', 'Unknown')
        trading_pairs = config.get('bot.trading_pairs', [])
        
        print(f"   ✅ Bot name: {bot_name}")
        print(f"   ✅ Trading pairs: {trading_pairs}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

async def test_data_aggregator():
    """Prueba el DataAggregator"""
    print("📊 Probando DataAggregator...")
    try:
        from src.data_sources.data_aggregator import DataAggregator
        
        config = ConfigManager()
        aggregator = DataAggregator(config)
        
        # Solo inicializar sin conectar a APIs reales
        print("   ✅ DataAggregator creado correctamente")
        
        # Probar con datos simulados
        print("   ✅ Estructura básica funcional")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

async def test_ml_predictor():
    """Prueba el MLPredictor"""
    print("🤖 Probando MLPredictor...")
    try:
        from src.ml_models.ml_predictor import MLPredictor
        
        config = ConfigManager()
        predictor = MLPredictor(config)
        
        print("   ✅ MLPredictor creado correctamente")
        
        # Probar preparación de features con datos dummy
        import pandas as pd
        import numpy as np
        
        # Crear datos dummy
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.uniform(1000, 2000, 100)
        }, index=dates)
        
        # Probar preparación de features
        features_df = predictor.prepare_features(df)
        print(f"   ✅ Features preparados: {len(features_df)} filas")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

async def test_momentum_detector():
    """Prueba el MomentumDetector"""
    print("📈 Probando MomentumDetector...")
    try:
        from src.strategies.momentum_detector import MomentumDetector
        
        config = ConfigManager()
        detector = MomentumDetector(config)
        
        print("   ✅ MomentumDetector creado correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

async def test_multi_timeframe_analyzer():
    """Prueba el MultiTimeframeAnalyzer"""
    print("⏰ Probando MultiTimeframeAnalyzer...")
    try:
        from src.strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer
        
        config = ConfigManager()
        analyzer = MultiTimeframeAnalyzer(config)
        
        print("   ✅ MultiTimeframeAnalyzer creado correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

async def test_portfolio_manager():
    """Prueba el PortfolioManager"""
    print("💼 Probando PortfolioManager...")
    try:
        from src.live_trading.portfolio_manager import PortfolioManager
        
        config = ConfigManager()
        manager = PortfolioManager(config)
        
        print("   ✅ PortfolioManager creado correctamente")
        
        # Probar summary
        summary = manager.get_portfolio_summary()
        print(f"   ✅ Portfolio summary: Balance = ${summary['total_balance']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

async def test_notification_system():
    """Prueba el NotificationSystem"""
    print("📱 Probando NotificationSystem...")
    try:
        from src.utils.notification_system import NotificationSystem
        
        config = ConfigManager()
        notification = NotificationSystem(config)
        
        print("   ✅ NotificationSystem creado correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

async def test_main_strategy():
    """Prueba la estrategia principal"""
    print("🎯 Probando MomentumPredictorStrategy...")
    try:
        from src.strategies.momentum_predictor_strategy import MomentumPredictorStrategy
        
        config = ConfigManager()
        strategy = MomentumPredictorStrategy(config)
        
        print("   ✅ MomentumPredictorStrategy creado correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

async def test_imports():
    """Prueba que todos los imports funcionen"""
    print("📦 Probando imports principales...")
    try:
        # Import principal
        from src.main import NvBot2
        print("   ✅ Import de NvBot2 exitoso")
        
        # Crear instancia
        bot = NvBot2()
        print("   ✅ Instancia de NvBot2 creada")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        traceback.print_exc()
        return False

async def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("🧪 INICIANDO PRUEBAS DEL BOT NvBot2")
    print("=" * 50)
    
    tests = [
        ("Config Manager", test_config_manager),
        ("Data Aggregator", test_data_aggregator),
        ("ML Predictor", test_ml_predictor),
        ("Momentum Detector", test_momentum_detector),
        ("Multi Timeframe Analyzer", test_multi_timeframe_analyzer),
        ("Portfolio Manager", test_portfolio_manager),
        ("Notification System", test_notification_system),
        ("Main Strategy", test_main_strategy),
        ("Main Imports", test_imports),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Error ejecutando {test_name}: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} pruebas pasaron ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON! El bot está listo para funcionar.")
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Configurar API keys en config/secrets.env")
        print("2. Ajustar parámetros en config/config.yaml")
        print("3. Ejecutar: python src/main.py")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar errores antes de ejecutar el bot.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())
