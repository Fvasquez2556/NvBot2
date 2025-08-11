#!/usr/bin/env python3
"""
Demo del Predictor de Price Targets
Muestra cómo el bot calcula potencial de subida basado en momentum
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Agregar directorio src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

async def demo_price_targets():
    """Demostración de predicción de price targets"""
    print("🎯 === DEMO: PREDICCIÓN DE PRICE TARGETS ===\n")
    
    try:
        from ml_models.price_target_predictor import PriceTargetPredictor, PriceTarget
        
        # Crear predictor
        predictor = PriceTargetPredictor()
        print("✅ Predictor inicializado")
        
        # Simular datos de diferentes escenarios
        scenarios = [
            {
                'name': 'Bitcoin - Momentum Alto',
                'symbol': 'BTCUSDT',
                'current_price': 45000,
                'momentum_score': 85,
                'trend': 'bullish'
            },
            {
                'name': 'Ethereum - Momentum Medio',
                'symbol': 'ETHUSDT', 
                'current_price': 2800,
                'momentum_score': 65,
                'trend': 'neutral'
            },
            {
                'name': 'Altcoin - Momentum Bajo',
                'symbol': 'ADAUSDT',
                'current_price': 0.45,
                'momentum_score': 35,
                'trend': 'bearish'
            },
            {
                'name': 'Breakout - Momentum Extremo',
                'symbol': 'SOLUSDT',
                'current_price': 120,
                'momentum_score': 95,
                'trend': 'explosive'
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📊 === {scenario['name']} ===")
            
            # Generar datos de prueba
            market_data = generate_market_data(scenario['trend'], scenario['current_price'])
            
            # Predecir targets
            target = predictor.predict_targets(
                symbol=scenario['symbol'],
                current_price=scenario['current_price'],
                market_data=market_data,
                momentum_score=scenario['momentum_score']
            )
            
            # Mostrar resultados
            display_target_analysis(target)
        
        print("\n🎉 Demo completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()

def generate_market_data(trend: str, current_price: float) -> pd.DataFrame:
    """Genera datos de mercado sintéticos"""
    np.random.seed(42)  # Para reproducibilidad
    
    # Base price series
    if trend == 'bullish':
        price_changes = np.random.normal(0.001, 0.02, 100)  # Tendencia alcista
        volume_multiplier = np.random.uniform(1.2, 2.5, 100)  # Volumen alto
    elif trend == 'bearish':
        price_changes = np.random.normal(-0.001, 0.02, 100)  # Tendencia bajista
        volume_multiplier = np.random.uniform(0.8, 1.5, 100)  # Volumen medio
    elif trend == 'explosive':
        price_changes = np.random.normal(0.003, 0.03, 100)  # Tendencia explosiva
        volume_multiplier = np.random.uniform(2.0, 4.0, 100)  # Volumen muy alto
    else:  # neutral
        price_changes = np.random.normal(0, 0.015, 100)  # Sin tendencia
        volume_multiplier = np.random.uniform(0.9, 1.3, 100)  # Volumen normal
    
    # Generar precios
    prices = [current_price]
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices[1:])  # Remover precio inicial
    
    # Generar OHLC
    opens = prices
    closes = prices * (1 + np.random.normal(0, 0.001, 100))
    highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, 0.01, 100))
    lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, 0.01, 100))
    
    # Generar volumen
    base_volume = 1000000
    volumes = base_volume * volume_multiplier
    
    # Crear DataFrame
    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return data

def display_target_analysis(target):
    """Muestra análisis detallado de targets"""
    
    print(f"💰 Precio actual: ${target.current_price:,.2f}")
    print(f"📈 Momentum Score: {target.momentum_score:.0f}/100")
    print(f"🎯 Confianza: {target.confidence:.1%}")
    print(f"⏰ Timeframe: {target.timeframe_days} días")
    print(f"⚠️ Riesgo: {target.risk_level}")
    
    print(f"\n🎯 === TARGETS DE PRECIO ===")
    print(f"🟢 Conservador: +{target.conservative_target:.1f}% → ${target.conservative_price:,.2f}")
    print(f"🟡 Moderado:    +{target.moderate_target:.1f}% → ${target.moderate_price:,.2f}")
    print(f"🔴 Agresivo:    +{target.aggressive_target:.1f}% → ${target.aggressive_price:,.2f}")
    
    print(f"\n📊 === ANÁLISIS ===")
    print(f"✅ Probabilidad de éxito: {target.probability_success:.1%}")
    
    # Calcular potential returns
    conservative_return = target.conservative_target
    moderate_return = target.moderate_target
    aggressive_return = target.aggressive_target
    
    print(f"\n💎 === POTENCIAL DE GANANCIA ===")
    print(f"💚 Escenario conservador: +{conservative_return:.1f}%")
    print(f"💛 Escenario moderado:    +{moderate_return:.1f}%")
    print(f"❤️ Escenario agresivo:    +{aggressive_return:.1f}%")
    
    # Risk/Reward analysis
    if target.confidence > 0.7:
        recommendation = "🟢 RECOMENDADO"
    elif target.confidence > 0.5:
        recommendation = "🟡 PRECAUCIÓN"
    else:
        recommendation = "🔴 ALTO RIESGO"
    
    print(f"\n🎯 Recomendación: {recommendation}")
    
    # Technical factors summary
    tech_factors = target.technical_factors
    if tech_factors:
        print(f"\n🔧 Factores técnicos detectados:")
        if 'fibonacci_levels' in tech_factors:
            print(f"   📐 Fibonacci targets calculados")
        if 'resistance_levels' in tech_factors:
            print(f"   🚧 {len(tech_factors.get('resistance_levels', []))} niveles de resistencia")
        if 'bollinger_upper' in tech_factors:
            print(f"   📊 Bollinger Band superior: ${tech_factors['bollinger_upper']:,.2f}")

async def interactive_demo():
    """Demo interactivo para probar diferentes valores"""
    print("🎮 === DEMO INTERACTIVO ===")
    print("Introduce valores personalizados para ver predicciones\n")
    
    try:
        from ml_models.price_target_predictor import PriceTargetPredictor
        predictor = PriceTargetPredictor()
        
        # Solicitar inputs del usuario
        symbol = input("💰 Símbolo (ej: BTCUSDT): ").upper() or "BTCUSDT"
        current_price = float(input("💵 Precio actual (ej: 45000): ") or "45000")
        momentum_score = float(input("📈 Momentum Score 0-100 (ej: 75): ") or "75")
        
        print(f"\n🔄 Calculando targets para {symbol}...")
        
        # Generar datos basados en momentum
        if momentum_score > 70:
            trend = 'bullish'
        elif momentum_score < 40:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        market_data = generate_market_data(trend, current_price)
        
        # Predecir
        target = predictor.predict_targets(symbol, current_price, market_data, momentum_score)
        
        # Mostrar resultado
        print(f"\n✨ === RESULTADO PARA {symbol} ===")
        display_target_analysis(target)
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def main():
    """Función principal"""
    print("🎯 === PREDICTOR DE PRICE TARGETS ===")
    print("Sistema avanzado para calcular potencial de subida\n")
    
    # Demo automático
    await demo_price_targets()
    
    print("\n" + "="*60)
    
    # Demo interactivo
    try_interactive = input("\n🎮 ¿Quieres probar el demo interactivo? (y/N): ").lower()
    if try_interactive.startswith('y'):
        await interactive_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
