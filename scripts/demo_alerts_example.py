"""
Ejemplo con alertas de precaución activadas
Simula condiciones de mercado volátiles con warnings
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Agregar path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.strategies.intraday_trading_signals import IntradayTradingSignals, SignalType, RiskLevel
except ImportError:
    from strategies.intraday_trading_signals import IntradayTradingSignals, SignalType, RiskLevel

def generate_volatile_market_data(symbol: str = 'BTCUSDT') -> pd.DataFrame:
    """Genera datos de mercado con alta volatilidad para mostrar alertas"""
    np.random.seed(123)  # Para reproducibilidad
    
    base_price = 45000
    periods = 100
    dates = pd.date_range(start=datetime.now() - timedelta(hours=periods), 
                         periods=periods, freq='1H')
    
    # Crear volatilidad extrema y patrones que activen alertas
    returns = []
    volumes = []
    
    base_volume = 1000000
    
    for i in range(periods):
        # Simular eventos de alta volatilidad
        if i > 80:  # Últimos 20 períodos muy volátiles
            return_vol = np.random.normal(0.001, 0.08)  # 8% volatilidad horaria
            volume_mult = np.random.uniform(2, 5)  # Volumen 2-5x normal
        elif i > 60:  # Momentum building up
            return_vol = np.random.normal(0.002, 0.04)  # 4% volatilidad
            volume_mult = np.random.uniform(1.5, 3)
        else:
            return_vol = np.random.normal(0.0005, 0.02)  # Volatilidad normal
            volume_mult = np.random.uniform(0.8, 1.2)
        
        returns.append(return_vol)
        volumes.append(base_volume * volume_mult)
    
    # Calcular precios
    price_changes = np.cumprod(1 + np.array(returns))
    prices = base_price * price_changes
    
    # Crear OHLCV
    data = []
    for i in range(len(prices)):
        price = prices[i]
        vol = volumes[i]
        
        # Crear variación intrabar
        high = price * (1 + abs(returns[i]) * 0.5)
        low = price * (1 - abs(returns[i]) * 0.5)
        open_price = prices[i-1] if i > 0 else price
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': vol
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def demo_precaution_alerts():
    """Demostración específica de alertas de precaución"""
    print("🚨 DEMO: Alertas de Precaución y Warnings")
    print("💡 Simulando condiciones de mercado volátiles")
    print("=" * 70)
    
    # Crear sistema de señales
    config = {
        'min_confidence': 0.4,  # Más permisivo para mostrar alertas
        'enable_precaution_alerts': True,
        'enable_entry_optimization': True
    }
    
    trading_signals = IntradayTradingSignals(config)
    
    # Generar datos volátiles
    volatile_data = generate_volatile_market_data('BTCUSDT')
    current_price = volatile_data['close'].iloc[-1]
    
    print(f"\n🎯 ANÁLISIS: BTCUSDT")
    print(f"💰 Precio Actual: ${current_price:,.2f}")
    print(f"🌊 Datos simulados con ALTA VOLATILIDAD")
    print("-" * 70)
    
    # Generar señal con momentum alto para activar alertas
    high_momentum = 85
    signal = trading_signals.generate_trading_signal(
        'BTCUSDT', current_price, volatile_data, high_momentum
    )
    
    # Mostrar alertas detalladamente
    print("\n⚠️ ALERTAS DE PRECAUCIÓN DETECTADAS:")
    print("=" * 50)
    
    if signal.alerts:
        for i, alert in enumerate(signal.alerts, 1):
            severity_color = "🔴" if alert.severity >= 8 else "🟠" if alert.severity >= 6 else "🟡"
            print(f"\n{severity_color} ALERTA #{i} - {alert.type}")
            print(f"   📢 {alert.message}")
            print(f"   🎯 Acción Requerida: {alert.action_required}")
            print(f"   ⏰ Timeframe: {alert.timeframe}")
            print(f"   📊 Severidad: {alert.severity}/10")
    else:
        print("✅ No se detectaron alertas especiales")
    
    # Mostrar precauciones específicas
    print(f"\n🔍 PRECAUCIONES ESPECÍFICAS:")
    print("-" * 50)
    
    if signal.precautions:
        for i, precaution in enumerate(signal.precautions, 1):
            print(f"{i}. {precaution}")
    
    # Mostrar estrategia de entrada con warnings
    print(f"\n💎 ESTRATEGIA DE ENTRADA (CON PRECAUCIONES)")
    print("-" * 50)
    print(f"🎯 Precio Óptimo: ${signal.entry_strategy.optimal_price:,.2f}")
    print(f"🛡️  Stop Loss: ${signal.entry_strategy.stop_loss:,.2f}")
    print(f"📏 Tamaño Posición: {signal.entry_strategy.position_size_recommended*100:.1f}%")
    
    # Calcular distancia al stop loss
    stop_distance = abs(current_price - signal.entry_strategy.stop_loss) / current_price * 100
    if stop_distance > 5:
        print(f"⚠️  WARNING: Stop loss está {stop_distance:.1f}% alejado - Alta volatilidad")
    
    # Análisis de timing
    current_hour = datetime.now().hour
    print(f"\n⏰ ANÁLISIS DE TIMING:")
    print("-" * 50)
    print(f"🕐 Hora actual: {current_hour}:00")
    
    if current_hour < 6:
        print("🌙 HORARIO NOCTURNO: Baja liquidez, spreads amplios")
        print("⚠️  PRECAUCIÓN: Mayor slippage esperado")
    elif 9 <= current_hour <= 11:
        print("🌅 HORARIO DE APERTURA: Alta volatilidad")
        print("⚠️  PRECAUCIÓN: Movimientos bruscos posibles")
    elif 21 <= current_hour <= 23:
        print("🌏 HORARIO ASIÁTICO: Actividad moderada")
        print("💡 INFO: Menor volumen en pares occidentales")
    else:
        print("📈 HORARIO NORMAL: Condiciones estándar")
    
    # Análisis de condiciones críticas
    print(f"\n🚨 ANÁLISIS DE CONDICIONES CRÍTICAS:")
    print("-" * 50)
    
    volatility = volatile_data['close'].pct_change().std()
    volume_spike = volatile_data['volume'].iloc[-1] / volatile_data['volume'].mean()
    rsi = calculate_simple_rsi(volatile_data['close'])
    
    print(f"🌊 Volatilidad: {volatility*100:.2f}% {'🔴 EXTREMA' if volatility > 0.05 else '🟡 ALTA' if volatility > 0.03 else '🟢 NORMAL'}")
    print(f"📊 Volumen: {volume_spike:.1f}x promedio {'🔴 ANÓMALO' if volume_spike > 3 else '🟡 ALTO' if volume_spike > 2 else '🟢 NORMAL'}")
    print(f"📈 RSI: {rsi:.1f} {'🔴 SOBRECOMPRADO' if rsi > 80 else '🔴 SOBREVENDIDO' if rsi < 20 else '🟡 NEUTRAL'}")
    
    # Recomendaciones finales
    print(f"\n💡 RECOMENDACIONES FINALES:")
    print("-" * 50)
    
    if signal.signal_type == SignalType.BUY:
        print("🟢 SEÑAL DE COMPRA DETECTADA")
        print("⚠️  Pero considera las siguientes precauciones:")
        print("   • Usar stop loss ajustado por volatilidad")
        print("   • Reducir tamaño de posición por riesgo elevado")
        print("   • Monitorear de cerca los primeros 15-30 minutos")
        print("   • Tomar parciales en resistencias clave")
        
        if len(signal.alerts) > 2:
            print("🚨 ALTO RIESGO: Considera esperar mejor momento")
        
    elif signal.signal_type == SignalType.WATCH:
        print("🟡 SEÑAL DE OBSERVACIÓN")
        print("💡 Recomendaciones:")
        print("   • Esperar confirmación adicional")
        print("   • Observar reacción en niveles clave")
        print("   • Preparar entrada en retroceso")
    
    # Mostrar próximos niveles críticos
    print(f"\n🎯 NIVELES CRÍTICOS A OBSERVAR:")
    print("-" * 50)
    recent_high = volatile_data['high'].tail(20).max()
    recent_low = volatile_data['low'].tail(20).min()
    
    print(f"📈 Resistencia próxima: ${recent_high:,.2f}")
    print(f"📉 Soporte próximo: ${recent_low:,.2f}")
    print(f"⚠️  Observar reacción en estos niveles")
    
    resistance_distance = (recent_high - current_price) / current_price * 100
    support_distance = (current_price - recent_low) / current_price * 100
    
    if resistance_distance < 2:
        print(f"🚧 RESISTENCIA MUY CERCANA: {resistance_distance:.1f}% - PRECAUCIÓN MÁXIMA")
    if support_distance < 3:
        print(f"🛡️  SOPORTE CERCANO: {support_distance:.1f}% - Posible rebote")

def calculate_simple_rsi(prices, period=14):
    """Calcula RSI simple"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else 50

if __name__ == "__main__":
    demo_precaution_alerts()
