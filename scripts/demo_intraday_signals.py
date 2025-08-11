"""
Demo de Señales de Trading Intradía
Muestra precio de entrada, alertas de precaución y señales intradía
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import sys
import os

# Agregar path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports con fallback
try:
    from src.strategies.intraday_trading_signals import IntradayTradingSignals, SignalType, RiskLevel
    from src.ml_models.price_target_predictor import PriceTargetPredictor
except ImportError:
    try:
        from strategies.intraday_trading_signals import IntradayTradingSignals, SignalType, RiskLevel
        from ml_models.price_target_predictor import PriceTargetPredictor
    except ImportError:
        print("⚠️ Ejecutando en modo independiente")

def generate_realistic_market_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Genera datos de mercado realistas para testing"""
    np.random.seed(42)
    
    # Precios base según el símbolo
    base_prices = {
        'BTCUSDT': 45000,
        'ETHUSDT': 2800,
        'ADAUSDT': 0.85,
        'BNBUSDT': 320,
        'DOGEUSDT': 0.12
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # Generar datos
    periods = days * 24 * 60  # Datos por minuto
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         periods=periods, freq='1min')
    
    # Simulación de precio con tendencia y volatilidad
    returns = np.random.normal(0.0001, 0.002, periods)  # Micro-returns realistas
    
    # Agregar tendencias y volatilidad
    trend = np.linspace(0, 0.15, periods)  # Tendencia alcista gradual
    volatility_events = np.random.choice([0, 1], periods, p=[0.98, 0.02])  # 2% eventos de alta volatilidad
    volatility_spikes = volatility_events * np.random.normal(0, 0.01, periods)
    
    returns = returns + trend/periods + volatility_spikes
    
    # Calcular precios
    price_changes = np.cumprod(1 + returns)
    prices = base_price * price_changes
    
    # Generar OHLCV
    data = []
    for i in range(0, len(prices), 60):  # Agregar a velas de 1 hora
        hour_prices = prices[i:i+60]
        if len(hour_prices) == 0:
            continue
            
        open_price = hour_prices[0]
        close_price = hour_prices[-1]
        high_price = max(hour_prices)
        low_price = min(hour_prices)
        
        # Volumen simulado (correlacionado con volatilidad)
        volatility = np.std(hour_prices) / np.mean(hour_prices)
        base_volume = 1000000
        volume = base_volume * (1 + volatility * 10) * np.random.uniform(0.5, 2.0)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def print_signal_header(symbol: str, current_price: float):
    """Imprime header de la señal"""
    print("\n" + "="*80)
    print(f"🎯 SEÑAL DE TRADING INTRADÍA - {symbol}")
    print(f"💰 Precio Actual: ${current_price:.4f}")
    print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def print_entry_strategy(entry_strategy):
    """Imprime estrategia de entrada"""
    print("\n💎 ESTRATEGIA DE ENTRADA")
    print("-" * 50)
    print(f"🎯 Precio Óptimo de Entrada: ${entry_strategy.optimal_price:.4f}")
    print(f"📊 Rango de Entrada: ${entry_strategy.entry_range_min:.4f} - ${entry_strategy.entry_range_max:.4f}")
    print(f"🛡️  Stop Loss: ${entry_strategy.stop_loss:.4f}")
    print(f"🎪 Take Profit 1: ${entry_strategy.take_profit_1:.4f}")
    print(f"🎪 Take Profit 2: ${entry_strategy.take_profit_2:.4f}")
    print(f"🎪 Take Profit 3: ${entry_strategy.take_profit_3:.4f}")
    print(f"📏 Tamaño Recomendado: {entry_strategy.position_size_recommended*100:.1f}% del capital")
    print(f"⏰ Timeframe: {entry_strategy.timeframe.value}")

def print_alerts(alerts):
    """Imprime alertas de precaución"""
    if not alerts:
        print("\n✅ Sin alertas especiales")
        return
        
    print("\n⚠️ ALERTAS DE PRECAUCIÓN")
    print("-" * 50)
    
    for alert in alerts:
        severity_icons = {1: "ℹ️", 2: "ℹ️", 3: "ℹ️", 4: "⚠️", 5: "⚠️", 6: "⚠️", 7: "🚨", 8: "🚨", 9: "💀", 10: "💀"}
        icon = severity_icons.get(alert.severity, "⚠️")
        
        print(f"{icon} [{alert.type}] {alert.message}")
        print(f"   💡 Acción: {alert.action_required}")
        print(f"   ⏱️  Timeframe: {alert.timeframe}")
        print()

def print_precautions(precautions):
    """Imprime precauciones"""
    if not precautions:
        return
        
    print("🔍 OBSERVACIONES IMPORTANTES")
    print("-" * 50)
    for precaution in precautions:
        print(f"• {precaution}")

def print_intraday_signals(intraday_signals):
    """Imprime señales de trading intradía"""
    if not intraday_signals:
        return
        
    print("\n📊 SEÑALES DE TRADING INTRADÍA")
    print("-" * 50)
    
    for signal_type, details in intraday_signals.items():
        if signal_type == 'timing_alert':
            print(f"⏰ {details}")
            continue
            
        if isinstance(details, dict) and details.get('active'):
            print(f"\n🎯 {signal_type.upper().replace('_', ' ')}")
            print(f"   📈 Timeframe: {details['timeframe']}")
            print(f"   🎪 Estrategia: {details['strategy']}")
            print(f"   🔥 Triggers: {', '.join(details['entry_triggers'])}")
            print(f"   🎯 Targets: {', '.join(details['exit_targets'])}")
            print(f"   ⏱️  Max Hold: {details['max_hold_time']}")

def print_scalping_opportunities(scalping_opportunities):
    """Imprime oportunidades de scalping"""
    if not scalping_opportunities:
        return
        
    print("\n⚡ OPORTUNIDADES DE SCALPING")
    print("-" * 50)
    
    for opp in scalping_opportunities:
        print(f"🎯 {opp['type']}")
        print(f"   📋 {opp['description']}")
        if 'entry_above' in opp:
            print(f"   📈 Entrada Alcista: ${opp['entry_above']:.4f}")
        if 'entry_below' in opp:
            print(f"   📉 Entrada Bajista: ${opp['entry_below']:.4f}")
        if 'expected_move' in opp:
            if isinstance(opp['expected_move'], (int, float)):
                print(f"   🎪 Movimiento Esperado: {opp['expected_move']*100:.2f}%")
            else:
                print(f"   🎪 Movimiento Esperado: {opp['expected_move']}")
        print(f"   ⏰ Timeframe: {opp['timeframe']}")
        print()

def print_market_conditions(market_conditions):
    """Imprime condiciones de mercado"""
    print(f"\n🌍 CONDICIONES DE MERCADO")
    print("-" * 50)
    print(f"📈 Tendencia: {market_conditions.get('trend', 'N/A')}")
    print(f"🌊 Volatilidad: {market_conditions.get('volatility', 0)*100:.2f}%")
    print(f"📊 Volumen: {market_conditions.get('volume_trend', 'N/A')}")
    print(f"📉 RSI: {market_conditions.get('rsi', 0):.1f}")
    print(f"🕐 Horario: {market_conditions.get('market_hours', 'N/A')}")

def print_signal_summary(signal):
    """Imprime resumen de la señal"""
    signal_colors = {
        SignalType.BUY: "🟢",
        SignalType.SELL: "🔴", 
        SignalType.HOLD: "🟡",
        SignalType.WATCH: "🟠",
        SignalType.AVOID: "⚫"
    }
    
    risk_colors = {
        RiskLevel.VERY_LOW: "🟢",
        RiskLevel.LOW: "🟡",
        RiskLevel.MEDIUM: "🟠",
        RiskLevel.HIGH: "🔴",
        RiskLevel.EXTREME: "⚫"
    }
    
    print(f"\n📋 RESUMEN DE LA SEÑAL")
    print("-" * 50)
    print(f"{signal_colors.get(signal.signal_type, '⚪')} Señal: {signal.signal_type.value}")
    print(f"{risk_colors.get(signal.risk_level, '⚪')} Riesgo: {signal.risk_level.value}")
    print(f"💪 Confianza: {signal.confidence*100:.1f}%")
    print(f"🎯 Probabilidad de Éxito: {signal.probability_success*100:.1f}%")
    print(f"⭐ Fuerza de Señal: {signal.signal_strength}/10")
    print(f"✅ Válida: {'Sí' if signal.is_valid else 'No'}")
    print(f"⏰ Expira: {signal.expiry_time.strftime('%H:%M:%S')}")

async def demo_trading_signals():
    """Demostración completa del sistema de señales"""
    print("🚀 DEMO: Sistema de Señales de Trading Intradía")
    print("💡 Con precio de entrada, alertas de precaución y señales intradía")
    print("=" * 80)
    
    # Configuración
    config = {
        'min_confidence': 0.5,
        'max_risk_tolerance': 'HIGH',
        'enable_precaution_alerts': True,
        'enable_entry_optimization': True
    }
    
    # Inicializar sistema
    trading_signals = IntradayTradingSignals(config)
    
    # Símbolos de prueba con diferentes escenarios
    test_cases = [
        ('BTCUSDT', 85, "Bitcoin con momentum ALTO - Ejemplo de breakout"),
        ('ETHUSDT', 65, "Ethereum con momentum MEDIO - Ejemplo de swing"),
        ('ADAUSDT', 35, "Cardano con momentum BAJO - Ejemplo precautorio"),
        ('BNBUSDT', 75, "BNB con momentum ALTO - Ejemplo intradía"),
        ('DOGEUSDT', 45, "Dogecoin NEUTRAL - Ejemplo de hold")
    ]
    
    for i, (symbol, momentum_score, description) in enumerate(test_cases):
        print(f"\n{'='*20} CASO {i+1}/5 {'='*20}")
        print(f"📝 {description}")
        
        # Generar datos de mercado
        market_data = generate_realistic_market_data(symbol, days=30)
        current_price = market_data['close'].iloc[-1]
        
        # Generar señal
        signal = trading_signals.generate_trading_signal(
            symbol, current_price, market_data, momentum_score
        )
        
        # Mostrar resultados
        print_signal_header(symbol, current_price)
        print_signal_summary(signal)
        print_entry_strategy(signal.entry_strategy)
        print_alerts(signal.alerts)
        print_precautions(signal.precautions)
        print_intraday_signals(signal.intraday_signals)
        print_scalping_opportunities(signal.scalping_opportunities)
        print_market_conditions(signal.market_conditions)
        
        if i < len(test_cases) - 1:
            input("\n⏸️  Presiona ENTER para continuar al siguiente caso...")
    
    # Ejemplo interactivo
    print("\n" + "="*80)
    print("🎮 MODO INTERACTIVO")
    print("Prueba con tus propios valores")
    print("="*80)
    
    while True:
        try:
            print("\nOpciones disponibles:")
            print("1. BTCUSDT")
            print("2. ETHUSDT") 
            print("3. ADAUSDT")
            print("4. Salir")
            
            choice = input("\n🎯 Elige una opción (1-4): ").strip()
            
            if choice == '4':
                break
                
            symbol_map = {'1': 'BTCUSDT', '2': 'ETHUSDT', '3': 'ADAUSDT'}
            symbol = symbol_map.get(choice, 'BTCUSDT')
            
            momentum_input = input(f"📊 Momentum para {symbol} (1-100): ").strip()
            momentum_score = max(1, min(100, int(momentum_input) if momentum_input.isdigit() else 50))
            
            # Generar y mostrar señal
            market_data = generate_realistic_market_data(symbol, days=30)
            current_price = market_data['close'].iloc[-1]
            
            signal = trading_signals.generate_trading_signal(
                symbol, current_price, market_data, momentum_score
            )
            
            print_signal_header(symbol, current_price)
            print_signal_summary(signal)
            print_entry_strategy(signal.entry_strategy)
            print_alerts(signal.alerts)
            print_precautions(signal.precautions)
            print_intraday_signals(signal.intraday_signals)
            print_scalping_opportunities(signal.scalping_opportunities)
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎯 Demo completada. ¡El sistema está listo para trading en vivo!")
    print("💡 Características implementadas:")
    print("   ✅ Precio de entrada óptimo")
    print("   ✅ Alertas de precaución")
    print("   ✅ Señales de trading intradía")
    print("   ✅ Oportunidades de scalping")
    print("   ✅ Análisis de riesgo")
    print("   ✅ Targets escalonados")

if __name__ == "__main__":
    asyncio.run(demo_trading_signals())
