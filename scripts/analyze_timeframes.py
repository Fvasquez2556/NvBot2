#!/usr/bin/env python3
"""
Script para analizar timeframes múltiples
"""

import asyncio
import argparse
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.strategies.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager

logger = get_logger(__name__)

async def analyze_symbol_timeframes(symbol: str, config: dict):
    """
    Analiza un símbolo específico en múltiples timeframes
    """
    analyzer = MultiTimeframeAnalyzer(config)
    
    print(f"\n🔍 Analyzing {symbol} across multiple timeframes...")
    print("=" * 60)
    
    try:
        # Simular precio actual (en implementación real vendría del exchange)
        current_price = 100.0
        
        # Análisis de confluencia
        result = await analyzer.analyze_confluence(symbol, current_price)
        
        print(f"📊 Confluence Analysis Results:")
        print(f"   Unified Score: {result.unified_score:.3f}")
        print(f"   Confidence: {result.confidence_level:.3f}")
        print(f"   Signal Strength: {result.signal_strength}")
        print(f"   Dominant Timeframe: {result.dominant_timeframe}")
        print(f"   Risk Level: {result.risk_level}")
        print(f"   Entry Recommendation: {'✅ YES' if result.entry_recommendation else '❌ NO'}")
        
        if result.entry_recommendation:
            print(f"\n💡 Trade Recommendations:")
            print(f"   Entry Price: ${current_price:.4f}")
            print(f"   Stop Loss: ${result.stop_loss_level:.4f}")
            print(f"   Take Profit Levels:")
            for i, tp in enumerate(result.take_profit_levels, 1):
                print(f"     TP{i}: ${tp:.4f}")
            print(f"   Max Position Size: {result.max_position_size:.1%}")
        
        # Status de salud de timeframes
        print(f"\n🏥 Timeframe Health Status:")
        health_status = await analyzer.get_timeframe_health_status()
        for tf, status in health_status.items():
            status_emoji = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'degraded' else "❌"
            print(f"   {tf}: {status_emoji} {status['status']} (weight: {status['weight']:.1%})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description='Multi-timeframe analysis tool')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to analyze')
    parser.add_argument('--timeframes', nargs='+', default=['3m', '5m', '15m'], 
                       help='Timeframes to analyze')
    parser.add_argument('--config', help='Config file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    if args.config:
        config = config_manager.load_config(args.config)
    else:
        config = config_manager.get_strategy_config('momentum_predictor')
    
    # Override timeframes if specified
    if args.timeframes:
        config['multi_timeframe'] = config.get('multi_timeframe', {})
        config['multi_timeframe']['timeframes'] = args.timeframes
    
    print(f"🚀 Multi-Timeframe Analyzer")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run analysis
    result = await analyze_symbol_timeframes(args.symbol, config.get('multi_timeframe', {}))
    
    if result:
        print(f"\n✅ Analysis completed successfully!")
    else:
        print(f"\n❌ Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
