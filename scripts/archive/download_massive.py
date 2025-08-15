#!/usr/bin/env python3
"""
🚀 DESCARGADOR MASIVO DE DATOS
- 24 criptomonedas (doble de las originales)
- 5 timeframes incluyendo 5m
- 3000 velas por timeframe
- Descarga optimizada con reintentos
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import time
import json

def download_massive_crypto_data():
    """Descarga masiva de datos históricos optimizada"""
    print("🚀 DESCARGADOR MASIVO DE DATOS CRYPTO")
    print("=" * 55)
    
    # Configurar exchange
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
            'rateLimit': 100,  # Más conservador
        })
        print("✅ Binance conectado exitosamente")
    except Exception as e:
        print(f"❌ Error conectando a Binance: {e}")
        return False
    
    # 🚀 CONFIGURACIÓN EXPANDIDA
    symbols = [
        # Top cryptocurrencies originales
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT',
        'MATICUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT',
        
        # 12 cryptocurrencies adicionales
        'LTCUSDT', 'BCHUSDT', 'FILUSDT', 'TRXUSDT', 'ALGOUSDT', 'VETUSDT',
        'XLMUSDT', 'ICPUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT'
    ]
    
    # 5 timeframes incluyendo 5m
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    # Target: 3000 velas por timeframe
    target_candles = 3000
    
    print(f"💰 Símbolos: {len(symbols)} criptomonedas")
    print(f"📊 Timeframes: {timeframes}")
    print(f"🎯 Target: {target_candles:,} velas por timeframe")
    print(f"📈 Total estimado: {len(symbols) * len(timeframes) * target_candles:,} velas")
    print()
    
    successful_downloads = 0
    total_candles = 0
    
    # Crear directorios
    for tf in timeframes:
        Path(f"data/raw/{tf}").mkdir(parents=True, exist_ok=True)
    
    def download_with_retry(symbol, tf, max_retries=3):
        """Descarga con reintentos"""
        for attempt in range(max_retries):
            try:
                # Calcular fechas según timeframe para obtener suficientes datos
                end_date = datetime.now()
                
                if tf == '5m':
                    days_back = int(target_candles * 5 / (24 * 60)) + 10  # ~10 días
                elif tf == '15m':
                    days_back = int(target_candles * 15 / (24 * 60)) + 15  # ~31 días
                elif tf == '1h':
                    days_back = int(target_candles / 24) + 30  # ~155 días
                elif tf == '4h':
                    days_back = int(target_candles * 4 / 24) + 60  # ~560 días
                else:  # 1d
                    days_back = target_candles + 100  # ~3100 días
                
                start_date = end_date - timedelta(days=days_back)
                
                # Descargar en lotes
                all_data = []
                current_date = start_date
                
                while current_date < end_date and len(all_data) < target_candles:
                    since = exchange.parse8601(current_date.isoformat())
                    
                    try:
                        ohlcv = exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=tf,
                            since=since,
                            limit=1000  # Max por request
                        )
                        
                        if ohlcv:
                            all_data.extend(ohlcv)
                            # Actualizar fecha para siguiente lote
                            if len(ohlcv) > 0:
                                last_timestamp = ohlcv[-1][0]
                                current_date = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=1)
                            else:
                                break
                        else:
                            break
                            
                        time.sleep(0.2)  # Rate limiting
                        
                    except Exception as e:
                        if "429" in str(e):  # Rate limit
                            print(f"⏳ Rate limit, esperando...")
                            time.sleep(5)
                        else:
                            raise e
                
                # Procesar datos
                if all_data and len(all_data) > 0:
                    # Remover duplicados y ordenar
                    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    
                    # Tomar las últimas N velas
                    if len(df) > target_candles:
                        df = df.tail(target_candles)
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Guardar
                    filename = f"data/raw/{tf}/{symbol}_{tf}.csv"
                    df.to_csv(filename)
                    
                    return len(df)
                else:
                    return 0
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Intento {attempt + 1} falló, reintentando...")
                    time.sleep(2)
                else:
                    raise e
        
        return 0
    
    try:
        for tf in timeframes:
            print(f"🕐 TIMEFRAME: {tf}")
            print("-" * 35)
            
            tf_candles = 0
            
            for i, symbol in enumerate(symbols, 1):
                try:
                    print(f"  📈 [{i:2d}/{len(symbols)}] {symbol} {tf}... ", end="", flush=True)
                    
                    candles = download_with_retry(symbol, tf)
                    
                    if candles > 0:
                        print(f"✅ {candles:,} velas")
                        successful_downloads += 1
                        total_candles += candles
                        tf_candles += candles
                    else:
                        print("❌ Sin datos")
                    
                    # Rate limiting entre símbolos
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)[:50]}...")
            
            print(f"   📊 Subtotal {tf}: {tf_candles:,} velas")
            print()
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por usuario")
    
    # Resumen detallado
    print("=" * 55)
    print("📋 RESUMEN DESCARGA MASIVA:")
    print(f"✅ Descargas exitosas: {successful_downloads}")
    print(f"📈 Total velas descargadas: {total_candles:,}")
    print(f"💰 Símbolos procesados: {len(symbols)}")
    print(f"📊 Timeframes: {len(timeframes)}")
    
    # Estadísticas por timeframe
    print(f"\n📊 ESTADÍSTICAS POR TIMEFRAME:")
    for tf in timeframes:
        tf_dir = Path(f"data/raw/{tf}")
        if tf_dir.exists():
            files = list(tf_dir.glob("*.csv"))
            tf_total = 0
            for file in files:
                try:
                    df = pd.read_csv(file)
                    tf_total += len(df)
                except:
                    pass
            print(f"   {tf}: {len(files)} archivos, {tf_total:,} velas")
    
    if successful_downloads > 0:
        print(f"\n🎉 ¡DESCARGA MASIVA COMPLETADA!")
        print(f"📊 Dataset expandido significativamente")
        print(f"⚡ Listo para entrenamiento masivo")
        print(f"🚀 Ejecuta: python scripts/train_crypto_optimized.py")
        return True
    else:
        print(f"\n❌ No se pudieron descargar datos")
        return False

def validate_downloads():
    """Valida la calidad de las descargas"""
    print(f"\n🔍 VALIDANDO CALIDAD DE DATOS...")
    
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    validation_results = {}
    
    for tf in timeframes:
        tf_dir = Path(f"data/raw/{tf}")
        if not tf_dir.exists():
            continue
        
        files = list(tf_dir.glob("*.csv"))
        valid_files = 0
        total_candles = 0
        
        for file in files:
            try:
                df = pd.read_csv(file)
                if len(df) > 100:  # Mínimo 100 velas para ser válido
                    valid_files += 1
                    total_candles += len(df)
            except:
                pass
        
        validation_results[tf] = {
            'files': len(files),
            'valid_files': valid_files,
            'total_candles': total_candles
        }
        
        print(f"   {tf}: {valid_files}/{len(files)} archivos válidos, {total_candles:,} velas")
    
    return validation_results

if __name__ == "__main__":
    print("🚀 INICIANDO DESCARGA MASIVA...")
    success = download_massive_crypto_data()
    
    if success:
        print("\n" + "="*55)
        validate_downloads()
        print("\n✅ PROCESO COMPLETADO - DATASET EXPANDIDO")
    else:
        print("\n❌ DESCARGA FALLÓ")
        exit(1)
