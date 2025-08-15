#!/usr/bin/env python3
"""
Descargador completo de datos históricos para trading intraday
Múltiples timeframes, criptomonedas y 2 años de historia
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import ccxt
from dotenv import load_dotenv
import time
import json

# Cargar variables de entorno
load_dotenv(Path(__file__).parent.parent / "config" / "secrets.env")


class AdvancedDataDownloader:
    def __init__(self):
        self.exchange = None
        self.setup_exchange()
        
        # Configuración de descarga
        self.timeframes = {
            '15m': '15 minutos - Señales rápidas intraday',
            '1h': '1 hora - Tendencias corto plazo', 
            '4h': '4 horas - Contexto mediano plazo',
            '1d': '1 día - Tendencia general'
        }
        
        # Criptomonedas por categoría
        self.crypto_pairs = {
            'majors': [
                'BTCUSDT',   # Bitcoin - El rey
                'ETHUSDT',   # Ethereum - DeFi leader
                'BNBUSDT',   # Binance Coin - Exchange token
                'ADAUSDT',   # Cardano - Smart contracts
                'SOLUSDT',   # Solana - High performance
                'XRPUSDT',   # Ripple - Payments
            ],
            'altcoins': [
                'MATICUSDT', # Polygon - Layer 2
                'DOTUSDT',   # Polkadot - Interoperability  
                'AVAXUSDT',  # Avalanche - Fast blockchain
                'LINKUSDT',  # Chainlink - Oracles
                'UNIUSDT',   # Uniswap - DEX
                'ATOMUSDT',  # Cosmos - Internet of blockchains
            ]
        }
        
        # Período de descarga (2 años)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=730)  # 2 años
        
    def setup_exchange(self):
        """Configura el exchange para descarga"""
        try:
            self.exchange = ccxt.binance({
                'sandbox': False,
                'enableRateLimit': True,
                'timeout': 30000,
                'rateLimit': 1200,  # Más conservador para evitar límites
                'options': {
                    'defaultType': 'spot',  # Asegurar que usamos spot trading
                }
            })
            print("✅ Exchange configurado correctamente")
        except Exception as e:
            print(f"❌ Error configurando exchange: {e}")
            raise Exception(f"No se pudo inicializar el exchange: {e}")
    def download_symbol_timeframe(self, symbol, timeframe, max_retries=3):
        """Descarga datos para un símbolo y timeframe específico"""
        if self.exchange is None:
            print(f"    ❌ Exchange no inicializado para {symbol} {timeframe}")
            return None
            
        for attempt in range(max_retries):
            try:
                print(f"  📈 Descargando {symbol} {timeframe}... (intento {attempt + 1})")
                                
                # Calcular límite por timeframe
                timeframe_limits = {
                    '15m': 1000,  # ~10 días por llamada
                    '1h': 1000,   # ~41 días por llamada  
                    '4h': 1000,   # ~166 días por llamada
                    '1d': 1000    # ~2.7 años por llamada
                }
                
                limit = timeframe_limits.get(timeframe, 1000)
                
                # Para timeframes pequeños, necesitamos múltiples llamadas
                all_data = []
                current_start = self.start_date
                
                while current_start < self.end_date:
                    since = self.exchange.parse8601(current_start.isoformat())
                    
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=since,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_data.extend(ohlcv)
                    
                    # Calcular siguiente fecha de inicio
                    last_timestamp = ohlcv[-1][0]
                    current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(hours=1)
                    
                    # Si descargamos menos del límite, hemos llegado al final
                    if len(ohlcv) < limit:
                        break
                        
                    # Rate limiting
                    time.sleep(1.5)
                
                if not all_data:
                    print(f"    ⚠️  No hay datos para {symbol} {timeframe}")
                    return None
                
                # Convertir a DataFrame
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Eliminar duplicados y ordenar
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                
                # Filtrar por fechas exactas
                df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
                
                print(f"    ✅ {len(df):,} velas descargadas")
                return df
                
            except Exception as e:
                print(f"    ❌ Error (intento {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"    🔄 Reintentando en 3 segundos...")
                    time.sleep(3)
                else:
                    print(f"    💀 Falló después de {max_retries} intentos")
                    return None
        
        return None
    
    def save_data(self, df, symbol, timeframe):
        """Guarda los datos en archivo CSV"""
        if df is None or df.empty:
            return False
            
        # Crear directorio por timeframe
        data_dir = Path("data/raw") / timeframe
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo
        filename = f"{symbol}_{timeframe}.csv"
        filepath = data_dir / filename
        df.to_csv(filepath)
        
        # Calcular tamaño del archivo
        file_size = filepath.stat().st_size / (1024 * 1024)  # MB
        
        print(f"    💾 Guardado: {filename} ({file_size:.1f} MB)")
        return True
    
    def download_all_data(self):
        """Descarga todos los datos históricos"""
        print("🚀 DESCARGADOR AVANZADO DE DATOS HISTÓRICOS")
        print("=" * 50)
        print(f"📅 Período: {self.start_date.strftime('%Y-%m-%d')} a {self.end_date.strftime('%Y-%m-%d')}")
        print(f"📊 Timeframes: {list(self.timeframes.keys())}")
        print(f"💰 Símbolos: {len(self.crypto_pairs['majors']) + len(self.crypto_pairs['altcoins'])} criptomonedas")
        print()
        
        # Estadísticas de descarga
        stats = {
            'total_files': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_candles': 0,
            'symbols_by_timeframe': {},
            'download_time': datetime.now()
        }
        
        start_time = time.time()
        
        try:
            # Descargar por timeframe
            for timeframe, description in self.timeframes.items():
                print(f"\n🕐 TIMEFRAME: {timeframe.upper()} ({description})")
                print("-" * 40)
                
                timeframe_stats = {'successful': 0, 'failed': 0, 'total_candles': 0}
                
                # Descargar majors
                print("💎 MAJOR CRYPTOCURRENCIES:")
                for symbol in self.crypto_pairs['majors']:
                    df = self.download_symbol_timeframe(symbol, timeframe)
                    if self.save_data(df, symbol, timeframe):
                        stats['successful_downloads'] += 1
                        timeframe_stats['successful'] += 1
                        timeframe_stats['total_candles'] += len(df) if df is not None else 0
                    else:
                        stats['failed_downloads'] += 1
                        timeframe_stats['failed'] += 1
                    
                    stats['total_files'] += 1
                    time.sleep(1)  # Rate limiting
                
                # Descargar altcoins  
                print("\n🔸 ALTCOINS:")
                for symbol in self.crypto_pairs['altcoins']:
                    df = self.download_symbol_timeframe(symbol, timeframe)
                    if self.save_data(df, symbol, timeframe):
                        stats['successful_downloads'] += 1
                        timeframe_stats['successful'] += 1
                        timeframe_stats['total_candles'] += len(df) if df is not None else 0
                    else:
                        stats['failed_downloads'] += 1
                        timeframe_stats['failed'] += 1
                    
                    stats['total_files'] += 1
                    time.sleep(1)  # Rate limiting
                
                stats['symbols_by_timeframe'][timeframe] = timeframe_stats
                stats['total_candles'] += timeframe_stats['total_candles']
                
                print(f"\n📊 Resumen {timeframe}: {timeframe_stats['successful']} éxitos, {timeframe_stats['failed']} fallos")
                print(f"📈 Total velas: {timeframe_stats['total_candles']:,}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Descarga interrumpida por el usuario")
        except Exception as e:
            print(f"\n❌ Error durante la descarga: {e}")
        
        # Resumen final
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "=" * 50)
        print("📋 RESUMEN FINAL DE DESCARGA")
        print("=" * 50)
        print(f"⏱️  Tiempo total: {duration/60:.1f} minutos")
        print(f"✅ Descargas exitosas: {stats['successful_downloads']}")
        print(f"❌ Descargas fallidas: {stats['failed_downloads']}")
        print(f"📊 Total archivos: {stats['total_files']}")
        print(f"📈 Total velas descargadas: {stats['total_candles']:,}")
        
        # Detalles por timeframe
        print(f"\n📊 DETALLES POR TIMEFRAME:")
        for tf, tf_stats in stats['symbols_by_timeframe'].items():
            success_rate = (tf_stats['successful'] / (tf_stats['successful'] + tf_stats['failed']) * 100) if (tf_stats['successful'] + tf_stats['failed']) > 0 else 0
            print(f"  {tf}: {tf_stats['successful']}/{tf_stats['successful'] + tf_stats['failed']} ({success_rate:.1f}%) - {tf_stats['total_candles']:,} velas")
        
        # Guardar estadísticas
        stats_file = Path("data/raw") / "download_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\n💾 Estadísticas guardadas en: {stats_file}")
        
        # Recomendaciones
        print(f"\n🎯 PRÓXIMOS PASOS:")
        if stats['successful_downloads'] > 0:
            print("1. ✅ Ejecutar entrenamiento con más datos:")
            print("   python scripts/train_models_advanced.py")
            print("2. 📊 Verificar calidad de datos:")
            print("   python scripts/analyze_data_quality.py")
            print("3. 🚀 Probar diferentes timeframes para optimizar")
        else:
            print("1. ❌ Revisar configuración de red")
            print("2. 🔄 Reintentar descarga")
            print("3. 📞 Verificar límites de API de Binance")
        
        return stats


def main():
    """Función principal"""
    print("🎯 DESCARGA MASIVA DE DATOS PARA TRADING INTRADAY")
    print("Preparando datos para modelos de IA avanzados...")
    print()
    
    downloader = AdvancedDataDownloader()
    stats = downloader.download_all_data()
    
    return stats['successful_downloads'] > 0


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Descarga no exitosa")
        exit(1)
    else:
        print("\n🎉 ¡Descarga completada! Listo para entrenar modelos avanzados.")
