#!/usr/bin/env python3
"""
🚀 DESCARGADOR CORREGIDO Y OPTIMIZADO
- Manejo robusto de errores y timeouts
- Recuperación automática desde donde se quedó
- Rate limiting inteligente
- Validación continua de datos
"""

import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import time
import json
import signal
import sys
from typing import Optional, Dict, List

class CorrectedDownloader:
    def __init__(self):
        self.exchange = None
        self.progress_file = "data/download_progress.json"
        self.interrupted = False
        
        # 24 criptomonedas como solicitaste (doble de las originales)
        self.symbols = [
            # 12 originales
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT',
            'MATICUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'ATOMUSDT',
            
            # 12 adicionales  
            'LTCUSDT', 'BCHUSDT', 'FILUSDT', 'TRXUSDT', 'ALGOUSDT', 'VETUSDT',
            'XLMUSDT', 'ICPUSDT', 'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'AXSUSDT'
        ]
        
        # 5 timeframes incluyendo 5m
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        
        # Target: 3000 velas por archivo
        self.target_candles = 3000
        
        # Configurar signal handler para interrupciones
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print(f"🎯 Configuración:")
        print(f"   📊 {len(self.symbols)} criptomonedas")
        print(f"   ⏰ {len(self.timeframes)} timeframes: {self.timeframes}")
        print(f"   📈 {self.target_candles:,} velas objetivo por archivo")
        print(f"   🎯 Total estimado: {len(self.symbols) * len(self.timeframes) * self.target_candles:,} velas")
    
    def signal_handler(self, sig, frame):
        """Maneja interrupciones del usuario"""
        print(f"\n⚠️ Interrupción detectada, guardando progreso...")
        self.interrupted = True
    
    def setup_exchange(self) -> bool:
        """Configura exchange con parámetros ultra-robustos"""
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 120000,  # 2 minutos timeout
                'rateLimit': 300,   # Muy conservador (300ms entre requests)
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                }
            })
            
            # Test de conectividad
            self.exchange.load_markets()
            print("✅ Exchange configurado y probado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error configurando exchange: {e}")
            return False
    
    def load_progress(self) -> Dict:
        """Carga progreso anterior si existe"""
        try:
            if Path(self.progress_file).exists():
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                print(f"📂 Progreso anterior encontrado")
                return progress
        except Exception as e:
            print(f"⚠️ Error cargando progreso: {e}")
        
        # Progreso inicial
        return {
            'completed_files': set(),
            'current_timeframe': None,
            'current_symbol_index': 0,
            'total_downloaded': 0,
            'failed_attempts': {}
        }
    
    def save_progress(self, progress: Dict):
        """Guarda progreso actual"""
        try:
            Path("data").mkdir(exist_ok=True)
            # Convertir set a list para JSON
            progress_copy = progress.copy()
            progress_copy['completed_files'] = list(progress['completed_files'])
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_copy, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error guardando progreso: {e}")
    
    def calculate_date_range(self, timeframe: str) -> tuple:
        """Calcula rango de fechas para obtener suficientes velas"""
        end_date = datetime.now()
        
        # Calcular días necesarios para obtener target_candles
        timeframe_minutes = {
            '5m': 5,
            '15m': 15, 
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        minutes_per_candle = timeframe_minutes.get(timeframe, 60)
        total_minutes_needed = self.target_candles * minutes_per_candle
        days_needed = int(total_minutes_needed / (24 * 60)) + 30  # +30 días de buffer
        
        start_date = end_date - timedelta(days=days_needed)
        
        return start_date, end_date
    
    def download_symbol_robust(self, symbol: str, timeframe: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Descarga datos de un símbolo con manejo robusto de errores"""
        
        if self.exchange is None:
            print("❌ Exchange no configurado")
            return None
        
        for attempt in range(max_retries):
            if self.interrupted:
                return None
            
            try:
                start_date, end_date = self.calculate_date_range(timeframe)
                
                print(f"      Intento {attempt + 1}/{max_retries} ", end="", flush=True)
                
                # Descargar datos en chunks pequeños para evitar timeouts
                all_ohlcv = []
                current_start = start_date
                chunk_size = 500  # Chunks más pequeños para estabilidad
                
                while current_start < end_date and len(all_ohlcv) < self.target_candles:
                    if self.interrupted:
                        break
                    
                    try:
                        since = self.exchange.parse8601(current_start.isoformat())
                        
                        # Request con timeout explícito
                        ohlcv = self.exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=since,
                            limit=chunk_size
                        )
                        
                        if not ohlcv:
                            print("(sin datos) ", end="", flush=True)
                            break
                        
                        all_ohlcv.extend(ohlcv)
                        
                        # Actualizar fecha para siguiente chunk
                        if len(ohlcv) > 0:
                            last_timestamp = ohlcv[-1][0]
                            current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=1)
                            print(".", end="", flush=True)
                        else:
                            break
                        
                        # Rate limiting progresivo
                        time.sleep(0.6)  # 600ms entre requests
                        
                    except ccxt.RateLimitExceeded as e:
                        print(f"(límite) ", end="", flush=True)
                        time.sleep(15)  # Esperar más en rate limits
                        continue
                    except ccxt.NetworkError as e:
                        print(f"(red) ", end="", flush=True)
                        time.sleep(5)  # Esperar en errores de red
                        continue
                    except Exception as e:
                        print(f"(error: {str(e)[:20]}) ", end="", flush=True)
                        break
                
                # Procesar datos si se obtuvieron
                if all_ohlcv and len(all_ohlcv) > 0:
                    # Crear DataFrame
                    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Limpiar duplicados
                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    
                    # Convertir timestamps
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Tomar últimas N velas
                    if len(df) > self.target_candles:
                        df = df.tail(self.target_candles)
                    
                    # Validar datos básicos
                    if len(df) < 100:
                        print(f"(pocos datos: {len(df)}) ", end="", flush=True)
                        continue
                    
                    print(f"✅ {len(df):,} velas")
                    return df
                else:
                    print("❌ Sin datos válidos")
                    continue
                    
            except ccxt.BaseError as e:
                error_str = str(e)[:30]
                print(f"❌ CCXT Error: {error_str}")
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"      Esperando {wait_time}s...")
                    time.sleep(wait_time)
                
            except Exception as e:
                error_str = str(e)[:30]
                print(f"❌ Error: {error_str}")
                
                if attempt < max_retries - 1:
                    time.sleep(3)
        
        print(f"❌ Falló después de {max_retries} intentos")
        return None
    
    def save_dataframe(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """Guarda DataFrame con validación"""
        try:
            if df is None or df.empty:
                return False
            
            # Crear directorio
            data_dir = Path(f"data/raw/{timeframe}")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar archivo
            filename = f"{symbol}_{timeframe}.csv"
            filepath = data_dir / filename
            df.to_csv(filepath)
            
            # Verificar que se guardó correctamente
            if filepath.exists() and filepath.stat().st_size > 1024:  # Al menos 1KB
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Error guardando {symbol}_{timeframe}: {e}")
            return False
    
    def download_all_corrected(self) -> bool:
        """Descarga principal corregida"""
        print("🚀 DESCARGADOR CORREGIDO Y OPTIMIZADO")
        print("=" * 55)
        
        if not self.setup_exchange():
            return False
        
        # Cargar progreso
        progress = self.load_progress()
        if 'completed_files' not in progress:
            progress['completed_files'] = set()
        elif isinstance(progress['completed_files'], list):
            progress['completed_files'] = set(progress['completed_files'])
        
        total_files = len(self.symbols) * len(self.timeframes)
        completed_files = len(progress.get('completed_files', set()))
        
        print(f"📊 Progreso: {completed_files}/{total_files} archivos completados")
        print()
        
        try:
            for tf_idx, timeframe in enumerate(self.timeframes):
                if self.interrupted:
                    break
                
                print(f"🕐 TIMEFRAME: {timeframe.upper()}")
                print("-" * 40)
                
                tf_success = 0
                tf_failed = 0
                
                for sym_idx, symbol in enumerate(self.symbols):
                    if self.interrupted:
                        break
                    
                    file_key = f"{symbol}_{timeframe}"
                    
                    # Saltar si ya está completado
                    if file_key in progress.get('completed_files', set()):
                        print(f"  📈 [{sym_idx+1:2d}/{len(self.symbols)}] {symbol} {timeframe}... ✅ Ya completado")
                        tf_success += 1
                        continue
                    
                    print(f"  📈 [{sym_idx+1:2d}/{len(self.symbols)}] {symbol} {timeframe}... ", end="", flush=True)
                    
                    # Verificar si ya existe archivo válido
                    existing_file = Path(f"data/raw/{timeframe}/{file_key}.csv")
                    if existing_file.exists():
                        try:
                            test_df = pd.read_csv(existing_file)
                            if len(test_df) >= 100:
                                print(f"✅ Archivo existente válido ({len(test_df)} velas)")
                                progress['completed_files'].add(file_key)
                                tf_success += 1
                                continue
                        except:
                            pass
                    
                    # Descargar datos
                    df = self.download_symbol_robust(symbol, timeframe)
                    
                    if df is not None and self.save_dataframe(df, symbol, timeframe):
                        progress['completed_files'].add(file_key)
                        progress['total_downloaded'] += len(df)
                        tf_success += 1
                    else:
                        tf_failed += 1
                        # Registrar fallos
                        if 'failed_attempts' not in progress:
                            progress['failed_attempts'] = {}
                        progress['failed_attempts'][file_key] = datetime.now().isoformat()
                    
                    # Guardar progreso cada símbolo
                    self.save_progress(progress)
                    
                    # Pausa entre símbolos para rate limiting
                    if not self.interrupted:
                        time.sleep(1.5)
                
                print(f"\n📊 Resumen {timeframe}: {tf_success} éxitos, {tf_failed} fallos")
                
                if self.interrupted:
                    break
                
                # Pausa entre timeframes
                if tf_idx < len(self.timeframes) - 1:
                    print("⏳ Pausa entre timeframes (10s)...")
                    time.sleep(10)
        
        except KeyboardInterrupt:
            self.interrupted = True
        except Exception as e:
            print(f"\n❌ Error crítico: {e}")
            self.interrupted = True
        
        # Resumen final
        print("\n" + "=" * 55)
        if self.interrupted:
            print("⚠️ DESCARGA INTERRUMPIDA")
            print("📂 Progreso guardado - ejecuta el script nuevamente para continuar")
        else:
            print("✅ DESCARGA COMPLETADA")
            # Limpiar progreso si está completo
            try:
                Path(self.progress_file).unlink()
            except:
                pass
        
        self.validate_final_data()
        
        return not self.interrupted
    
    def validate_final_data(self):
        """Validación final de todos los datos"""
        print(f"\n🔍 VALIDACIÓN FINAL...")
        
        total_files = 0
        total_candles = 0
        
        for timeframe in self.timeframes:
            tf_dir = Path(f"data/raw/{timeframe}")
            if not tf_dir.exists():
                print(f"   {timeframe}: ❌ Directorio no existe")
                continue
            
            files = list(tf_dir.glob("*.csv"))
            valid_files = 0
            tf_candles = 0
            
            for file in files:
                try:
                    df = pd.read_csv(file)
                    if len(df) >= 100:
                        valid_files += 1
                        tf_candles += len(df)
                except:
                    pass
            
            total_files += valid_files
            total_candles += tf_candles
            
            expected_files = len(self.symbols)
            completion_rate = (valid_files / expected_files) * 100 if expected_files > 0 else 0
            
            print(f"   {timeframe}: {valid_files}/{expected_files} archivos ({completion_rate:.1f}%) - {tf_candles:,} velas")
        
        print(f"\n📊 TOTAL: {total_files} archivos válidos, {total_candles:,} velas")
        
        if total_files > 50:  # Al menos 50 archivos para considerar éxito
            print(f"🎉 Dataset robusto creado - listo para entrenamiento!")
        else:
            print(f"⚠️ Dataset limitado - considera reintentar descarga")

def main():
    """Función principal"""
    downloader = CorrectedDownloader()
    success = downloader.download_all_corrected()
    
    if success:
        print(f"\n🎉 ¡DESCARGA EXITOSA!")
        print(f"🚀 Dataset masivo listo para entrenamiento")
        print(f"⚡ Ejecuta: python scripts/train_crypto_optimized.py")
    else:
        print(f"\n⚠️ Descarga interrumpida")
        print(f"💡 Ejecuta nuevamente para continuar desde donde se quedó")

if __name__ == "__main__":
    main()
