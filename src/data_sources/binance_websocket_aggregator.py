"""
Advanced Data Aggregator con WebSockets y procesamiento paralelo
Analiza TODOS los pares USDT de Binance de forma eficiente
"""

import asyncio
import json
import aiohttp
import websockets
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import concurrent.futures
from threading import Lock
import time
import logging

# Imports locales con fallback
try:
    from src.utils.logger import get_logger
    from src.utils.config_manager import ConfigManager
except ImportError:
    try:
        from utils.logger import get_logger
        from utils.config_manager import ConfigManager
    except ImportError:
        # Fallback básico para testing
        def get_logger(name):
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            return logger
        
        class ConfigManager:
            def __init__(self):
                pass
            def get(self, key, default=None):
                return default

logger = get_logger(__name__)

@dataclass
class TickerData:
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime

@dataclass
class KlineData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str

@dataclass
class MarketSnapshot:
    """Snapshot completo del mercado en un momento dado"""
    timestamp: datetime
    tickers: Dict[str, TickerData]
    top_gainers: List[str]
    top_losers: List[str]
    high_volume: List[str]
    total_pairs: int

class BinanceWebSocketAggregator:
    """
    Agregador de datos avanzado que usa WebSockets de Binance
    para obtener datos en tiempo real de TODOS los pares USDT
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443/ws/"
        
        # Almacenamiento de datos en memoria
        self.tickers: Dict[str, TickerData] = {}
        self.klines: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.usdt_pairs: Set[str] = set()
        
        # WebSocket connections
        self.ws_connections = {}
        self.is_running = False
        
        # Threading y locks para acceso concurrente
        self.data_lock = Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
        # Configuración
        self.min_volume_24h = config.get('data_aggregator.min_volume_24h', 1000000)  # $1M
        self.max_pairs = config.get('data_aggregator.max_pairs', 200)  # Máximo pares a analizar
        self.update_interval = config.get('data_aggregator.update_interval', 1)  # segundos
        
        # Callbacks para eventos
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Estadísticas
        self.stats = {
            'messages_received': 0,
            'last_update': None,
            'active_pairs': 0,
            'websockets_connected': 0
        }
    
    async def initialize(self):
        """Inicializa el agregador obteniendo lista de pares USDT"""
        try:
            logger.info("🚀 Inicializando Binance WebSocket Aggregator...")
            
            # Obtener todos los pares USDT activos
            await self._fetch_usdt_pairs()
            
            # Filtrar por volumen y limitar cantidad
            await self._filter_pairs_by_volume()
            
            logger.info(f"📊 {len(self.usdt_pairs)} pares USDT seleccionados para análisis")
            
            # Iniciar conexiones WebSocket
            await self._start_websocket_streams()
            
            # Obtener datos históricos iniciales
            await self._fetch_initial_data()
            
            logger.info("✅ Binance WebSocket Aggregator inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando aggregator: {e}")
            raise
    
    async def _fetch_usdt_pairs(self):
        """Obtiene todos los pares USDT activos de Binance"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v3/exchangeInfo") as response:
                    data = await response.json()
                    
                    for symbol_info in data['symbols']:
                        if (symbol_info['status'] == 'TRADING' and 
                            symbol_info['quoteAsset'] == 'USDT' and
                            symbol_info['symbol'].endswith('USDT')):
                            self.usdt_pairs.add(symbol_info['symbol'])
                    
                    logger.info(f"🔍 Encontrados {len(self.usdt_pairs)} pares USDT activos")
                    
        except Exception as e:
            logger.error(f"Error obteniendo pares USDT: {e}")
            raise
    
    async def _filter_pairs_by_volume(self):
        """Filtra pares por volumen 24h y otros criterios"""
        try:
            # Obtener estadísticas 24h de todos los pares
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/v3/ticker/24hr") as response:
                    tickers_24h = await response.json()
            
            # Filtrar y ordenar por volumen
            volume_filtered = []
            for ticker in tickers_24h:
                if (ticker['symbol'] in self.usdt_pairs and 
                    float(ticker['quoteVolume']) >= self.min_volume_24h):
                    
                    volume_filtered.append({
                        'symbol': ticker['symbol'],
                        'volume': float(ticker['quoteVolume']),
                        'change': abs(float(ticker['priceChangePercent']))
                    })
            
            # Ordenar por volumen descendente y tomar los top
            volume_filtered.sort(key=lambda x: x['volume'], reverse=True)
            selected_pairs = [pair['symbol'] for pair in volume_filtered[:self.max_pairs]]
            
            self.usdt_pairs = set(selected_pairs)
            
            logger.info(f"📈 Seleccionados top {len(self.usdt_pairs)} pares por volumen (min: ${self.min_volume_24h:,.0f})")
            
        except Exception as e:
            logger.error(f"Error filtrando pares por volumen: {e}")
            # Si falla, usar todos los pares encontrados
            pass
    
    async def _start_websocket_streams(self):
        """Inicia streams WebSocket para todos los pares seleccionados"""
        try:
            self.is_running = True
            
            # Crear chunks de pares para múltiples conexiones WebSocket
            pairs_list = list(self.usdt_pairs)
            chunk_size = 50  # Binance permite hasta 200 streams por conexión
            pair_chunks = [pairs_list[i:i + chunk_size] for i in range(0, len(pairs_list), chunk_size)]
            
            # Iniciar una conexión WebSocket por chunk
            tasks = []
            for i, chunk in enumerate(pair_chunks):
                task = asyncio.create_task(self._websocket_connection(chunk, i))
                tasks.append(task)
            
            logger.info(f"🌐 Iniciando {len(tasks)} conexiones WebSocket para {len(self.usdt_pairs)} pares")
            
            # No esperamos a que terminen (son conexiones permanentes)
            # Solo guardamos las tareas para poder cancelarlas después
            self.ws_tasks = tasks
            
        except Exception as e:
            logger.error(f"Error iniciando WebSocket streams: {e}")
            raise
    
    async def _websocket_connection(self, pairs: List[str], connection_id: int):
        """Maneja una conexión WebSocket para un grupo de pares"""
        try:
            # Crear streams para ticker y klines
            streams = []
            
            # Ticker streams (precios en tiempo real)
            for pair in pairs:
                streams.append(f"{pair.lower()}@ticker")
                streams.append(f"{pair.lower()}@kline_1m")  # Klines de 1 minuto
            
            # URL del WebSocket con múltiples streams
            stream_names = "/".join(streams)
            ws_url = f"{self.ws_url}{stream_names}"
            
            logger.info(f"📡 Conectando WebSocket #{connection_id} con {len(pairs)} pares")
            
            reconnect_attempts = 0
            max_reconnect = 5
            
            while self.is_running and reconnect_attempts < max_reconnect:
                try:
                    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as websocket:
                        self.ws_connections[connection_id] = websocket
                        self.stats['websockets_connected'] += 1
                        
                        logger.info(f"✅ WebSocket #{connection_id} conectado")
                        reconnect_attempts = 0  # Reset counter on successful connection
                        
                        async for message in websocket:
                            if not self.is_running:
                                break
                                
                            try:
                                data = json.loads(message)
                                await self._process_websocket_message(data)
                                
                            except json.JSONDecodeError:
                                logger.warning(f"Mensaje WebSocket inválido: {message[:100]}")
                            except Exception as e:
                                logger.error(f"Error procesando mensaje WebSocket: {e}")
                                
                except websockets.exceptions.WebSocketException as e:
                    reconnect_attempts += 1
                    logger.warning(f"WebSocket #{connection_id} desconectado: {e}. Reintento {reconnect_attempts}/{max_reconnect}")
                    await asyncio.sleep(min(reconnect_attempts * 2, 30))  # Backoff exponencial
                    
                except Exception as e:
                    logger.error(f"Error en WebSocket #{connection_id}: {e}")
                    reconnect_attempts += 1
                    await asyncio.sleep(5)
            
            logger.warning(f"❌ WebSocket #{connection_id} cerrado después de {reconnect_attempts} intentos")
            
        except Exception as e:
            logger.error(f"Error crítico en WebSocket #{connection_id}: {e}")
    
    async def _process_websocket_message(self, data: dict):
        """Procesa mensajes de WebSocket en paralelo"""
        try:
            # Log para debug (solo en modo debug)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Mensaje recibido: {data}")
            
            # Usar executor para procesar en paralelo sin bloquear el event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._process_message_sync, data)
            
        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}")
    
    def _process_message_sync(self, data: dict):
        """Procesamiento sincrónico de mensajes (ejecutado en thread pool)"""
        try:
            with self.data_lock:
                self.stats['messages_received'] += 1
                self.stats['last_update'] = datetime.now()
                
                # Manejar formato con stream wrapper
                if 'stream' in data and 'data' in data:
                    stream = data['stream']
                    payload = data['data']
                    
                    if '@ticker' in stream:
                        self._process_ticker_data(payload)
                    elif '@kline' in stream:
                        self._process_kline_data(payload)
                        
                # Manejar formato directo (sin wrapper)
                elif 's' in data and 'c' in data:  # Formato ticker directo
                    self._process_ticker_data(data)
                elif 'k' in data:  # Formato kline directo
                    self._process_kline_data(data)
                else:
                    # Log para debug si el formato es inesperado
                    logger.debug(f"Formato de mensaje desconocido: {list(data.keys())}")
                    
        except Exception as e:
            logger.error(f"Error en procesamiento sincrónico: {e}")
    
    def _process_ticker_data(self, data: dict):
        """Procesa datos de ticker"""
        try:
            symbol = data['s']
            
            ticker = TickerData(
                symbol=symbol,
                price=float(data['c']),
                change_24h=float(data['P']),
                volume_24h=float(data['q']),
                high_24h=float(data['h']),
                low_24h=float(data['l']),
                timestamp=datetime.now()
            )
            
            self.tickers[symbol] = ticker
            
            # Trigger callbacks
            self._trigger_callbacks('ticker_update', symbol, ticker)
            
        except Exception as e:
            logger.error(f"Error procesando ticker: {e}")
    
    def _process_kline_data(self, data: dict):
        """Procesa datos de klines"""
        try:
            if not data['k']['x']:  # Solo procesar klines cerradas
                return
                
            kline_data = data['k']
            symbol = kline_data['s']
            
            kline = KlineData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(kline_data['t'] / 1000),
                open=float(kline_data['o']),
                high=float(kline_data['h']),
                low=float(kline_data['l']),
                close=float(kline_data['c']),
                volume=float(kline_data['v']),
                timeframe='1m'
            )
            
            self.klines[symbol].append(kline)
            
            # Trigger callbacks
            self._trigger_callbacks('kline_update', symbol, kline)
            
        except Exception as e:
            logger.error(f"Error procesando kline: {e}")
    
    def _trigger_callbacks(self, event: str, *args):
        """Dispara callbacks registrados"""
        try:
            callback_count = len(self.callbacks[event])
            if callback_count > 0:
                logger.debug(f"Disparando {callback_count} callbacks para evento: {event}")
                
            for callback in self.callbacks[event]:
                try:
                    callback(*args)
                except Exception as e:
                    logger.error(f"Error en callback {event}: {e}")
        except Exception as e:
            logger.error(f"Error disparando callbacks {event}: {e}")
    
    async def _fetch_initial_data(self):
        """Obtiene datos históricos iniciales para todos los pares"""
        try:
            logger.info("📋 Obteniendo datos históricos iniciales...")
            
            # Procesar en chunks para evitar sobrecarga
            pairs_list = list(self.usdt_pairs)
            chunk_size = 20
            
            tasks = []
            for i in range(0, len(pairs_list), chunk_size):
                chunk = pairs_list[i:i + chunk_size]
                task = asyncio.create_task(self._fetch_chunk_historical_data(chunk))
                tasks.append(task)
            
            # Ejecutar en paralelo
            await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info("✅ Datos históricos iniciales obtenidos")
            
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos: {e}")
    
    async def _fetch_chunk_historical_data(self, pairs: List[str]):
        """Obtiene datos históricos para un chunk de pares"""
        try:
            async with aiohttp.ClientSession() as session:
                tasks = []
                for symbol in pairs:
                    task = self._fetch_symbol_klines(session, symbol)
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Error obteniendo datos de chunk: {e}")
    
    async def _fetch_symbol_klines(self, session: aiohttp.ClientSession, symbol: str):
        """Obtiene klines históricos para un símbolo"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',
                'limit': 100  # Últimas 100 velas
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                with self.data_lock:
                    for kline_raw in data:
                        kline = KlineData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(kline_raw[0] / 1000),
                            open=float(kline_raw[1]),
                            high=float(kline_raw[2]),
                            low=float(kline_raw[3]),
                            close=float(kline_raw[4]),
                            volume=float(kline_raw[5]),
                            timeframe='1m'
                        )
                        self.klines[symbol].append(kline)
                        
        except Exception as e:
            logger.warning(f"Error obteniendo klines para {symbol}: {e}")
    
    # API Pública para obtener datos
    
    def get_market_snapshot(self) -> MarketSnapshot:
        """Obtiene snapshot completo del mercado"""
        with self.data_lock:
            # Top gainers/losers
            sorted_tickers = sorted(
                self.tickers.values(), 
                key=lambda t: t.change_24h, 
                reverse=True
            )
            
            top_gainers = [t.symbol for t in sorted_tickers[:10]]
            top_losers = [t.symbol for t in sorted_tickers[-10:]]
            
            # High volume
            high_volume = sorted(
                self.tickers.values(),
                key=lambda t: t.volume_24h,
                reverse=True
            )[:10]
            high_volume_symbols = [t.symbol for t in high_volume]
            
            return MarketSnapshot(
                timestamp=datetime.now(),
                tickers=self.tickers.copy(),
                top_gainers=top_gainers,
                top_losers=top_losers,
                high_volume=high_volume_symbols,
                total_pairs=len(self.tickers)
            )
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Obtiene datos completos de un símbolo"""
        with self.data_lock:
            if symbol not in self.tickers:
                return None
                
            ticker = self.tickers[symbol]
            klines_list = list(self.klines[symbol])
            
            return {
                'ticker': ticker,
                'klines': klines_list,
                'klines_count': len(klines_list)
            }
    
    def get_top_movers(self, limit: int = 20) -> Dict[str, List[str]]:
        """Obtiene los principales movers del mercado"""
        with self.data_lock:
            sorted_tickers = sorted(
                self.tickers.values(),
                key=lambda t: abs(t.change_24h),
                reverse=True
            )
            
            gainers = [t.symbol for t in sorted_tickers if t.change_24h > 0][:limit]
            losers = [t.symbol for t in sorted_tickers if t.change_24h < 0][:limit]
            
            return {
                'gainers': gainers,
                'losers': losers
            }
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del agregador"""
        with self.data_lock:
            return {
                **self.stats,
                'active_pairs': len(self.tickers),
                'total_klines': sum(len(klines) for klines in self.klines.values())
            }
    
    def register_callback(self, event: str, callback: Callable):
        """Registra callback para eventos"""
        self.callbacks[event].append(callback)
    
    async def close(self):
        """Cierra todas las conexiones"""
        logger.info("🛑 Cerrando Binance WebSocket Aggregator...")
        
        self.is_running = False
        
        # Cerrar WebSockets
        for ws in self.ws_connections.values():
            await ws.close()
        
        # Cancelar tareas
        if hasattr(self, 'ws_tasks'):
            for task in self.ws_tasks:
                task.cancel()
        
        # Cerrar executor
        self.executor.shutdown(wait=True)
        
        logger.info("✅ Aggregator cerrado")

# Función de testing
async def test_advanced_aggregator():
    """Test del nuevo agregador"""
    from src.utils.config_manager import ConfigManager
    
    config = ConfigManager()
    aggregator = BinanceWebSocketAggregator(config)
    
    # Callback de ejemplo
    def on_ticker_update(symbol: str, ticker: TickerData):
        if ticker.change_24h > 5:  # Solo mostrar cambios > 5%
            print(f"📈 {symbol}: ${ticker.price:.4f} ({ticker.change_24h:+.2f}%)")
    
    aggregator.register_callback('ticker_update', on_ticker_update)
    
    try:
        await aggregator.initialize()
        
        # Ejecutar por 30 segundos
        await asyncio.sleep(30)
        
        # Mostrar estadísticas
        stats = aggregator.get_stats()
        print(f"\n📊 Estadísticas:")
        print(f"Mensajes recibidos: {stats['messages_received']}")
        print(f"Pares activos: {stats['active_pairs']}")
        print(f"WebSockets conectados: {stats['websockets_connected']}")
        
        # Mostrar top movers
        movers = aggregator.get_top_movers(5)
        print(f"\n🚀 Top Gainers: {movers['gainers']}")
        print(f"📉 Top Losers: {movers['losers']}")
        
    finally:
        await aggregator.close()

if __name__ == "__main__":
    asyncio.run(test_advanced_aggregator())
