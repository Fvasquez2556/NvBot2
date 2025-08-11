"""
Data Aggregator - Obtiene y procesa datos de múltiples exchanges
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import ccxt.async_support as ccxt
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    exchange: str

@dataclass
class TickerData:
    symbol: str
    last_price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    timestamp: datetime

class DataAggregator:
    """
    Agregador de datos que obtiene información de múltiples exchanges
    y proporciona datos unificados para el bot
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.exchanges = {}
        self.cache = {}
        self.cache_duration = timedelta(minutes=1)
        
    async def initialize(self):
        """Inicializa conexiones a exchanges"""
        try:
            # Configurar exchanges
            exchange_configs = self.config.get('exchanges', {})
            
            for exchange_name, config in exchange_configs.items():
                if exchange_name == 'binance' and config.get('enabled', False):
                    exchange = ccxt.binance({
                        'apiKey': config.get('api_key', ''),
                        'secret': config.get('secret_key', ''),
                        'sandbox': config.get('sandbox', True),
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'spot'  # spot, future, margin
                        }
                    })
                    self.exchanges['binance'] = exchange
                    
            logger.info(f"Inicializados {len(self.exchanges)} exchanges")
            
        except Exception as e:
            logger.error(f"Error inicializando exchanges: {e}")
            raise
    
    async def get_latest_data(self, symbols: Optional[List[str]] = None) -> Dict[str, List[MarketData]]:
        """
        Obtiene los datos más recientes para los símbolos especificados
        """
        if symbols is None:
            symbols = self.config.get('bot.trading_pairs', ['BTC/USDT'])
            
        data = {}
        
        for symbol in symbols:
            cache_key = f"latest_{symbol}"
            
            # Verificar cache
            if self._is_cache_valid(cache_key):
                data[symbol] = self.cache[cache_key]['data']
                continue
                
            try:
                symbol_data = []
                
                # Obtener datos de cada exchange
                for exchange_name, exchange in self.exchanges.items():
                    try:
                        # Obtener OHLCV data (1 minuto)
                        ohlcv = await exchange.fetch_ohlcv(
                            symbol, 
                            timeframe='1m', 
                            limit=100
                        )
                        
                        for candle in ohlcv[-10:]:  # Últimas 10 velas
                            market_data = MarketData(
                                symbol=symbol,
                                timestamp=datetime.fromtimestamp(candle[0] / 1000),
                                open=float(candle[1]),
                                high=float(candle[2]),
                                low=float(candle[3]),
                                close=float(candle[4]),
                                volume=float(candle[5]),
                                timeframe='1m',
                                exchange=exchange_name
                            )
                            symbol_data.append(market_data)
                            
                    except Exception as e:
                        logger.warning(f"Error obteniendo datos de {exchange_name} para {symbol}: {e}")
                        
                data[symbol] = symbol_data
                
                # Actualizar cache
                self.cache[cache_key] = {
                    'data': symbol_data,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Error obteniendo datos para {symbol}: {e}")
                data[symbol] = []
                
        return data
    
    async def get_ticker_data(self, symbols: Optional[List[str]] = None) -> Dict[str, TickerData]:
        """Obtiene datos de ticker en tiempo real"""
        if symbols is None:
            symbols = self.config.get('bot.trading_pairs', ['BTC/USDT'])
            
        ticker_data = {}
        
        for symbol in symbols:
            try:
                # Usar el primer exchange disponible
                exchange = list(self.exchanges.values())[0]
                ticker = await exchange.fetch_ticker(symbol)
                
                ticker_data[symbol] = TickerData(
                    symbol=symbol,
                    last_price=float(ticker['last']),
                    bid=float(ticker['bid']) if ticker['bid'] else 0,
                    ask=float(ticker['ask']) if ticker['ask'] else 0,
                    volume_24h=float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0,
                    change_24h=float(ticker['percentage']) if ticker['percentage'] else 0,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                logger.error(f"Error obteniendo ticker para {symbol}: {e}")
                
        return ticker_data
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Obtiene datos históricos para backtesting y entrenamiento de ML
        """
        try:
            exchange = list(self.exchanges.values())[0]
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos para {symbol}: {e}")
            return pd.DataFrame()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica si los datos en cache son válidos"""
        if cache_key not in self.cache:
            return False
            
        cache_time = self.cache[cache_key]['timestamp']
        return datetime.now() - cache_time < self.cache_duration
    
    async def close(self):
        """Cierra conexiones a exchanges"""
        for exchange in self.exchanges.values():
            await exchange.close()
            
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Función helper para testing
async def test_data_aggregator():
    """Función de prueba para el data aggregator"""
    from src.utils.config_manager import ConfigManager
    
    config = ConfigManager()
    
    async with DataAggregator(config) as aggregator:
        # Obtener datos recientes
        latest_data = await aggregator.get_latest_data(['BTC/USDT'])
        print(f"Datos obtenidos para BTC/USDT: {len(latest_data.get('BTC/USDT', []))} velas")
        
        # Obtener ticker
        ticker_data = await aggregator.get_ticker_data(['BTC/USDT'])
        if 'BTC/USDT' in ticker_data:
            ticker = ticker_data['BTC/USDT']
            print(f"Precio actual BTC: ${ticker.last_price:,.2f}")
            
        # Obtener datos históricos
        historical = await aggregator.get_historical_data('BTC/USDT', '1h', 100)
        print(f"Datos históricos: {len(historical)} velas")

if __name__ == "__main__":
    asyncio.run(test_data_aggregator())
