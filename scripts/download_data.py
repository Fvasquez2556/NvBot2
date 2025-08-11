#!/usr/bin/env python3
"""
Script para descargar datos históricos de mercado
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config_manager import ConfigManager
from utils.logger import setup_logger
from data_sources.binance_client import BinanceClient


async def main():
    """Función principal para descargar datos"""
    logger = setup_logger("download_data", "download_data.log")
    logger.info("Iniciando descarga de datos históricos...")
    
    try:
        # Cargar configuración
        config = ConfigManager()
        
        # Obtener pares de trading
        trading_pairs = config.get("bot.trading_pairs", ["BTC/USDT", "ETH/USDT"])
        
        # Configurar cliente de Binance
        binance_client = BinanceClient(config)
        
        # Configurar fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 año de datos
        
        for pair in trading_pairs:
            logger.info(f"Descargando datos para {pair}...")
            
            try:
                # Descargar datos de 1h
                data_1h = await binance_client.get_historical_data(
                    symbol=pair,
                    timeframe="1h",
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Guardar datos
                output_file = Path("data/raw") / f"{pair.replace('/', '_')}_1h.csv"
                data_1h.to_csv(output_file, index=False)
                logger.info(f"Guardado: {output_file} ({len(data_1h)} registros)")
                
                # Descargar datos de 4h
                data_4h = await binance_client.get_historical_data(
                    symbol=pair,
                    timeframe="4h",
                    start_date=start_date,
                    end_date=end_date
                )
                
                output_file_4h = Path("data/raw") / f"{pair.replace('/', '_')}_4h.csv"
                data_4h.to_csv(output_file_4h, index=False)
                logger.info(f"Guardado: {output_file_4h} ({len(data_4h)} registros)")
                
            except Exception as e:
                logger.error(f"Error descargando datos para {pair}: {e}")
                continue
        
        logger.info("Descarga de datos completada")
        
    except Exception as e:
        logger.error(f"Error en descarga de datos: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
