#!/usr/bin/env python3
"""
Archivo principal del bot de predicción de momentum
"""

import asyncio
import logging
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.config_manager import ConfigManager
from strategies.momentum_predictor import MomentumPredictor
from data_sources.data_aggregator import DataAggregator
from live_trading.portfolio_manager import PortfolioManager


async def main():
    """Función principal del bot"""
    # Configurar logging
    logger = setup_logger("main")
    logger.info("Iniciando Momentum Predictor Bot...")
    
    try:
        # Cargar configuración
        config = ConfigManager()
        
        # Inicializar componentes
        data_aggregator = DataAggregator(config)
        momentum_predictor = MomentumPredictor(config)
        portfolio_manager = PortfolioManager(config)
        
        # Bucle principal del bot
        while True:
            try:
                # Obtener datos de mercado
                market_data = await data_aggregator.get_latest_data()
                
                # Generar predicciones
                signals = await momentum_predictor.predict(market_data)
                
                # Ejecutar estrategias de trading
                if signals:
                    await portfolio_manager.execute_signals(signals)
                
                # Esperar antes del próximo ciclo
                await asyncio.sleep(config.get("bot.cycle_interval", 60))
                
            except KeyboardInterrupt:
                logger.info("Deteniendo bot por solicitud del usuario...")
                break
            except Exception as e:
                logger.error(f"Error en el bucle principal: {e}")
                await asyncio.sleep(10)
                
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
