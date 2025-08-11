"""
Clase base para todas las estrategias de trading
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BaseStrategy(ABC):
    """Clase base abstracta para estrategias de trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = self.__class__.__name__
        
    @abstractmethod
    async def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """
        Genera señales de trading basadas en los datos de mercado
        
        Args:
            data: DataFrame con datos de mercado
            
        Returns:
            Lista de señales de trading
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Dict) -> bool:
        """
        Valida una señal de trading
        
        Args:
            signal: Diccionario con información de la señal
            
        Returns:
            True si la señal es válida, False en caso contrario
        """
        pass
    
    def get_risk_parameters(self) -> Dict:
        """Retorna los parámetros de gestión de riesgo"""
        return self.config.get("risk_management", {})
