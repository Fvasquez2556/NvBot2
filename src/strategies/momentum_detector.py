"""
Detector de momentum en tiempo real
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from .base_strategy import BaseStrategy

@dataclass
class MomentumResult:
    direction: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    signal_strength: str  # 'weak', 'moderate', 'strong'
    indicators: Dict

class MomentumDetector(BaseStrategy):
    """Detecta patrones de momentum en los datos de mercado"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.rsi_period = config.get("momentum.rsi_period", 14)
        self.volume_threshold = config.get("momentum.volume_threshold", 1.5)
        self.price_change_threshold = config.get("momentum.price_change_threshold", 0.02)
    
    async def detect_momentum(self, data: pd.DataFrame) -> Optional[MomentumResult]:
        """
        Detecta momentum en los datos de mercado
        """
        try:
            if len(data) < self.rsi_period + 1:
                return None
                
            # Calcular indicadores
            data = self._calculate_indicators(data)
            
            # Determinar dirección del momentum
            if self._detect_bullish_momentum(data):
                direction = 'bullish'
                confidence = self._calculate_confidence(data, "bullish")
            elif self._detect_bearish_momentum(data):
                direction = 'bearish'
                confidence = self._calculate_confidence(data, "bearish")
            else:
                direction = 'neutral'
                confidence = 0.5
            
            # Determinar fuerza de la señal
            if confidence >= 0.8:
                signal_strength = 'strong'
            elif confidence >= 0.6:
                signal_strength = 'moderate'
            else:
                signal_strength = 'weak'
            
            # Recopilar indicadores
            latest = data.iloc[-1]
            indicators = {
                'rsi': latest.get('rsi', 50),
                'volume_ratio': latest.get('volume_ratio', 1),
                'price_change': latest.get('price_change', 0)
            }
            
            return MomentumResult(
                direction=direction,
                confidence=confidence,
                signal_strength=signal_strength,
                indicators=indicators
            )
            
        except Exception as e:
            print(f"Error detecting momentum: {e}")
            return None
        
    async def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Genera señales basadas en detección de momentum"""
        signals = []
        
        if len(data) < self.rsi_period + 1:
            return signals
            
        # Calcular indicadores
        data = self._calculate_indicators(data)
        
        # Detectar momentum alcista
        if self._detect_bullish_momentum(data):
            signals.append({
                "type": "BUY",
                "symbol": data["symbol"].iloc[-1],
                "price": data["close"].iloc[-1],
                "confidence": self._calculate_confidence(data, "bullish"),
                "timestamp": data["timestamp"].iloc[-1],
                "strategy": self.name
            })
            
        # Detectar momentum bajista
        elif self._detect_bearish_momentum(data):
            signals.append({
                "type": "SELL",
                "symbol": data["symbol"].iloc[-1],
                "price": data["close"].iloc[-1],
                "confidence": self._calculate_confidence(data, "bearish"),
                "timestamp": data["timestamp"].iloc[-1],
                "strategy": self.name
            })
            
        return signals
    
    def validate_signal(self, signal: Dict) -> bool:
        """Valida la señal generada"""
        required_fields = ["type", "symbol", "price", "confidence", "timestamp"]
        return all(field in signal for field in required_fields)
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos necesarios"""
        # RSI
        delta = data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))
        
        # Cambio porcentual de precio
        data["price_change"] = data["close"].pct_change()
        
        # Volumen relativo
        data["volume_avg"] = data["volume"].rolling(window=20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_avg"]
        
        return data
    
    def _detect_bullish_momentum(self, data: pd.DataFrame) -> bool:
        """Detecta momentum alcista"""
        latest = data.iloc[-1]
        
        return (
            latest["rsi"] > 50 and
            latest["price_change"] > self.price_change_threshold and
            latest["volume_ratio"] > self.volume_threshold
        )
    
    def _detect_bearish_momentum(self, data: pd.DataFrame) -> bool:
        """Detecta momentum bajista"""
        latest = data.iloc[-1]
        
        return (
            latest["rsi"] < 50 and
            latest["price_change"] < -self.price_change_threshold and
            latest["volume_ratio"] > self.volume_threshold
        )
    
    def _calculate_confidence(self, data: pd.DataFrame, direction: str) -> float:
        """Calcula el nivel de confianza de la señal"""
        latest = data.iloc[-1]
        
        if direction == "bullish":
            # Factor RSI (0-1)
            rsi_factor = min(1.0, (latest["rsi"] - 50) / 50)
            # Factor precio (0-1)
            price_factor = min(1.0, latest["price_change"] / (self.price_change_threshold * 5))
            # Factor volumen (0-1)
            volume_factor = min(1.0, (latest["volume_ratio"] - 1) / 2)
            
        else:  # bearish
            rsi_factor = min(1.0, (50 - latest["rsi"]) / 50)
            price_factor = min(1.0, abs(latest["price_change"]) / (self.price_change_threshold * 5))
            volume_factor = min(1.0, (latest["volume_ratio"] - 1) / 2)
        
        return (rsi_factor + price_factor + volume_factor) / 3
