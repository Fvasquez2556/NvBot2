"""
Sistema de notificaciones simple para demo (sin dependencias externas)
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class AlertMessage:
    type: str
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: str = "normal"

class SimpleNotificationSystem:
    """
    Sistema de notificaciones simplificado para demo
    Solo usa logs y consola - sin dependencias externas
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.enabled = config.get('notifications.enabled', True)
        self.console_only = config.get('notifications.console_only', True)
        
    async def initialize(self):
        """Inicializa el sistema de notificaciones"""
        logger.info("🔔 Simple Notification System inicializado (modo demo)")
        return True
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Envía una alerta usando solo logs
        """
        try:
            if not self.enabled:
                return True
            
            alert = AlertMessage(
                type=alert_data.get('type', 'info'),
                title=alert_data.get('title', 'Notification'),
                message=alert_data.get('message', ''),
                data=alert_data.get('data', {}),
                timestamp=datetime.now(),
                priority=alert_data.get('priority', 'normal')
            )
            
            # Log de la notificación
            log_message = f"[{alert.type.upper()}] {alert.title}: {alert.message}"
            
            if alert.type == 'error':
                logger.error(log_message)
            elif alert.type == 'warning':
                logger.warning(log_message)
            elif alert.type == 'trade':
                logger.info(f"💰 {log_message}")
            elif alert.type == 'summary':
                logger.info(f"📊 {log_message}")
            else:
                logger.info(f"ℹ️ {log_message}")
            
            # Si hay datos adicionales, mostrarlos
            if alert.data:
                logger.info(f"   Datos: {json.dumps(alert.data, indent=2, default=str)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {e}")
            return False
    
    async def send_trade_notification(self, trade_data: Dict[str, Any]) -> bool:
        """Envía notificación específica de trade"""
        return await self.send_alert({
            'type': 'trade',
            'title': f"Trade: {trade_data.get('symbol', 'Unknown')}",
            'message': f"{trade_data.get('side', 'Unknown')} {trade_data.get('amount', 0)} @ ${trade_data.get('price', 0):.2f}",
            'data': trade_data
        })
    
    async def send_portfolio_summary(self, summary: Dict[str, Any]) -> bool:
        """Envía resumen del portfolio"""
        return await self.send_alert({
            'type': 'summary',
            'title': 'Portfolio Summary',
            'message': f"Balance: ${summary.get('total_balance', 0):.2f} | PnL: ${summary.get('total_pnl', 0):.2f}",
            'data': summary
        })
    
    async def close(self):
        """Cierra el sistema de notificaciones"""
        logger.info("🔔 Simple Notification System cerrado")

# Función para usar el sistema simple en lugar del complejo
def get_notification_system(config: ConfigManager):
    """Factory function que retorna el sistema de notificaciones apropiado"""
    try:
        # Intentar importar el sistema completo
        from src.utils.notification_system import NotificationSystem
        return NotificationSystem(config)
    except ImportError:
        # Si falla, usar el sistema simple
        logger.info("📱 Usando sistema de notificaciones simplificado (demo mode)")
        return SimpleNotificationSystem(config)
