# @claude: Sistema completo de notificaciones que incluya:
# 1. Telegram bot para alertas
# 2. Email notifications
# 3. Webhook integrations
# 4. Dashboard real-time
# 5. Performance reporting
# 6. Error monitoring

import asyncio
from typing import Dict, List, Optional, Any
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import aiohttp
from dataclasses import dataclass, asdict
import os
from pathlib import Path

from telegram import Bot
from telegram.error import TelegramError
import requests

from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class AlertMessage:
    type: str  # 'momentum_signal', 'trade_executed', 'risk_warning', 'system_error'
    priority: str  # 'low', 'medium', 'high', 'critical'
    title: str
    message: str
    symbol: str = None
    price: float = None
    confidence: float = None
    additional_data: Dict = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.additional_data is None:
            self.additional_data = {}

@dataclass
class PerformanceReport:
    period: str  # 'daily', 'weekly', 'monthly'
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    best_trade: Dict
    worst_trade: Dict
    active_positions: int
    top_performers: List[Dict]

class NotificationSystem:
    """
    Sistema completo de notificaciones multi-canal
    """
    
    def __init__(self, config: Dict = None):
        self.config = ConfigManager().get_notification_config() if not config else config
        
        # Configuración de canales
        self.telegram_enabled = self.config.get('telegram', {}).get('enabled', False)
        self.email_enabled = self.config.get('email', {}).get('enabled', False)
        self.webhook_enabled = self.config.get('webhook', {}).get('enabled', False)
        self.dashboard_enabled = self.config.get('dashboard', {}).get('enabled', True)
        
        # Inicializar canales
        self.telegram_bot = None
        self.email_config = None
        self.webhook_urls = []
        
        # Estado interno
        self.alert_history: List[AlertMessage] = []
        self.performance_cache = {}
        self.dashboard_data = {
            'last_update': datetime.now(),
            'alerts': [],
            'performance': {},
            'system_status': {}
        }
        
        # Rate limiting
        self.rate_limits = {
            'telegram': {'count': 0, 'reset_time': datetime.now()},
            'email': {'count': 0, 'reset_time': datetime.now()},
            'webhook': {'count': 0, 'reset_time': datetime.now()}
        }
        
        self._initialize_channels()
        logger.info("NotificationSystem initialized")

    def _initialize_channels(self):
        """
        Inicializa todos los canales de notificación
        """
        # Telegram Bot
        if self.telegram_enabled:
            try:
                telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
                if telegram_token:
                    self.telegram_bot = Bot(token=telegram_token)
                    logger.info("Telegram bot initialized")
                else:
                    logger.warning("Telegram token not found in environment")
            except Exception as e:
                logger.error(f"Error initializing Telegram bot: {e}")
                self.telegram_enabled = False
        
        # Email configuration
        if self.email_enabled:
            try:
                self.email_config = {
                    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                    'email': os.getenv('EMAIL_ADDRESS'),
                    'password': os.getenv('EMAIL_PASSWORD'),
                    'recipients': os.getenv('EMAIL_RECIPIENTS', '').split(',')
                }
                if self.email_config['email'] and self.email_config['password']:
                    logger.info("Email notifications configured")
                else:
                    logger.warning("Email credentials not found")
                    self.email_enabled = False
            except Exception as e:
                logger.error(f"Error configuring email: {e}")
                self.email_enabled = False
        
        # Webhook URLs
        if self.webhook_enabled:
            webhook_urls = os.getenv('WEBHOOK_URLS', '').split(',')
            self.webhook_urls = [url.strip() for url in webhook_urls if url.strip()]
            if self.webhook_urls:
                logger.info(f"Configured {len(self.webhook_urls)} webhook URLs")
            else:
                logger.warning("No webhook URLs configured")
                self.webhook_enabled = False

    async def send_momentum_alert(self, signal_data: Dict):
        """
        Envía alerta de momentum detectado
        """
        alert = AlertMessage(
            type='momentum_signal',
            priority='high' if signal_data.get('confidence', 0) > 0.8 else 'medium',
            title=f"🚀 Momentum Signal: {signal_data.get('symbol', 'Unknown')}",
            message=self._format_momentum_message(signal_data),
            symbol=signal_data.get('symbol'),
            price=signal_data.get('price'),
            confidence=signal_data.get('confidence'),
            additional_data=signal_data
        )
        
        await self._dispatch_alert(alert)

    async def send_trade_alert(self, trade_data: Dict):
        """
        Envía alerta de trade ejecutado
        """
        action_emoji = "📈" if trade_data.get('action') == 'buy' else "📉"
        
        alert = AlertMessage(
            type='trade_executed',
            priority='medium',
            title=f"{action_emoji} Trade Executed: {trade_data.get('symbol', 'Unknown')}",
            message=self._format_trade_message(trade_data),
            symbol=trade_data.get('symbol'),
            price=trade_data.get('price'),
            additional_data=trade_data
        )
        
        await self._dispatch_alert(alert)

    async def send_risk_warning(self, risk_data: Dict):
        """
        Envía advertencia de riesgo
        """
        alert = AlertMessage(
            type='risk_warning',
            priority='high',
            title=f"⚠️ Risk Warning: {risk_data.get('type', 'Unknown Risk')}",
            message=self._format_risk_message(risk_data),
            additional_data=risk_data
        )
        
        await self._dispatch_alert(alert)

    async def send_system_error(self, error_data: Dict):
        """
        Envía alerta de error del sistema
        """
        alert = AlertMessage(
            type='system_error',
            priority='critical',
            title=f"🔥 System Error: {error_data.get('component', 'Unknown')}",
            message=self._format_error_message(error_data),
            additional_data=error_data
        )
        
        await self._dispatch_alert(alert)

    async def send_performance_report(self, report: PerformanceReport):
        """
        Envía reporte de performance
        """
        alert = AlertMessage(
            type='performance_report',
            priority='low',
            title=f"📊 {report.period.title()} Performance Report",
            message=self._format_performance_message(report),
            additional_data=asdict(report)
        )
        
        await self._dispatch_alert(alert)

    async def _dispatch_alert(self, alert: AlertMessage):
        """
        Distribuye alerta a todos los canales habilitados
        """
        try:
            # Agregar a historial
            self.alert_history.append(alert)
            self._cleanup_alert_history()
            
            # Actualizar dashboard
            self._update_dashboard(alert)
            
            # Verificar rate limits
            if not self._check_rate_limits(alert):
                logger.warning(f"Rate limit exceeded for alert: {alert.title}")
                return
            
            # Enviar a canales según prioridad
            tasks = []
            
            if self.telegram_enabled and self._should_send_to_channel('telegram', alert):
                tasks.append(self._send_telegram(alert))
            
            if self.email_enabled and self._should_send_to_channel('email', alert):
                tasks.append(self._send_email(alert))
            
            if self.webhook_enabled and self._should_send_to_channel('webhook', alert):
                tasks.append(self._send_webhook(alert))
            
            # Ejecutar en paralelo
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Error sending alert via channel {i}: {result}")
            
            logger.info(f"Alert dispatched: {alert.title}")
            
        except Exception as e:
            logger.error(f"Error dispatching alert: {e}")

    async def _send_telegram(self, alert: AlertMessage) -> bool:
        """
        Envía mensaje via Telegram
        """
        try:
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            if not chat_id or not self.telegram_bot:
                return False
            
            message = self._format_telegram_message(alert)
            
            await self.telegram_bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='HTML'
            )
            
            # Update rate limit
            self._update_rate_limit('telegram')
            
            logger.debug(f"Telegram message sent: {alert.title}")
            return True
            
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False

    async def _send_email(self, alert: AlertMessage) -> bool:
        """
        Envía mensaje via Email
        """
        try:
            if not self.email_config['email'] or not self.email_config['recipients']:
                return False
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"[Momentum Bot] {alert.title}"
            
            body = self._format_email_message(alert)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_email_sync, msg)
            
            # Update rate limit
            self._update_rate_limit('email')
            
            logger.debug(f"Email sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def _send_email_sync(self, msg: MIMEMultipart):
        """
        Envía email de forma sincrónica
        """
        with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
            server.starttls()
            server.login(self.email_config['email'], self.email_config['password'])
            server.send_message(msg)

    async def _send_webhook(self, alert: AlertMessage) -> bool:
        """
        Envía mensaje via Webhook
        """
        try:
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.type,
                'priority': alert.priority,
                'title': alert.title,
                'message': alert.message,
                'symbol': alert.symbol,
                'price': alert.price,
                'confidence': alert.confidence,
                'additional_data': alert.additional_data
            }
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                for url in self.webhook_urls:
                    task = session.post(url, json=payload, timeout=10)
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update rate limit
            self._update_rate_limit('webhook')
            
            logger.debug(f"Webhook sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False

    def _format_momentum_message(self, signal_data: Dict) -> str:
        """
        Formatea mensaje de momentum signal
        """
        return (
            f"Symbol: {signal_data.get('symbol', 'N/A')}\n"
            f"Price: ${signal_data.get('price', 0):.4f}\n"
            f"Confidence: {signal_data.get('confidence', 0):.1%}\n"
            f"Signal Strength: {signal_data.get('signal_strength', 'N/A')}\n"
            f"Timeframe: {signal_data.get('dominant_timeframe', 'N/A')}\n"
            f"Expected Return: {signal_data.get('expected_return', 0):.1%}\n"
            f"Risk Level: {signal_data.get('risk_level', 'N/A')}"
        )

    def _format_trade_message(self, trade_data: Dict) -> str:
        """
        Formatea mensaje de trade
        """
        return (
            f"Action: {trade_data.get('action', 'N/A').upper()}\n"
            f"Symbol: {trade_data.get('symbol', 'N/A')}\n"
            f"Price: ${trade_data.get('price', 0):.4f}\n"
            f"Quantity: {trade_data.get('quantity', 0):.6f}\n"
            f"Position Size: {trade_data.get('position_size', 0):.1%}\n"
            f"Stop Loss: ${trade_data.get('stop_loss', 0):.4f}\n"
            f"Take Profit: ${trade_data.get('take_profit', 0):.4f}"
        )

    def _format_risk_message(self, risk_data: Dict) -> str:
        """
        Formatea mensaje de riesgo
        """
        return (
            f"Risk Type: {risk_data.get('type', 'N/A')}\n"
            f"Severity: {risk_data.get('severity', 'N/A')}\n"
            f"Description: {risk_data.get('description', 'N/A')}\n"
            f"Current Exposure: {risk_data.get('exposure', 0):.1%}\n"
            f"Recommended Action: {risk_data.get('recommendation', 'N/A')}"
        )

    def _format_error_message(self, error_data: Dict) -> str:
        """
        Formatea mensaje de error
        """
        return (
            f"Component: {error_data.get('component', 'N/A')}\n"
            f"Error: {error_data.get('error', 'N/A')}\n"
            f"Timestamp: {error_data.get('timestamp', datetime.now())}\n"
            f"Impact: {error_data.get('impact', 'Unknown')}\n"
            f"Suggested Fix: {error_data.get('suggested_fix', 'Check logs')}"
        )

    def _format_performance_message(self, report: PerformanceReport) -> str:
        """
        Formatea mensaje de performance
        """
        return (
            f"Period: {report.start_date.strftime('%Y-%m-%d')} to {report.end_date.strftime('%Y-%m-%d')}\n"
            f"Total Trades: {report.total_trades}\n"
            f"Win Rate: {report.win_rate:.1%}\n"
            f"Total PnL: ${report.total_pnl:.2f}\n"
            f"Sharpe Ratio: {report.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {report.max_drawdown:.1%}\n"
            f"Active Positions: {report.active_positions}"
        )

    def _format_telegram_message(self, alert: AlertMessage) -> str:
        """
        Formatea mensaje específicamente para Telegram
        """
        priority_emoji = {
            'low': '🟢',
            'medium': '🟡', 
            'high': '🟠',
            'critical': '🔴'
        }
        
        emoji = priority_emoji.get(alert.priority, '⚪')
        
        return (
            f"{emoji} <b>{alert.title}</b>\n\n"
            f"{alert.message}\n\n"
            f"<i>Time: {alert.timestamp.strftime('%H:%M:%S')}</i>"
        )

    def _format_email_message(self, alert: AlertMessage) -> str:
        """
        Formatea mensaje específicamente para Email
        """
        return f"""
        <html>
        <body>
            <h2>{alert.title}</h2>
            <p><strong>Priority:</strong> {alert.priority.upper()}</p>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <br>
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                <pre>{alert.message}</pre>
            </div>
            <br>
            <small>Generated by Momentum Predictor Bot</small>
        </body>
        </html>
        """

    def _should_send_to_channel(self, channel: str, alert: AlertMessage) -> bool:
        """
        Determina si enviar alerta a canal específico según prioridad
        """
        channel_priorities = {
            'telegram': ['medium', 'high', 'critical'],
            'email': ['high', 'critical'],
            'webhook': ['low', 'medium', 'high', 'critical']
        }
        
        return alert.priority in channel_priorities.get(channel, [])

    def _check_rate_limits(self, alert: AlertMessage) -> bool:
        """
        Verifica límites de rate limiting
        """
        now = datetime.now()
        
        # Reset rate limits cada hora
        for channel, limit_data in self.rate_limits.items():
            if now - limit_data['reset_time'] > timedelta(hours=1):
                limit_data['count'] = 0
                limit_data['reset_time'] = now
        
        # Límites por prioridad
        max_alerts_per_hour = {
            'low': 10,
            'medium': 20,
            'high': 50,
            'critical': 100
        }
        
        max_allowed = max_alerts_per_hour.get(alert.priority, 10)
        
        # Verificar cada canal
        for channel in ['telegram', 'email', 'webhook']:
            if self.rate_limits[channel]['count'] >= max_allowed:
                return False
        
        return True

    def _update_rate_limit(self, channel: str):
        """
        Actualiza contador de rate limit
        """
        self.rate_limits[channel]['count'] += 1

    def _cleanup_alert_history(self):
        """
        Limpia historial de alertas antiguas
        """
        cutoff_time = datetime.now() - timedelta(days=7)  # Mantener 7 días
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert.timestamp > cutoff_time
        ]

    def _update_dashboard(self, alert: AlertMessage):
        """
        Actualiza datos del dashboard
        """
        self.dashboard_data['last_update'] = datetime.now()
        self.dashboard_data['alerts'].append({
            'timestamp': alert.timestamp.isoformat(),
            'type': alert.type,
            'priority': alert.priority,
            'title': alert.title,
            'symbol': alert.symbol
        })
        
        # Mantener solo últimas 50 alertas en dashboard
        self.dashboard_data['alerts'] = self.dashboard_data['alerts'][-50:]

    async def generate_daily_report(self) -> PerformanceReport:
        """
        Genera reporte diario de performance
        """
        # En implementación real, obtendría datos de la base de datos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        report = PerformanceReport(
            period='daily',
            start_date=start_date,
            end_date=end_date,
            total_trades=10,  # Simulated
            winning_trades=7,  # Simulated
            total_pnl=250.50,  # Simulated
            win_rate=0.70,
            sharpe_ratio=1.45,
            max_drawdown=0.08,
            best_trade={'symbol': 'BTCUSDT', 'pnl': 85.30},
            worst_trade={'symbol': 'ETHUSDT', 'pnl': -25.10},
            active_positions=3,
            top_performers=[
                {'symbol': 'BTCUSDT', 'return': 0.035},
                {'symbol': 'ADAUSDT', 'return': 0.028}
            ]
        )
        
        return report

    async def get_dashboard_data(self) -> Dict:
        """
        Obtiene datos para dashboard en tiempo real
        """
        return {
            **self.dashboard_data,
            'system_status': await self._get_system_status(),
            'alert_summary': self._get_alert_summary()
        }

    async def _get_system_status(self) -> Dict:
        """
        Obtiene status del sistema
        """
        return {
            'telegram_bot': 'online' if self.telegram_enabled else 'disabled',
            'email_service': 'online' if self.email_enabled else 'disabled',
            'webhook_service': 'online' if self.webhook_enabled else 'disabled',
            'dashboard': 'online' if self.dashboard_enabled else 'disabled',
            'uptime': '99.9%',  # Simulated
            'last_health_check': datetime.now().isoformat()
        }

    def _get_alert_summary(self) -> Dict:
        """
        Obtiene resumen de alertas
        """
        now = datetime.now()
        last_24h = [alert for alert in self.alert_history if now - alert.timestamp < timedelta(hours=24)]
        
        summary = {
            'total_24h': len(last_24h),
            'by_type': {},
            'by_priority': {}
        }
        
        for alert in last_24h:
            summary['by_type'][alert.type] = summary['by_type'].get(alert.type, 0) + 1
            summary['by_priority'][alert.priority] = summary['by_priority'].get(alert.priority, 0) + 1
        
        return summary

    async def test_notifications(self) -> Dict[str, bool]:
        """
        Prueba todos los canales de notificación
        """
        test_alert = AlertMessage(
            type='system_test',
            priority='low',
            title='🧪 Notification Test',
            message='This is a test message to verify notification channels are working properly.'
        )
        
        results = {}
        
        if self.telegram_enabled:
            results['telegram'] = await self._send_telegram(test_alert)
        
        if self.email_enabled:
            results['email'] = await self._send_email(test_alert)
        
        if self.webhook_enabled:
            results['webhook'] = await self._send_webhook(test_alert)
        
        logger.info(f"Notification test results: {results}")
        return results
