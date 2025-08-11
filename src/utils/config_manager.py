"""
Gestor de configuración del bot
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """Gestor de configuración centralizada"""
    
    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_file = Path(config_file)
        self._config = {}
        self._load_config()
        self._load_environment_variables()
    
    def _load_config(self):
        """Carga la configuración desde archivo YAML"""
        if self.config_file.exists():
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Configuración por defecto
            self._config = self._get_default_config()
    
    def _load_environment_variables(self):
        """Carga variables de entorno para configuración sensible"""
        env_mappings = {
            'BINANCE_API_KEY': 'exchanges.binance.api_key',
            'BINANCE_SECRET_KEY': 'exchanges.binance.secret_key',
            'COINBASE_API_KEY': 'exchanges.coinbase.api_key',
            'COINBASE_SECRET_KEY': 'exchanges.coinbase.secret_key',
            'TELEGRAM_BOT_TOKEN': 'notifications.telegram.bot_token',
            'TELEGRAM_CHAT_ID': 'notifications.telegram.chat_id'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_config(config_path, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración
        
        Args:
            key: Clave en formato 'section.subsection.key'
            default: Valor por defecto si no existe la clave
        
        Returns:
            Valor de configuración
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Establece un valor de configuración
        
        Args:
            key: Clave en formato 'section.subsection.key'
            value: Valor a establecer
        """
        self._set_nested_config(key, value)
    
    def _set_nested_config(self, key: str, value: Any):
        """Establece un valor en configuración anidada"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def _get_default_config(self) -> Dict:
        """Retorna configuración por defecto"""
        return {
            'bot': {
                'name': 'MomentumPredictorBot',
                'version': '1.0.0',
                'cycle_interval': 60,
                'trading_pairs': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            },
            'momentum': {
                'rsi_period': 14,
                'volume_threshold': 1.5,
                'price_change_threshold': 0.02,
                'prediction_horizon': 24
            },
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'max_daily_trades': 10
            },
            'exchanges': {
                'binance': {
                    'enabled': True,
                    'testnet': True
                },
                'coinbase': {
                    'enabled': False,
                    'sandbox': True
                }
            },
            'ml_models': {
                'retrain_interval': 168,  # horas
                'min_training_samples': 1000,
                'model_type': 'transformer'
            },
            'notifications': {
                'enabled': True,
                'telegram': {
                    'enabled': False
                }
            }
        }
    
    def save_config(self, file_path: Optional[str] = None):
        """Guarda la configuración actual a archivo"""
        target_file = Path(file_path) if file_path else self.config_file
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_file, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
