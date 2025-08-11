# @claude: Configurar fixtures globales para testing que incluyan:
# 1. Mock data para diferentes market conditions
# 2. Dummy ML models para testing
# 3. Database fixtures
# 4. API mocks para exchanges

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from typing import Dict, List

from src.strategies.multi_timeframe_analyzer import TimeframeSignal, ConfluenceResult
from src.strategies.momentum_predictor_strategy import TradeSignal, RiskMetrics
from src.utils.notification_system import AlertMessage

@pytest.fixture
def sample_market_data():
    """
    Genera datos de mercado realistas para testing
    """
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
    
    # Simular precio con trend y volatilidad realista
    np.random.seed(42)  # Para reproducibilidad
    returns = np.random.normal(0.0001, 0.02, 1000)  # 0.01% mean return, 2% volatility
    
    # Crear momentum periods ocasionales
    momentum_periods = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
    returns = returns + momentum_periods * np.random.normal(0.01, 0.005, 1000)
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Crear OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.999, 1.001, 1000),
        'high': prices * np.random.uniform(1.001, 1.02, 1000),
        'low': prices * np.random.uniform(0.98, 0.999, 1000),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    # Asegurar que high >= max(open, close) y low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

@pytest.fixture
def sample_timeframe_data():
    """
    Genera datos para múltiples timeframes
    """
    timeframes = ['3m', '5m', '15m']
    data = {}
    
    for tf in timeframes:
        periods = {'3m': 1500, '5m': 1000, '15m': 300}[tf]
        dates = pd.date_range(start='2024-01-01', periods=periods, freq=tf.replace('m', 'min'))
        
        np.random.seed(hash(tf) % 2**32)
        returns = np.random.normal(0.0001, 0.015, periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data[tf] = pd.DataFrame({
            'timestamp': dates,
            'open': prices * np.random.uniform(0.999, 1.001, periods),
            'high': prices * np.random.uniform(1.001, 1.015, periods),
            'low': prices * np.random.uniform(0.985, 0.999, periods),
            'close': prices,
            'volume': np.random.uniform(500, 5000, periods)
        })
        
        # Fix OHLC consistency
        data[tf]['high'] = np.maximum(data[tf]['high'], 
                                    np.maximum(data[tf]['open'], data[tf]['close']))
        data[tf]['low'] = np.minimum(data[tf]['low'], 
                                   np.minimum(data[tf]['open'], data[tf]['close']))
    
    return data

@pytest.fixture
def mock_momentum_signals():
    """
    Señales de momentum mockeadas para testing
    """
    return {
        'strong_bullish': {
            'confidence': 0.92,
            'momentum_score': 0.88,
            'signal_strength': 'very_strong',
            'expected_return': 0.08,
            'risk_level': 'medium'
        },
        'weak_bullish': {
            'confidence': 0.62,
            'momentum_score': 0.58,
            'signal_strength': 'moderate',
            'expected_return': 0.03,
            'risk_level': 'medium'
        },
        'bearish': {
            'confidence': 0.25,
            'momentum_score': 0.15,
            'signal_strength': 'weak',
            'expected_return': -0.02,
            'risk_level': 'high'
        },
        'neutral': {
            'confidence': 0.45,
            'momentum_score': 0.48,
            'signal_strength': 'weak',
            'expected_return': 0.001,
            'risk_level': 'low'
        }
    }

@pytest.fixture
def mock_timeframe_signals():
    """
    Señales de timeframe mockeadas
    """
    signals = {}
    timeframes = ['3m', '5m', '15m']
    
    for tf in timeframes:
        signals[tf] = TimeframeSignal(
            timeframe=tf,
            momentum_score=np.random.uniform(0.5, 0.95),
            volume_confirmation=np.random.choice([True, False], p=[0.7, 0.3]),
            technical_score=np.random.uniform(0.4, 0.9),
            ml_prediction=np.random.uniform(0.3, 0.95),
            confidence=np.random.uniform(0.6, 0.95),
            timestamp=datetime.now(),
            support_resistance_levels={
                'support': np.random.uniform(95, 100),
                'resistance': np.random.uniform(105, 110)
            },
            trend_direction=np.random.choice(['bullish', 'bearish', 'neutral'], p=[0.5, 0.2, 0.3]),
            volatility_regime=np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        )
    
    return signals

@pytest.fixture
def mock_confluence_result():
    """
    Resultado de confluencia mockeado
    """
    return ConfluenceResult(
        unified_score=0.82,
        confidence_level=0.78,
        dominant_timeframe='5m',
        signal_strength='strong',
        risk_level='medium',
        entry_recommendation=True,
        stop_loss_level=98.50,
        take_profit_levels=[103.0, 106.0, 110.0],
        max_position_size=0.04
    )

@pytest.fixture
def mock_trade_signal():
    """
    Señal de trade mockeada
    """
    return TradeSignal(
        symbol='BTCUSDT',
        action='buy',
        confidence=0.85,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit_levels=[105.0, 110.0, 115.0],
        position_size=0.05,
        reasoning='Strong momentum confluence across timeframes',
        timestamp=datetime.now(),
        risk_level='medium',
        expected_return=0.08,
        max_holding_time=timedelta(hours=24)
    )

@pytest.fixture
def mock_risk_metrics():
    """
    Métricas de riesgo mockeadas
    """
    return RiskMetrics(
        portfolio_risk=0.12,
        symbol_risk=0.35,
        correlation_risk=0.25,
        liquidity_risk=0.15,
        concentration_risk=0.30,
        total_risk_score=0.45
    )

@pytest.fixture
def mock_alert_message():
    """
    Mensaje de alerta mockeado
    """
    return AlertMessage(
        type='momentum_signal',
        priority='high',
        title='Strong Momentum Detected',
        message='BTCUSDT showing strong bullish momentum with 85% confidence',
        symbol='BTCUSDT',
        price=100.0,
        confidence=0.85,
        additional_data={'timeframe': '5m', 'volume_surge': True},
        timestamp=datetime.now()
    )

@pytest.fixture
def mock_exchange_api():
    """
    Mock del API del exchange
    """
    mock_api = Mock()
    
    # Mock price data
    mock_api.fetch_ohlcv.return_value = [
        [1640995200000, 100.0, 102.0, 98.0, 101.0, 1000.0],  # timestamp, o, h, l, c, v
        [1640995500000, 101.0, 103.0, 100.0, 102.0, 1200.0],
        [1640995800000, 102.0, 104.0, 101.0, 103.0, 900.0]
    ]
    
    # Mock ticker data
    mock_api.fetch_ticker.return_value = {
        'symbol': 'BTCUSDT',
        'last': 102.5,
        'bid': 102.4,
        'ask': 102.6,
        'high': 105.0,
        'low': 98.0,
        'volume': 10000.0
    }
    
    # Mock balance
    mock_api.fetch_balance.return_value = {
        'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0},
        'BTC': {'free': 0.1, 'used': 0.0, 'total': 0.1}
    }
    
    # Mock order placement
    mock_api.create_market_buy_order.return_value = {
        'id': '12345',
        'symbol': 'BTCUSDT',
        'amount': 0.01,
        'price': 102.5,
        'status': 'closed',
        'filled': 0.01
    }
    
    return mock_api

@pytest.fixture
def mock_database():
    """
    Mock de la base de datos
    """
    db_mock = Mock()
    
    # Mock trade history
    db_mock.get_trades.return_value = [
        {
            'id': 1,
            'symbol': 'BTCUSDT',
            'action': 'buy',
            'quantity': 0.01,
            'price': 100.0,
            'timestamp': datetime.now() - timedelta(hours=2),
            'pnl': 2.5
        }
    ]
    
    # Mock performance metrics
    db_mock.get_performance_metrics.return_value = {
        'total_trades': 10,
        'winning_trades': 7,
        'total_pnl': 150.0,
        'max_drawdown': 0.08,
        'sharpe_ratio': 1.45
    }
    
    return db_mock

@pytest.fixture
def mock_ml_model():
    """
    Mock del modelo de ML
    """
    model_mock = Mock()
    
    # Mock predictions
    model_mock.predict.return_value = np.array([0.85])  # High confidence prediction
    model_mock.predict_proba.return_value = np.array([[0.15, 0.85]])  # [bearish, bullish]
    
    # Mock feature importance
    model_mock.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Mock model metrics
    model_mock.score.return_value = 0.88  # 88% accuracy
    
    return model_mock

@pytest.fixture
def mock_notification_channels():
    """
    Mock de los canales de notificación
    """
    channels = {
        'telegram': AsyncMock(),
        'email': AsyncMock(),
        'webhook': AsyncMock()
    }
    
    # Configure successful sends
    for channel in channels.values():
        channel.send.return_value = True
    
    return channels

@pytest.fixture
def sample_portfolio_state():
    """
    Estado de portfolio para testing
    """
    return {
        'balance': 10000.0,
        'positions': [
            {
                'symbol': 'BTCUSDT',
                'quantity': 0.1,
                'entry_price': 100.0,
                'current_price': 102.0,
                'pnl': 20.0,
                'entry_time': datetime.now() - timedelta(hours=1)
            },
            {
                'symbol': 'ETHUSDT',
                'quantity': 2.0,
                'entry_price': 50.0,
                'current_price': 51.5,
                'pnl': 3.0,
                'entry_time': datetime.now() - timedelta(hours=2)
            }
        ],
        'open_orders': [],
        'daily_pnl': 23.0,
        'total_pnl': 150.0
    }

@pytest.fixture
def market_conditions():
    """
    Diferentes condiciones de mercado para testing
    """
    return {
        'bull_market': {
            'trend': 'bullish',
            'volatility': 'medium',
            'volume': 'high',
            'momentum_strength': 0.8
        },
        'bear_market': {
            'trend': 'bearish',
            'volatility': 'high',
            'volume': 'medium',
            'momentum_strength': 0.3
        },
        'sideways_market': {
            'trend': 'neutral',
            'volatility': 'low',
            'volume': 'low',
            'momentum_strength': 0.4
        },
        'volatile_market': {
            'trend': 'neutral',
            'volatility': 'very_high',
            'volume': 'high',
            'momentum_strength': 0.6
        }
    }

@pytest.fixture
def test_config():
    """
    Configuración para testing
    """
    return {
        'strategy': {
            'momentum_predictor': {
                'max_portfolio_risk': 0.15,
                'max_position_risk': 0.05,
                'min_confidence': 0.70,
                'min_confluence_score': 0.65,
                'max_positions': 5,
                'timeframes': ['3m', '5m', '15m']
            }
        },
        'notifications': {
            'telegram': {'enabled': True},
            'email': {'enabled': True},
            'webhook': {'enabled': True}
        },
        'ml_models': {
            'ensemble_predictor': {
                'models': ['xgboost', 'lightgbm', 'transformer'],
                'confidence_threshold': 0.75
            }
        }
    }

@pytest.fixture(scope="session")
def event_loop():
    """
    Crea event loop para tests asyncio
    """
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Fixtures para diferentes símbolos
@pytest.fixture
def btc_data():
    """Datos específicos de Bitcoin"""
    return {
        'symbol': 'BTCUSDT',
        'current_price': 45000.0,
        'volatility': 0.04,  # 4% diario
        'volume_24h': 1000000.0,
        'market_cap_rank': 1
    }

@pytest.fixture
def eth_data():
    """Datos específicos de Ethereum"""
    return {
        'symbol': 'ETHUSDT',
        'current_price': 3000.0,
        'volatility': 0.05,  # 5% diario
        'volume_24h': 500000.0,
        'market_cap_rank': 2
    }

@pytest.fixture
def alt_coin_data():
    """Datos de altcoin para testing"""
    return {
        'symbol': 'ADAUSDT',
        'current_price': 1.50,
        'volatility': 0.08,  # 8% diario
        'volume_24h': 100000.0,
        'market_cap_rank': 10
    }

# Helper fixtures
@pytest.fixture
def freeze_time():
    """Congela el tiempo para testing determinístico"""
    from freezegun import freeze_time as _freeze_time
    test_time = datetime(2024, 1, 15, 10, 30, 0)
    with _freeze_time(test_time) as frozen_time:
        yield frozen_time

@pytest.fixture
def temp_data_dir(tmp_path):
    """Directorio temporal para datos de testing"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Limpieza automática después de cada test"""
    yield
    # Cleanup code here if needed
    pass
