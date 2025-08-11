# 🚀 NvBot2 - AI-Powered Momentum Trading Bot

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](docker/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-orange.svg)](tests/)

**NvBot2** es un bot de trading avanzado que utiliza Inteligencia Artificial para detectar momentum en criptomonedas con alta precisión. Combina análisis técnico multi-timeframe, modelos de Machine Learning y gestión de riesgo sofisticada.

## ✨ Características Principales

### 🧠 **Análisis Multi-Timeframe Inteligente**
- Confluencia de señales entre 3m, 5m y 15m
- Detección automática de regímenes de mercado
- Scoring ponderado por timeframe
- Sistema de health monitoring

### 🤖 **Machine Learning Avanzado**
- Ensemble de modelos (XGBoost, LightGBM, Transformer)
- Predicción de momentum con >85% precisión
- Reentrenamiento automático
- Feature engineering optimizado

### 🛡️ **Gestión de Riesgo Profesional**
- Risk scoring multi-dimensional
- Position sizing dinámico
- Stop-loss y take-profit adaptativos
- Máximo drawdown controlado (<15%)

### 📱 **Sistema de Notificaciones**
- Telegram bot integrado
- Email notifications
- Webhooks para sistemas externos
- Dashboard en tiempo real

## 🏗️ Arquitectura del Sistema

```
src/
├── strategies/
│   ├── momentum_detector.py          # Detección de momentum base
│   ├── multi_timeframe_analyzer.py   # Análisis multi-timeframe
│   └── momentum_predictor_strategy.py # Estrategia principal
├── ml_models/
│   └── ensemble_predictor.py         # Modelos de ML
├── utils/
│   ├── notification_system.py        # Sistema de alertas
│   ├── config_manager.py             # Gestión de configuración
│   └── logger.py                     # Sistema de logging
└── main.py                           # Punto de entrada principal
```

## 🚀 Instalación Rápida

### Prerrequisitos
- Python 3.9+
- Docker (opcional)
- Cuenta en Binance con API keys
- Bot de Telegram (opcional)

### 1. Clonar Repositorio
```bash
git clone https://github.com/Fvasquez2556/NvBot2.git
cd NvBot2
```

### 2. Configurar Entorno Virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno
```bash
cp .env.example .env
# Editar .env con tus API keys
```

### 5. Validar Configuración
```bash
python scripts/validate_config.py
```

### 6. Ejecutar Bot
```bash
python src/main.py --env development
```

## ⚙️ Configuración

### Variables de Entorno Requeridas
```bash
# API Keys de Trading
BINANCE_API_KEY=tu_binance_api_key
BINANCE_SECRET_KEY=tu_binance_secret_key

# Notificaciones Telegram
TELEGRAM_BOT_TOKEN=tu_telegram_bot_token
TELEGRAM_CHAT_ID=tu_telegram_chat_id

# Configuración de Trading
INITIAL_BALANCE=1000
MAX_TRADES_PER_DAY=4
RISK_PER_TRADE=0.02
```

## 🧪 Testing

### Ejecutar Tests Completos
```bash
pytest tests/ -v --cov=src
```

### Probar Notificaciones
```bash
python scripts/test_notifications.py
```

### Análisis Multi-Timeframe
```bash
python scripts/analyze_timeframes.py --symbol BTCUSDT
```

## 🐳 Docker Deployment

### Deploy a Producción
```bash
./scripts/deploy.sh production
```

## 📊 Métricas de Performance

### Targets de Performance
- **Precisión**: >85% para momentum 7%+
- **Sharpe Ratio**: >2.0
- **Win Rate**: >70%
- **Max Drawdown**: <15%
- **Latencia**: <100ms end-to-end

## 🛠️ Desarrollo

### Setup para Desarrollo
```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Formatear código
black src/ tests/
flake8 src/
```

## ⚠️ Disclaimer

**Importante**: Este bot está diseñado para fines educativos y de investigación. El trading de criptomonedas conlleva riesgos significativos. Nunca inviertas más de lo que puedas permitirte perder.

---

⭐ **¡Si te gusta el proyecto, dale una estrella!** ⭐
