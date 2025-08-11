# Momentum Predictor Bot

Un bot avanzado de trading de criptomonedas que utiliza machine learning para predecir momentum de mercado y ejecutar estrategias de trading automatizadas.

## Características Principales

- 🤖 **Predicción de Momentum**: Modelos de ML para predecir cambios de momentum
- 📊 **Análisis Técnico Avanzado**: Múltiples indicadores técnicos y análisis de volumen
- 🔄 **Trading Automatizado**: Ejecución automática de órdenes basada en señales
- 📈 **Backtesting**: Sistema completo de pruebas históricas
- 🛡️ **Gestión de Riesgo**: Controles avanzados de riesgo y money management
- 📱 **Notificaciones**: Alertas en tiempo real vía Telegram
- 🐳 **Multi-Exchange**: Soporte para Binance y Coinbase Pro

## Estructura del Proyecto

```
momentum_predictor_bot/
├── src/                    # Código fuente principal
│   ├── strategies/         # Estrategias de trading
│   ├── indicators/         # Indicadores técnicos
│   ├── ml_models/         # Modelos de machine learning
│   ├── utils/             # Utilidades y helpers
│   ├── data_sources/      # Conectores de exchanges
│   ├── backtesting/       # Motor de backtesting
│   └── live_trading/      # Trading en vivo
├── config/                # Archivos de configuración
├── data/                  # Datos de mercado y modelos
├── notebooks/             # Jupyter notebooks para análisis
├── tests/                 # Tests unitarios y de integración
├── scripts/               # Scripts de utilidad
└── docs/                  # Documentación
```

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip o conda
- Git

### Instalación Paso a Paso

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/momentum-predictor-bot.git
cd momentum-predictor-bot
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**
```bash
cp .env.example .env
# Editar .env con tus credenciales de API
```

5. **Configurar el bot**
```bash
# Editar config/config.yaml según tus necesidades
```

## Configuración

### APIs de Exchanges

1. **Binance**
   - Crear cuenta en [Binance](https://binance.com)
   - Generar API Key y Secret
   - Habilitar trading (si vas a usar en producción)

2. **Coinbase Pro**
   - Crear cuenta en [Coinbase Pro](https://pro.coinbase.com)
   - Generar API credentials

### Variables de Entorno

Edita el archivo `.env` con tus credenciales:

```env
BINANCE_API_KEY=tu_api_key
BINANCE_SECRET_KEY=tu_secret_key
TELEGRAM_BOT_TOKEN=tu_bot_token
ENVIRONMENT=development
```

## Uso

### Modo Desarrollo (Testnet)

```bash
python src/main.py
```

### Entrenar Modelos

```bash
python scripts/train_models.py
```

### Descargar Datos Históricos

```bash
python scripts/download_data.py
```

### Ejecutar Backtesting

```bash
# Usar Jupyter notebooks en notebooks/backtesting/
jupyter lab notebooks/backtesting/
```

## Estrategias Implementadas

### 1. Momentum Detector
- Detecta cambios de momentum usando RSI, volumen y precio
- Genera señales de compra/venta basadas en múltiples indicadores

### 2. ML Predictor
- Utiliza modelos Transformer para predecir momentum futuro
- Incorpora análisis de sentimiento y datos on-chain

## Gestión de Riesgo

- **Stop Loss**: Pérdidas máximas por posición
- **Take Profit**: Objetivos de ganancia automáticos
- **Position Sizing**: Tamaño de posición basado en volatilidad
- **Portfolio Risk**: Control de riesgo total del portfolio

## Testing

```bash
# Ejecutar todos los tests
pytest

# Tests con cobertura
pytest --cov=src

# Tests específicos
pytest tests/unit/test_strategies.py
```

## Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Roadmap

- [ ] Implementar más modelos de ML (LSTM, GAN)
- [ ] Soporte para más exchanges (Kraken, Huobi)
- [ ] Dashboard web para monitoreo
- [ ] Análisis de sentimiento de redes sociales
- [ ] Trading de futuros y opciones
- [ ] Estrategias de arbitraje

## Advertencias

⚠️ **IMPORTANTE**: 
- Este bot es para fines educativos y de investigación
- El trading de criptomonedas conlleva riesgos significativos
- Siempre usa cuentas de prueba antes de trading real
- Nunca inviertas más de lo que puedes permitirte perder

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## Soporte

Para soporte y preguntas:
- Abrir un [Issue](https://github.com/tu-usuario/momentum-predictor-bot/issues)
- Documentación: [Wiki](https://github.com/tu-usuario/momentum-predictor-bot/wiki)
- Email: tu.email@example.com

---

⭐ Si este proyecto te resulta útil, ¡no olvides darle una estrella!
