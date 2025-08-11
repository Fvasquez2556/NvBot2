# Contributing to NvBot2

¡Gracias por tu interés en contribuir a NvBot2! Este documento proporciona las pautas para contribuir al proyecto.

## 🤝 Cómo Contribuir

### 1. Fork y Clone
```bash
# Fork el repositorio en GitHub
# Luego clona tu fork
git clone https://github.com/tu-usuario/NvBot2.git
cd NvBot2
```

### 2. Configurar Entorno de Desarrollo
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt
pip install -e .

# Configurar pre-commit hooks
pre-commit install
```

### 3. Crear Branch para Feature
```bash
git checkout -b feature/nombre-de-feature
```

## 📋 Guidelines de Desarrollo

### Code Style
- **Python**: Seguir PEP 8
- **Formateo**: Usar `black` para formateo automático
- **Linting**: Usar `flake8` para linting
- **Type Hints**: Incluir type hints en todas las funciones
- **Docstrings**: Documentar todas las funciones públicas

```bash
# Formatear código
black src/ tests/

# Verificar linting  
flake8 src/

# Verificar type hints
mypy src/
```

### Testing
- **Coverage**: Mantener >95% de cobertura
- **Tests**: Escribir tests para toda nueva funcionalidad
- **Fixtures**: Usar fixtures de `tests/conftest.py`
- **Mocking**: Mock APIs externas y servicios

```bash
# Ejecutar tests
pytest tests/ -v

# Con coverage
pytest tests/ --cov=src --cov-report=html

# Solo tests rápidos
pytest tests/unit/ -v
```

### Commit Messages
Usar [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new momentum detection algorithm
fix: resolve race condition in multi-timeframe analysis
docs: update API documentation
test: add integration tests for notification system
refactor: optimize risk management calculations
```

### Branch Naming
- `feature/descripcion-corta` - Nuevas características
- `fix/descripcion-del-bug` - Bug fixes
- `docs/que-se-actualiza` - Documentación
- `test/area-de-testing` - Mejoras en testing
- `refactor/componente` - Refactoring

## 🏗️ Estructura del Proyecto

### Directorios Principales
```
src/
├── strategies/     # Estrategias de trading
├── ml_models/      # Modelos de machine learning  
├── utils/          # Utilidades y helpers
├── data_sources/   # Conectores de exchanges
├── backtesting/    # Motor de backtesting
└── live_trading/   # Trading en vivo
```

### Convenciones de Naming
- **Classes**: `PascalCase` (ej: `MomentumDetector`)
- **Functions**: `snake_case` (ej: `analyze_timeframes`)
- **Constants**: `UPPER_SNAKE_CASE` (ej: `MAX_POSITIONS`)
- **Files**: `snake_case.py` (ej: `momentum_detector.py`)

## 🧪 Testing Guidelines

### Tipos de Tests
- **Unit Tests**: `tests/unit/` - Funciones individuales
- **Integration Tests**: `tests/integration/` - Componentes integrados
- **ML Tests**: Marcar con `@pytest.mark.ml`
- **Slow Tests**: Marcar con `@pytest.mark.slow`

### Writing Tests
```python
import pytest
from src.strategies.momentum_detector import MomentumDetector

class TestMomentumDetector:
    def test_detect_momentum_with_valid_data(self, sample_market_data):
        detector = MomentumDetector()
        result = detector.detect_momentum(sample_market_data)
        
        assert result is not None
        assert 0 <= result.confidence <= 1
        assert result.signal_strength in ['weak', 'moderate', 'strong']
    
    @pytest.mark.asyncio
    async def test_async_analysis(self, mock_exchange_api):
        detector = MomentumDetector()
        result = await detector.analyze_symbol("BTCUSDT")
        
        assert result.symbol == "BTCUSDT"
```

## 🚀 Deployment & CI/CD

### Pre-commit Checks
Antes de cada commit se ejecutan automáticamente:
- Black formatting
- Flake8 linting
- Basic test suite
- Security checks

### Pull Request Process
1. Ejecutar tests localmente
2. Verificar que pasan todos los checks
3. Actualizar documentación si es necesario
4. Crear PR con descripción detallada

### CI/CD Pipeline
- **Tests**: Ejecutar en Python 3.9, 3.10, 3.11
- **Coverage**: Verificar >95% coverage
- **Security**: Scan de vulnerabilidades
- **Docker**: Build y test de imagen

## 📝 Documentation

### Docstrings
Usar formato Google style:

```python
def analyze_confluence(self, symbol: str, current_price: float) -> ConfluenceResult:
    """
    Analiza confluencia entre todos los timeframes.
    
    Args:
        symbol: El símbolo a analizar (ej: 'BTCUSDT')
        current_price: Precio actual del símbolo
        
    Returns:
        ConfluenceResult con el análisis completo
        
    Raises:
        ValueError: Si el símbolo no es válido
        APIError: Si hay error conectando con exchange
        
    Example:
        >>> analyzer = MultiTimeframeAnalyzer(config)
        >>> result = await analyzer.analyze_confluence("BTCUSDT", 45000.0)
        >>> print(f"Confidence: {result.confidence_level}")
    """
```

### README Updates
- Actualizar README.md si hay cambios en API
- Incluir ejemplos de uso
- Documentar nuevas configuraciones

## 🐛 Bug Reports

### Template para Issues
```markdown
**Describe the bug**
Una descripción clara del bug.

**To Reproduce**
Pasos para reproducir:
1. Configurar '...'
2. Ejecutar '...'
3. Ver error

**Expected behavior**
Qué esperabas que pasara.

**Environment:**
- OS: [ej: Windows 10]
- Python: [ej: 3.9.7]
- Version: [ej: 1.0.0]

**Logs**
```
Incluir logs relevantes
```

## 💡 Feature Requests

### Template para Features
```markdown
**Is your feature request related to a problem?**
Descripción del problema.

**Describe the solution you'd like**
Descripción clara de lo que quieres.

**Describe alternatives you've considered**
Alternativas consideradas.

**Additional context**
Contexto adicional, screenshots, etc.
```

## 🔒 Security

### Reporting Security Issues
- **NO** crear issue público para problemas de seguridad
- Enviar email a: security@nvbot2.com
- Incluir descripción detallada del problema

### Security Guidelines
- Nunca commitear API keys o secrets
- Usar variables de entorno para configuración sensible
- Validar todas las entradas de usuario
- Sanitizar logs para evitar información sensible

## 📞 Contacto

- **Issues**: [GitHub Issues](https://github.com/Fvasquez2556/NvBot2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Fvasquez2556/NvBot2/discussions)
- **Email**: contribute@nvbot2.com

## 🏆 Contributors

Agradecemos a todos los contributors que hacen posible este proyecto:

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- Será actualizado automáticamente -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## 📄 License

Al contribuir, aceptas que tus contribuciones estarán bajo la misma [MIT License](LICENSE) que cubre el proyecto.

---

¡Gracias por contribuir a NvBot2! 🚀
