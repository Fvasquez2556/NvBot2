# 📁 Scripts Directory - Momentum Predictor Bot

## 🚀 **ARCHIVOS PRINCIPALES (PRODUCCIÓN)**

### **Bot de Trading:**

- `production_bot.py` - **Bot ultra-optimizado para trading** (MAE: 0.139%)
  - Usa modelos masivos entrenados con 176,610 muestras
  - Ensemble RandomForest + Ridge optimizado
  - Predicciones realistas (-2% a +2%)

### **Gestión de Datos:**

- `download_crypto_data.py` - **Descargador robusto de datos crypto**
  - 24 criptomonedas, 5 timeframes
  - Recuperación automática de errores
  - Rate limiting y progreso detallado

- `download_massive_data.py` - **Descargador para datasets masivos**
  - 3000+ velas por símbolo/timeframe
  - Manejo de APIs y timeouts

### **Entrenamiento de Modelos:**

- `train_production_models.py` - **Entrenador principal de modelos**
  - Dataset masivo (176,610+ muestras)
  - Validación cruzada temporal
  - Ensemble optimizado con pesos automáticos

## 🔧 **UTILIDADES Y HERRAMIENTAS**

### **Análisis y Configuración:**

- `analyze_timeframes.py` - Análisis de timeframes disponibles
- `validate_config.py` - Validación de configuración
- `setup_live_trading.py` - Configuración para trading en vivo

### **Testing y Diagnóstico:**

- `test_notifications.py` - Test de notificaciones
- `test_websocket_system.py` - Test de conexiones WebSocket
- `websocket_demo.py` - Demo de conexiones en tiempo real
- `websocket_diagnostic.py` - Diagnóstico de WebSocket

### **Demos y Ejemplos:**

- `quick_live_demo.py` - Demo rápido en vivo

## 📦 **ARCHIVOS ARCHIVADOS**

El directorio `archive/` contiene versiones anteriores y experimentales:

- Scripts de entrenamiento obsoletos
- Versiones anteriores de descargadores
- Demos experimentales

## 🎯 **USO RECOMENDADO**

### **Para Comenzar:**

1. `python download_crypto_data.py` - Descargar datos
2. `python train_production_models.py` - Entrenar modelos
3. `python production_bot.py` - Ejecutar bot optimizado

### **Para Desarrollo:**

1. Usar scripts en `archive/` como referencia
2. Modificar `production_bot.py` para nuevas funcionalidades
3. Test con `test_*.py` antes de producción

---
**Última actualización:** Agosto 2025  
**Estado:** ✅ Producción - Ultra-optimizado (MAE: 0.139%)
