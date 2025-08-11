# WebSocket Aggregator - Análisis Masivo de Pares USDT

## 🌍 Descripción General

El **BinanceWebSocketAggregator** es un sistema avanzado que permite analizar **TODOS los pares USDT de Binance** en tiempo real usando WebSockets, evitando completamente los límites de rate limiting de la API REST.

## 🚀 Características Principales

### ✅ Análisis Masivo
- **200+ pares USDT** analizados simultáneamente
- Descubrimiento automático de pares activos
- Filtrado por volumen mínimo (configurable)
- Selección de top pares por actividad

### ✅ WebSocket Avanzado
- **Múltiples conexiones** WebSocket (50 pares por conexión)
- **Reconexión automática** con backoff exponencial
- **Heartbeat/ping** para mantener conexiones vivas
- **Manejo de errores** robusto

### ✅ Procesamiento Paralelo
- **ThreadPoolExecutor** con hasta 20 workers
- Procesamiento **no bloqueante** de datos
- **Thread-safe** con locks apropiados
- Buffer circular para klines

### ✅ Datos en Tiempo Real
- **Ticker data**: precio, volumen, cambio 24h
- **Kline data**: OHLCV en tiempo real
- **Callbacks personalizables** para eventos
- **Market snapshots** completos

## 📊 Configuración

### config.yaml - Sección data_aggregator

```yaml
data_aggregator:
  # WebSocket configuration
  use_websockets: true
  max_reconnections: 5
  ping_interval: 20
  
  # Filtros para seleccionar pares
  min_volume_24h: 1000000    # Mínimo $1M volumen diario
  max_pairs: 200             # Máximo pares a analizar
  min_price_change: 0.5      # Mínimo 0.5% cambio
  
  # Performance optimization
  parallel_processing: true
  max_workers: 20
  chunk_size: 50             # Pares por conexión WebSocket
  update_interval: 1         # Segundos entre actualizaciones
  
  # Almacenamiento en memoria
  klines_buffer_size: 1000   # Máximo klines por par
  ticker_cache_duration: 5   # Segundos para cache
```

### Bot Configuration

```yaml
bot:
  # Modo avanzado: analizar TODOS los pares USDT
  analyze_all_usdt_pairs: true
  
  # Pares específicos (solo si analyze_all_usdt_pairs = false)
  trading_pairs:
    - "BTC/USDT"
    - "ETH/USDT"
```

## 🔧 Uso del Sistema

### 1. Inicialización Básica

```python
from data_sources.binance_websocket_aggregator import BinanceWebSocketAggregator

# Crear agregador
aggregator = BinanceWebSocketAggregator(
    max_workers=20,
    chunk_size=50,
    min_volume_24h=1000000,
    max_pairs=200
)

# Iniciar conexiones
await aggregator.start()
```

### 2. Configurar Callbacks

```python
# Callback para tickers
async def on_ticker(symbol: str, ticker: TickerData):
    print(f"{symbol}: ${ticker.last_price:.4f} ({ticker.price_change_percent:+.2f}%)")

# Callback para klines
async def on_kline(symbol: str, kline: KlineData):
    print(f"{symbol} OHLC: {kline.open_price:.4f}/{kline.high_price:.4f}/{kline.low_price:.4f}/{kline.close_price:.4f}")

# Registrar callbacks
aggregator.register_ticker_callback(on_ticker)
aggregator.register_kline_callback(on_kline)
```

### 3. Obtener Market Snapshot

```python
# Obtener snapshot completo del mercado
snapshot = await aggregator.get_market_snapshot()

print(f"Pares activos: {len(snapshot.tickers)}")
print(f"Última actualización: {snapshot.timestamp}")

# Top 10 por volumen
top_pairs = sorted(
    snapshot.tickers.items(),
    key=lambda x: x[1].volume_24h,
    reverse=True
)[:10]

for symbol, ticker in top_pairs:
    print(f"{symbol}: ${ticker.volume_24h:,.0f}")
```

## 🎯 Arquitectura del Sistema

### Componentes Principales

```
BinanceWebSocketAggregator
├── PairDiscovery
│   ├── get_all_usdt_pairs()
│   ├── filter_by_volume()
│   └── select_top_pairs()
├── ConnectionManager
│   ├── create_websocket_connections()
│   ├── manage_heartbeat()
│   └── handle_reconnections()
├── DataProcessor
│   ├── ThreadPoolExecutor
│   ├── process_ticker_data()
│   └── process_kline_data()
└── CallbackSystem
    ├── ticker_callbacks[]
    ├── kline_callbacks[]
    └── error_callbacks[]
```

### Flujo de Datos

```
1. Binance API REST → Descobrir pares USDT
2. Filtrar por volumen → Seleccionar top pares
3. Crear conexiones WebSocket → Chunked (50 pares/conexión)
4. Recibir datos streaming → Ticker + Kline
5. Procesar en paralelo → ThreadPoolExecutor
6. Disparar callbacks → Usuario recibe datos
7. Mantener conexiones → Heartbeat + Reconnect
```

## 🏃‍♂️ Scripts de Demo

### Demo WebSocket

```bash
# Ejecutar demo de 5 minutos
python scripts/websocket_demo.py
```

Este demo muestra:
- ✅ Conexión a WebSockets de Binance
- ✅ Datos en tiempo real de 50+ pares
- ✅ Estadísticas cada 30 segundos
- ✅ Top 5 pares por volumen
- ✅ Contadores de tickers/klines recibidos

### Bot Avanzado

```bash
# Ejecutar bot completo con WebSockets
python src/main_advanced.py
```

Este bot incluye:
- ✅ WebSocket aggregator completo
- ✅ Análisis de momentum en tiempo real
- ✅ Gestión de portfolio
- ✅ Notificaciones automáticas
- ✅ ML predictions

## 📈 Performance y Escalabilidad

### Métricas Esperadas

| Métrica | Valor |
|---------|-------|
| Pares simultáneos | 200+ |
| Latencia promedio | < 100ms |
| Throughput | 1000+ msg/seg |
| Memoria RAM | 200-500MB |
| CPU | 10-30% |
| Conexiones WebSocket | 4-8 |

### Optimizaciones

1. **Chunking**: 50 pares por conexión WebSocket
2. **Threading**: 20 workers para procesamiento paralelo
3. **Buffering**: Circular buffer para klines (1000 max)
4. **Caching**: Cache de tickers (5 segundos)
5. **Filtering**: Solo pares con volumen > $1M

## 🔒 Rate Limiting y Binance

### Ventajas WebSocket vs REST

| Aspecto | WebSocket | REST API |
|---------|-----------|----------|
| Rate Limits | ❌ Sin límites | ⚠️ 1200 req/min |
| Latencia | ✅ < 50ms | ⚠️ 200-500ms |
| Datos Tiempo Real | ✅ Streaming | ❌ Polling |
| Ancho de banda | ✅ Eficiente | ⚠️ Overhead |
| Pares simultáneos | ✅ 200+ | ⚠️ Limitado |

### Binance WebSocket Limits

- **Conexiones**: 5 por IP (usamos 4 máximo)
- **Streams**: 1024 por conexión (usamos 50)
- **Rate**: Sin límites en streams establecidos
- **Heartbeat**: 60 segundos máximo (usamos 20)

## 🛠️ Troubleshooting

### Problemas Comunes

1. **Conexión rechazada**
   ```python
   # Verificar IP no bloqueada
   # Reducir concurrent connections
   chunk_size = 25  # En lugar de 50
   ```

2. **Pérdida de datos**
   ```python
   # Aumentar buffer size
   klines_buffer_size = 2000
   # Verificar callbacks no bloqueantes
   ```

3. **Alta latencia**
   ```python
   # Reducir workers si CPU limitado
   max_workers = 10
   # Optimizar callbacks
   ```

4. **Desconexiones frecuentes**
   ```python
   # Aumentar ping interval
   ping_interval = 30
   # Verificar estabilidad de red
   ```

### Logs Importantes

```python
# Activar logging detallado
logging:
  level: "DEBUG"
  log_websocket_messages: true
  log_performance_metrics: true
```

## 🚦 Estados del Sistema

### Estados de Conexión

- 🟢 **CONNECTED**: Recibiendo datos normalmente
- 🟡 **RECONNECTING**: Reconectando automáticamente  
- 🔴 **DISCONNECTED**: Desconectado (error crítico)
- 🟠 **THROTTLED**: Limitado (reducir carga)

### Monitoreo en Tiempo Real

```python
# Verificar estado
status = await aggregator.get_connection_status()
print(f"Conexiones activas: {status.active_connections}")
print(f"Pares streaming: {status.streaming_pairs}")
print(f"Último heartbeat: {status.last_heartbeat}")
```

## 🎯 Casos de Uso

### Trading de Alta Frecuencia
- Análisis de 200+ pares simultáneamente
- Latencia ultra-baja para arbitraje
- Detección de breakouts en tiempo real

### Research y Backtesting  
- Datos históricos completos
- Market snapshots para análisis
- Correlaciones entre múltiples pares

### Portfolio Diversificado
- Monitoreo de todo el mercado USDT
- Rebalanceo automático basado en momentum
- Risk management cross-pair

---

## 📋 Siguiente Paso

**¡El sistema está listo para uso!** 

Ejecuta el demo para ver el agregador en acción:
```bash
python scripts/websocket_demo.py
```

O inicia el bot completo:
```bash
python src/main_advanced.py
```
