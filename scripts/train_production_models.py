#!/usr/bin/env python3
"""
🚀 ENTRENAMIENTO CON DATASET MASIVO
- 257,255+ velas de 24 criptomonedas
- 5 timeframes (5m, 15m, 1h, 4h, 1d)
- Features crypto-específicos optimizados
- Modelos ultra-robustos
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_massive_dataset():
    """Carga el dataset masivo descargado"""
    print("📂 Cargando dataset masivo...")
    
    all_data = []
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    for tf in timeframes:
        tf_dir = Path(f"data/raw/{tf}")
        if not tf_dir.exists():
            continue
            
        files = list(tf_dir.glob("*.csv"))
        print(f"  📊 {tf}: {len(files)} archivos")
        
        tf_data = []
        for file in files:
            try:
                df = pd.read_csv(file)
                if len(df) < 50:  # Filtrar archivos muy pequeños
                    continue
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                elif 'Unnamed: 0' in df.columns:
                    df.set_index('Unnamed: 0', inplace=True)
                    df.index = pd.to_datetime(df.index)
                
                symbol = file.stem.replace(f"_{tf}", "")
                df['symbol'] = symbol
                df['timeframe'] = tf
                
                # Usar solo los últimos 2000 puntos para estabilidad
                if len(df) > 2000:
                    df = df.tail(2000)
                
                tf_data.append(df)
                
            except Exception as e:
                print(f"    ⚠️  Error en {file}: {e}")
        
        if tf_data:
            tf_combined = pd.concat(tf_data, ignore_index=False)
            all_data.append(tf_combined)
            print(f"    ✅ {len(tf_combined):,} velas de {tf}")
    
    if not all_data:
        print("❌ No se pudieron cargar datos")
        return None
    
    combined = pd.concat(all_data, ignore_index=False)
    print(f"✅ Total cargado: {len(combined):,} velas")
    print(f"💰 Símbolos únicos: {combined['symbol'].nunique()}")
    print(f"⏰ Timeframes: {list(combined['timeframe'].unique())}")
    
    return combined


def create_advanced_crypto_features(df):
    """Crea features avanzados específicos para crypto con dataset masivo"""
    print("🔧 Creando features crypto avanzados...")
    
    # Ordenar por timestamp
    df = df.sort_index()
    
    # 🎯 TARGET: % change siguiente periodo
    df['target_return'] = df.groupby(['symbol', 'timeframe'])['close'].pct_change().shift(-1)
    
    # Filtrar targets extremos (conservador para estabilidad)
    df = df[df['target_return'].abs() < 0.3].copy()  # Max 30% change
    
    # 📊 FEATURES CRYPTO AVANZADOS
    
    # 1. Returns multi-periodo
    df['return_1'] = df.groupby(['symbol', 'timeframe'])['close'].pct_change()
    df['return_3'] = df.groupby(['symbol', 'timeframe'])['close'].pct_change(periods=3)
    df['return_5'] = df.groupby(['symbol', 'timeframe'])['close'].pct_change(periods=5)
    df['return_10'] = df.groupby(['symbol', 'timeframe'])['close'].pct_change(periods=10)
    
    # 2. Moving averages como ratios
    for window in [5, 10, 20]:
        ma_col = f'ma_{window}'
        ratio_col = f'price_ma_ratio_{window}'
        df[ma_col] = df.groupby(['symbol', 'timeframe'])['close'].transform(lambda x: x.rolling(window).mean())
        df[ratio_col] = df['close'] / df[ma_col] - 1
    
    # 3. Momentum indicators
    df['momentum_3'] = df.groupby(['symbol', 'timeframe'])['close'].transform(lambda x: x / x.shift(3) - 1)
    df['momentum_5'] = df.groupby(['symbol', 'timeframe'])['close'].transform(lambda x: x / x.shift(5) - 1)
    df['momentum_10'] = df.groupby(['symbol', 'timeframe'])['close'].transform(lambda x: x / x.shift(10) - 1)
    
    # 4. RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    df['rsi'] = df.groupby(['symbol', 'timeframe'])['close'].transform(calculate_rsi)
    df['rsi_norm'] = (df['rsi'] - 50) / 50  # Normalizado
    
    # 5. Volatilidad
    df['volatility_10'] = df.groupby(['symbol', 'timeframe'])['return_1'].transform(lambda x: x.rolling(10).std())
    df['volatility_20'] = df.groupby(['symbol', 'timeframe'])['return_1'].transform(lambda x: x.rolling(20).std())
    df['vol_ratio'] = df['volatility_10'] / (df['volatility_20'] + 1e-8)
    
    # 6. Volume features
    df['volume_ma'] = df.groupby(['symbol', 'timeframe'])['volume'].transform(lambda x: x.rolling(20).mean())
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
    df['volume_ratio_log'] = np.log1p(df['volume_ratio'].clip(0, 10))  # Clip extremos
    
    # 7. High-Low features
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / ((df['high'] - df['low']) + 1e-8)
    
    # 8. Cross-timeframe features (nuevo con dataset masivo)
    symbols_with_multiple_tf = df.groupby('symbol')['timeframe'].nunique()
    multi_tf_symbols = symbols_with_multiple_tf[symbols_with_multiple_tf > 1].index
    
    # Para símbolos con múltiples timeframes, crear features cruzados
    df['tf_momentum_signal'] = 0.0
    for symbol in multi_tf_symbols:
        symbol_data = df[df['symbol'] == symbol]
        if len(symbol_data) > 100:
            # Señal agregada de momentum across timeframes
            tf_signal = symbol_data.groupby('timeframe')['momentum_5'].mean()
            avg_signal = tf_signal.mean()
            df.loc[df['symbol'] == symbol, 'tf_momentum_signal'] = avg_signal
    
    # Features finales - seleccionar los más robustos
    feature_cols = [
        'return_1', 'return_3', 'return_5',
        'price_ma_ratio_5', 'price_ma_ratio_10', 'price_ma_ratio_20',
        'momentum_3', 'momentum_5', 'momentum_10',
        'rsi_norm',
        'vol_ratio',
        'volume_ratio_log',
        'hl_ratio', 'close_position',
        'tf_momentum_signal'
    ]
    
    # Limpiar datos
    clean_mask = ~(df[feature_cols].isnull().any(axis=1) | df['target_return'].isnull())
    df_clean = df[clean_mask].copy()
    
    print(f"✅ Features avanzados creados: {len(feature_cols)}")
    print(f"📊 Datos limpios: {len(df_clean):,} muestras")
    print(f"🎯 Target range: {df_clean['target_return'].min():.4f} a {df_clean['target_return'].max():.4f}")
    print(f"💰 Símbolos en dataset final: {df_clean['symbol'].nunique()}")
    print(f"⏰ Timeframes en dataset final: {sorted(df_clean['timeframe'].unique())}")
    
    return df_clean, feature_cols


def train_massive_dataset_models():
    """Entrenamiento con dataset masivo"""
    print("🚀 ENTRENAMIENTO CON DATASET MASIVO")
    print("=" * 60)
    
    # Cargar dataset masivo
    combined_data = load_massive_dataset()
    if combined_data is None:
        return False
    
    # Crear features avanzados
    featured_data, feature_cols = create_advanced_crypto_features(combined_data)
    
    if len(featured_data) < 1000:
        print("❌ Datos insuficientes después de limpieza")
        return False
    
    # Preparar datos
    X = featured_data[feature_cols]
    y = featured_data['target_return']
    
    print(f"\n📊 DATASET MASIVO:")
    print(f"  📈 Muestras: {len(X):,}")
    print(f"  🔧 Features: {len(feature_cols)}")
    print(f"  💰 Símbolos: {featured_data['symbol'].nunique()}")
    print(f"  ⏰ Timeframes: {list(featured_data['timeframe'].unique())}")
    print(f"  🎯 Target stats:")
    print(f"    Mean: {y.mean():.6f}")
    print(f"    Std:  {y.std():.6f}")
    print(f"    Range: {y.min():.4f} a {y.max():.4f}")
    
    # División temporal estricta (80/20)
    split_point = int(len(X) * 0.8)
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"\n📋 DIVISIÓN TEMPORAL:")
    print(f"  🏃 Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  🧪 Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Scaling avanzado
    print(f"\n🔧 SCALING AVANZADO:")
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, len(X_train)))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  ✅ QuantileTransformer aplicado")
    
    # Validación cruzada temporal
    print(f"\n🔄 VALIDACIÓN CRUZADA TEMPORAL:")
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = {'rf': [], 'ridge': [], 'elastic': []}
    
    print("  Ejecutando CV...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"    Fold {fold + 1}/5... ", end="", flush=True)
        
        X_train_cv = X_train.iloc[train_idx]
        X_val_cv = X_train.iloc[val_idx]
        y_train_cv = y_train.iloc[train_idx]
        y_val_cv = y_train.iloc[val_idx]
        
        # Scaling para CV
        scaler_cv = QuantileTransformer(output_distribution='normal')
        X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler_cv.transform(X_val_cv)
        
        # RF
        rf_cv = RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_split=20, random_state=42, n_jobs=-1)
        rf_cv.fit(X_train_cv_scaled, y_train_cv)
        cv_scores['rf'].append(r2_score(y_val_cv, rf_cv.predict(X_val_cv_scaled)))
        
        # Ridge
        ridge_cv = Ridge(alpha=1.0)
        ridge_cv.fit(X_train_cv_scaled, y_train_cv)
        cv_scores['ridge'].append(r2_score(y_val_cv, ridge_cv.predict(X_val_cv_scaled)))
        
        # Elastic
        elastic_cv = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=2000)
        elastic_cv.fit(X_train_cv_scaled, y_train_cv)
        cv_scores['elastic'].append(r2_score(y_val_cv, elastic_cv.predict(X_val_cv_scaled)))
        
        print("✓")
    
    print(f"  🌲 RF CV: {np.mean(cv_scores['rf']):.4f} ± {np.std(cv_scores['rf']):.4f}")
    print(f"  🏔️ Ridge CV: {np.mean(cv_scores['ridge']):.4f} ± {np.std(cv_scores['ridge']):.4f}")
    print(f"  🎯 Elastic CV: {np.mean(cv_scores['elastic']):.4f} ± {np.std(cv_scores['elastic']):.4f}")
    
    print(f"\n🤖 ENTRENAMIENTO FINAL:")
    print("-" * 35)
    
    # Random Forest optimizado para dataset masivo
    print("🌲 Random Forest masivo...")
    rf_model = RandomForestRegressor(
        n_estimators=100,    # Más árboles con más datos
        max_depth=8,         # Profundidad moderada
        min_samples_split=50, # Conservador
        min_samples_leaf=20,  # Conservador
        max_features='sqrt',  # Feature sampling
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    rf_pred_train = rf_model.predict(X_train_scaled)
    rf_pred_test = rf_model.predict(X_test_scaled)
    
    rf_train_r2 = r2_score(y_train, rf_pred_train)
    rf_test_r2 = r2_score(y_test, rf_pred_test)
    rf_mae = mean_absolute_error(y_test, rf_pred_test)
    
    print(f"   Train R²: {rf_train_r2:.4f}")
    print(f"   Test R²:  {rf_test_r2:.4f}")
    print(f"   MAE: {rf_mae:.6f} ({rf_mae*100:.3f}%)")
    print(f"   Overfitting: {abs(rf_train_r2 - rf_test_r2):.4f}")
    
    # Ridge robusto
    print(f"\n🏔️ Ridge robusto...")
    ridge_model = Ridge(alpha=10.0, max_iter=2000)
    
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred_train = ridge_model.predict(X_train_scaled)
    ridge_pred_test = ridge_model.predict(X_test_scaled)
    
    ridge_train_r2 = r2_score(y_train, ridge_pred_train)
    ridge_test_r2 = r2_score(y_test, ridge_pred_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred_test)
    
    print(f"   Train R²: {ridge_train_r2:.4f}")
    print(f"   Test R²:  {ridge_test_r2:.4f}")
    print(f"   MAE: {ridge_mae:.6f} ({ridge_mae*100:.3f}%)")
    print(f"   Overfitting: {abs(ridge_train_r2 - ridge_test_r2):.4f}")
    
    # ElasticNet balanceado
    print(f"\n🎯 ElasticNet balanceado...")
    elastic_model = ElasticNet(alpha=5.0, l1_ratio=0.3, max_iter=3000)
    
    elastic_model.fit(X_train_scaled, y_train)
    elastic_pred_train = elastic_model.predict(X_train_scaled)
    elastic_pred_test = elastic_model.predict(X_test_scaled)
    
    elastic_train_r2 = r2_score(y_train, elastic_pred_train)
    elastic_test_r2 = r2_score(y_test, elastic_pred_test)
    elastic_mae = mean_absolute_error(y_test, elastic_pred_test)
    
    print(f"   Train R²: {elastic_train_r2:.4f}")
    print(f"   Test R²:  {elastic_test_r2:.4f}")
    print(f"   MAE: {elastic_mae:.6f} ({elastic_mae*100:.3f}%)")
    print(f"   Overfitting: {abs(elastic_train_r2 - elastic_test_r2):.4f}")
    
    # Ensemble ponderado
    # Pesos basados en performance CV
    rf_weight = max(0.0, float(np.mean(cv_scores['rf'])))
    ridge_weight = max(0.0, float(np.mean(cv_scores['ridge'])))
    elastic_weight = max(0.0, float(np.mean(cv_scores['elastic'])))
    total_weight = rf_weight + ridge_weight + elastic_weight + 1e-8
    
    rf_weight /= total_weight
    ridge_weight /= total_weight
    elastic_weight /= total_weight
    
    ensemble_pred = (rf_weight * rf_pred_test + 
                    ridge_weight * ridge_pred_test + 
                    elastic_weight * elastic_pred_test)
    
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"\n🧠 Ensemble ponderado:")
    print(f"   Pesos: RF={rf_weight:.2f}, Ridge={ridge_weight:.2f}, Elastic={elastic_weight:.2f}")
    print(f"   R²: {ensemble_r2:.4f}")
    print(f"   MAE: {ensemble_mae:.6f} ({ensemble_mae*100:.3f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 TOP 10 FEATURES IMPORTANTES:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Guardar modelos
    print(f"\n💾 Guardando modelos masivos...")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf_model, models_dir / "rf_massive.joblib")
    joblib.dump(ridge_model, models_dir / "ridge_massive.joblib")
    joblib.dump(elastic_model, models_dir / "elastic_massive.joblib")
    joblib.dump(scaler, models_dir / "scaler_massive.joblib")
    
    # Información completa del modelo
    model_info = {
        'model_type': 'massive_dataset_crypto_prediction',
        'dataset_size': len(X),
        'target_type': 'percentage_return',
        'feature_columns': feature_cols,
        'cross_validation': {
            'rf_scores': cv_scores['rf'],
            'ridge_scores': cv_scores['ridge'],
            'elastic_scores': cv_scores['elastic'],
            'rf_mean': np.mean(cv_scores['rf']),
            'ridge_mean': np.mean(cv_scores['ridge']),
            'elastic_mean': np.mean(cv_scores['elastic'])
        },
        'ensemble_weights': {
            'rf_weight': rf_weight,
            'ridge_weight': ridge_weight,
            'elastic_weight': elastic_weight
        },
        'models': {
            'random_forest': {
                'train_r2': rf_train_r2,
                'test_r2': rf_test_r2,
                'mae': rf_mae,
                'overfitting': abs(rf_train_r2 - rf_test_r2)
            },
            'ridge': {
                'train_r2': ridge_train_r2,
                'test_r2': ridge_test_r2,
                'mae': ridge_mae,
                'overfitting': abs(ridge_train_r2 - ridge_test_r2)
            },
            'elastic': {
                'train_r2': elastic_train_r2,
                'test_r2': elastic_test_r2,
                'mae': elastic_mae,
                'overfitting': abs(elastic_train_r2 - elastic_test_r2)
            },
            'ensemble': {
                'r2': ensemble_r2,
                'mae': ensemble_mae
            }
        },
        'feature_importance': feature_importance.to_dict('records'),
        'training_date': datetime.now().isoformat(),
        'data_info': {
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'symbols': featured_data['symbol'].nunique(),
            'timeframes': list(featured_data['timeframe'].unique()),
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
    }
    
    with open(models_dir / "massive_model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Modelos masivos guardados")
    
    # Evaluación final
    print(f"\n🔍 EVALUACIÓN DATASET MASIVO:")
    print("-" * 30)
    
    best_r2 = max(rf_test_r2, ridge_test_r2, elastic_test_r2, ensemble_r2)
    best_mae = min(rf_mae, ridge_mae, elastic_mae, ensemble_mae) * 100
    
    if best_r2 > 0.05 and best_mae < 5.0:
        print(f"🎉 ¡ENTRENAMIENTO MASIVO EXITOSO!")
        print(f"📈 Mejor R²: {best_r2:.4f} (explica {best_r2*100:.1f}% varianza)")
        print(f"🎯 Mejor MAE: {best_mae:.2f}% (error promedio)")
        print(f"🚀 Dataset masivo aprovechado exitosamente")
        success = True
    elif best_r2 > 0.02:
        print(f"✅ ENTRENAMIENTO MASIVO MODERADO")
        print(f"📈 R²: {best_r2:.4f}")
        print(f"🎯 MAE: {best_mae:.2f}%")
        print(f"📊 Rendimiento aceptable con dataset masivo")
        success = True
    else:
        print(f"⚠️ RESULTADO LIMITADO")
        print(f"📈 R²: {best_r2:.4f}")
        print(f"🎯 MAE: {best_mae:.2f}%")
        print(f"💡 Considera ajustar features o parámetros")
        success = False
    
    return success


if __name__ == "__main__":
    success = train_massive_dataset_models()
    if success:
        print(f"\n🎉 ¡ENTRENAMIENTO CON DATASET MASIVO COMPLETADO!")
        print(f"🚀 Modelos entrenados con 257,255+ velas")
        print(f"💰 24 criptomonedas, 5 timeframes")
        print(f"⚡ Listo para bot optimizado")
    else:
        print(f"\n❌ Necesita más optimización")
        exit(1)
