#!/usr/bin/env python3
"""
Entrenamiento simplificado pero efectivo con los nuevos datos
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_and_combine_data():
    """Carga y combina datos de forma simple"""
    print("📂 Cargando datos históricos...")
    
    all_data = []
    timeframes = ['15m', '1h', '4h', '1d']
    
    for tf in timeframes:
        tf_dir = Path(f"data/raw/{tf}")
        if not tf_dir.exists():
            continue
            
        files = list(tf_dir.glob("*.csv"))
        print(f"  📊 {tf}: {len(files)} archivos")
        
        for file in files:
            try:
                df = pd.read_csv(file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                
                symbol = file.stem.replace(f"_{tf}", "")
                df['symbol'] = symbol
                df['timeframe'] = tf
                
                all_data.append(df)
                
            except Exception as e:
                print(f"    ⚠️  Error en {file}: {e}")
    
    if not all_data:
        print("❌ No se pudieron cargar datos")
        return None
    
    combined = pd.concat(all_data, ignore_index=False)
    print(f"✅ Total datos cargados: {len(combined):,} velas")
    
    return combined


def create_simple_features(df):
    """Crea features simples pero efectivos"""
    print("🔧 Creando features...")
    
    # Ordenar por timestamp para mantener secuencia temporal
    df = df.sort_index()
    
    # Features básicos
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ma_ratio'] = df['sma_5'] / df['sma_20']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Volatilidad
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Volume features
    df['volume_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # High-Low features
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Target
    df['target'] = df['close'].shift(-1)
    
    # Limpiar datos
    feature_cols = ['returns', 'log_returns', 'ma_ratio', 'rsi', 'volatility', 
                   'volume_ratio', 'hl_ratio']
    
    # Eliminar NaN
    clean_mask = ~(df[feature_cols].isnull().any(axis=1) | df['target'].isnull())
    df_clean = df[clean_mask].copy()
    
    print(f"✅ Features creados: {len(feature_cols)}")
    print(f"📊 Datos limpios: {len(df_clean):,} muestras")
    
    return df_clean, feature_cols


def train_improved_models():
    """Entrenamiento mejorado"""
    print("🎯 ENTRENAMIENTO MEJORADO CON NUEVOS DATOS")
    print("=" * 45)
    
    # Cargar datos
    combined_data = load_and_combine_data()
    if combined_data is None:
        return False
    
    # Crear features
    featured_data, feature_cols = create_simple_features(combined_data)
    
    if len(featured_data) < 100:
        print("❌ Datos insuficientes después de limpieza")
        return False
    
    # Preparar datos
    X = featured_data[feature_cols]
    y = featured_data['target']
    
    print(f"\n📊 DATASET:")
    print(f"  📈 Muestras: {len(X):,}")
    print(f"  🔧 Features: {len(feature_cols)}")
    print(f"  💰 Símbolos: {featured_data['symbol'].nunique()}")
    print(f"  ⏰ Timeframes: {list(featured_data['timeframe'].unique())}")
    
    # División temporal
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"\n📋 DIVISIÓN:")
    print(f"  🏃 Train: {len(X_train):,}")
    print(f"  🧪 Test: {len(X_test):,}")
    
    # Escalado
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n🤖 MODELOS:")
    print("-" * 25)
    
    # Random Forest conservador
    print("🌲 Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=30,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    
    rf_model.fit(X_train_scaled, y_train)
    rf_pred_train = rf_model.predict(X_train_scaled)
    rf_pred_test = rf_model.predict(X_test_scaled)
    
    rf_train_r2 = r2_score(y_train, rf_pred_train)
    rf_test_r2 = r2_score(y_test, rf_pred_test)
    rf_mae = mean_absolute_error(y_test, rf_pred_test)
    
    print(f"   Train R²: {rf_train_r2:.3f}")
    print(f"   Test R²:  {rf_test_r2:.3f}")
    print(f"   MAE: {rf_mae:.4f}")
    print(f"   Diff: {abs(rf_train_r2 - rf_test_r2):.3f}")
    
    # Ridge
    print(f"\n🏔️ Ridge...")
    ridge_model = Ridge(alpha=10.0)
    
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred_train = ridge_model.predict(X_train_scaled)
    ridge_pred_test = ridge_model.predict(X_test_scaled)
    
    ridge_train_r2 = r2_score(y_train, ridge_pred_train)
    ridge_test_r2 = r2_score(y_test, ridge_pred_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred_test)
    
    print(f"   Train R²: {ridge_train_r2:.3f}")
    print(f"   Test R²:  {ridge_test_r2:.3f}")
    print(f"   MAE: {ridge_mae:.4f}")
    print(f"   Diff: {abs(ridge_train_r2 - ridge_test_r2):.3f}")
    
    # Ensemble
    ensemble_pred = (rf_pred_test + ridge_pred_test) / 2
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"\n🧠 Ensemble:")
    print(f"   R²: {ensemble_r2:.3f}")
    print(f"   MAE: {ensemble_mae:.4f}")
    
    # Guardar modelos
    print(f"\n💾 Guardando...")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf_model, models_dir / "rf_improved.joblib")
    joblib.dump(ridge_model, models_dir / "ridge_improved.joblib")
    joblib.dump(scaler, models_dir / "scaler_improved.joblib")
    
    # Información del modelo
    model_info = {
        'feature_columns': feature_cols,
        'models': {
            'random_forest': {
                'train_r2': rf_train_r2,
                'test_r2': rf_test_r2,
                'mae': rf_mae
            },
            'ridge': {
                'train_r2': ridge_train_r2,
                'test_r2': ridge_test_r2,
                'mae': ridge_mae
            },
            'ensemble': {
                'r2': ensemble_r2,
                'mae': ensemble_mae
            }
        },
        'training_date': datetime.now().isoformat(),
        'data_info': {
            'total_samples': len(X),
            'symbols': featured_data['symbol'].nunique(),
            'timeframes': list(featured_data['timeframe'].unique())
        }
    }
    
    with open(models_dir / "improved_model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Modelos guardados")
    
    # Evaluación
    print(f"\n🔍 EVALUACIÓN:")
    print("-" * 20)
    
    def check_health(train_r2, test_r2, model_name):
        if test_r2 < 0:
            status = "🚨 PREDICCIÓN INVERSA"
            health = False
        elif test_r2 < 0.01:
            status = "❌ MUY BAJO"
            health = False
        elif abs(train_r2 - test_r2) > 0.2:
            status = "🚨 OVERFITTING"
            health = False
        elif test_r2 > 0.1:
            status = "✅ BUENO"
            health = True
        else:
            status = "⚠️  MODERADO"
            health = True
        
        print(f"{model_name}: {status}")
        print(f"  Test R²: {test_r2:.3f}")
        return health
    
    rf_ok = check_health(rf_train_r2, rf_test_r2, "Random Forest")
    ridge_ok = check_health(ridge_train_r2, ridge_test_r2, "Ridge")
    
    # Resultado final
    print(f"\n🎯 RESULTADO:")
    print("-" * 15)
    
    best_r2 = max(rf_test_r2, ridge_test_r2, ensemble_r2)
    
    if best_r2 > 0.1:
        print(f"🎉 ¡ÉXITO! R² = {best_r2:.3f}")
        print(f"📈 Modelo explica {best_r2*100:.1f}% de varianza")
        print(f"🚀 Listo para testing avanzado")
        success = True
    elif best_r2 > 0.05:
        print(f"⚠️  MODERADO. R² = {best_r2:.3f}")
        print(f"📈 Modelo explica {best_r2*100:.1f}% de varianza")
        print(f"🔧 Considera más optimización")
        success = True
    else:
        print(f"❌ BAJO. R² = {best_r2:.3f}")
        print(f"📊 Modelo necesita mejoras")
        print(f"🔄 Revisar estrategia")
        success = False
    
    return success


if __name__ == "__main__":
    success = train_improved_models()
    if success:
        print(f"\n✅ Entrenamiento exitoso")
    else:
        print(f"\n❌ Necesita mejoras")
        exit(1)
