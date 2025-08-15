#!/usr/bin/env python3
"""
🚀 ENTRENAMIENTO ULTRA-OPTIMIZADO PARA CRYPTO
- Enfoque específico para trading de criptomonedas
- Features especializados para momentum y volatilidad
- Regularización extrema para evitar overfitting
- Targets realistas y estables
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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_and_combine_data():
    """Carga y combina datos optimizada para crypto"""
    print("📂 Cargando datos históricos...")
    
    all_data = []
    timeframes = ['1h', '4h', '1d']  # Solo timeframes más estables
    
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
                
                # Solo tomar datos recientes para estabilidad
                if len(df) > 500:
                    df = df.tail(500)
                
                all_data.append(df)
                
            except Exception as e:
                print(f"    ⚠️  Error en {file}: {e}")
    
    if not all_data:
        print("❌ No se pudieron cargar datos")
        return None
    
    combined = pd.concat(all_data, ignore_index=False)
    print(f"✅ Total datos cargados: {len(combined):,} velas")
    
    return combined


def create_crypto_features(df):
    """Features específicos para crypto trading"""
    print("🔧 Creando features crypto-específicos...")
    
    # Ordenar por timestamp
    df = df.sort_index()
    
    # 🎯 TARGET SIMPLE: % change pequeño y realista
    df['next_return'] = df['close'].pct_change().shift(-1)
    
    # Filtrar targets extremos (crypto es volátil pero no tanto)
    df = df[df['next_return'].abs() < 0.2].copy()  # Max 20% change
    
    # 📊 FEATURES CRYPTO-ESPECÍFICOS
    
    # 1. Returns simples y estables
    df['return_1h'] = df['close'].pct_change()
    df['return_6h'] = df['close'].pct_change(periods=6)
    df['return_24h'] = df['close'].pct_change(periods=24)
    
    # 2. Momentum crypto (más corto)
    df['momentum_short'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_medium'] = df['close'] / df['close'].shift(6) - 1
    
    # 3. MA específicos para crypto
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_ratio'] = df['ema_9'] / df['ema_21'] - 1
    
    # 4. RSI simplificado
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_signal'] = (df['rsi'] - 50) / 50  # Normalizado
    
    # 5. Volatilidad crypto
    df['volatility'] = df['return_1h'].rolling(window=24).std()
    df['vol_percentile'] = df['volatility'].rolling(window=100).rank(pct=True)
    
    # 6. Volume crypto
    df['volume_ma'] = df['volume'].rolling(window=24).mean()
    df['volume_surge'] = df['volume'] / df['volume_ma'] - 1
    df['volume_surge'] = np.clip(df['volume_surge'], -2, 5)  # Clip extremos
    
    # Features finales - SOLO 8 features más importantes
    feature_cols = [
        'return_1h',
        'return_6h', 
        'momentum_short',
        'ema_ratio',
        'rsi_signal',
        'vol_percentile',
        'volume_surge',
        'return_24h'
    ]
    
    # Limpiar datos
    clean_mask = ~(df[feature_cols].isnull().any(axis=1) | df['next_return'].isnull())
    df_clean = df[clean_mask].copy()
    
    print(f"✅ Features crypto creados: {len(feature_cols)}")
    print(f"📊 Datos limpios: {len(df_clean):,} muestras")
    print(f"🎯 Target range: {df_clean['next_return'].min():.4f} a {df_clean['next_return'].max():.4f}")
    
    return df_clean, feature_cols


def train_ultra_conservative_models():
    """Entrenamiento ultra-conservador para crypto"""
    print("🚀 ENTRENAMIENTO ULTRA-OPTIMIZADO CRYPTO")
    print("=" * 55)
    
    # Cargar datos
    combined_data = load_and_combine_data()
    if combined_data is None:
        return False
    
    # Crear features crypto
    featured_data, feature_cols = create_crypto_features(combined_data)
    
    if len(featured_data) < 100:
        print("❌ Datos insuficientes después de limpieza")
        return False
    
    # Preparar datos
    X = featured_data[feature_cols]
    y = featured_data['next_return']
    
    print(f"\n📊 DATASET CRYPTO:")
    print(f"  📈 Muestras: {len(X):,}")
    print(f"  🔧 Features: {len(feature_cols)}")
    print(f"  💰 Símbolos: {featured_data['symbol'].nunique()}")
    print(f"  ⏰ Timeframes: {list(featured_data['timeframe'].unique())}")
    print(f"  🎯 Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # División temporal más conservadora
    split_point = int(len(X) * 0.8)
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"\n📋 DIVISIÓN:")
    print(f"  🏃 Train: {len(X_train):,}")
    print(f"  🧪 Test: {len(X_test):,}")
    
    # Scaling robusto
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n🤖 MODELOS ULTRA-CONSERVADORES:")
    print("-" * 35)
    
    # Random Forest ULTRA conservador
    print("🌲 Random Forest ultra-conservador...")
    rf_model = RandomForestRegressor(
        n_estimators=10,  # Muy pocos árboles
        max_depth=3,      # Muy poca profundidad
        min_samples_split=100,  # Muchas muestras por split
        min_samples_leaf=50,    # Muchas muestras por hoja
        max_features=3,   # Solo 3 features máximo
        bootstrap=True,
        random_state=42
    )
    
    rf_model.fit(X_train_scaled, y_train)
    rf_pred_train = rf_model.predict(X_train_scaled)
    rf_pred_test = rf_model.predict(X_test_scaled)
    
    rf_train_r2 = r2_score(y_train, rf_pred_train)
    rf_test_r2 = r2_score(y_test, rf_pred_test)
    rf_mae = mean_absolute_error(y_test, rf_pred_test)
    
    print(f"   Train R²: {rf_train_r2:.4f}")
    print(f"   Test R²:  {rf_test_r2:.4f}")
    print(f"   MAE: {rf_mae:.4f} ({rf_mae*100:.2f}%)")
    print(f"   Overfitting: {abs(rf_train_r2 - rf_test_r2):.4f}")
    
    # Ridge ultra-conservador
    print(f"\n🏔️ Ridge ultra-conservador...")
    ridge_model = Ridge(alpha=100.0)  # Alpha muy alto
    
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred_train = ridge_model.predict(X_train_scaled)
    ridge_pred_test = ridge_model.predict(X_test_scaled)
    
    ridge_train_r2 = r2_score(y_train, ridge_pred_train)
    ridge_test_r2 = r2_score(y_test, ridge_pred_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred_test)
    
    print(f"   Train R²: {ridge_train_r2:.4f}")
    print(f"   Test R²:  {ridge_test_r2:.4f}")
    print(f"   MAE: {ridge_mae:.4f} ({ridge_mae*100:.2f}%)")
    print(f"   Overfitting: {abs(ridge_train_r2 - ridge_test_r2):.4f}")
    
    # ElasticNet como tercer modelo
    print(f"\n🎯 ElasticNet...")
    elastic_model = ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=2000)
    
    elastic_model.fit(X_train_scaled, y_train)
    elastic_pred_train = elastic_model.predict(X_train_scaled)
    elastic_pred_test = elastic_model.predict(X_test_scaled)
    
    elastic_train_r2 = r2_score(y_train, elastic_pred_train)
    elastic_test_r2 = r2_score(y_test, elastic_pred_test)
    elastic_mae = mean_absolute_error(y_test, elastic_pred_test)
    
    print(f"   Train R²: {elastic_train_r2:.4f}")
    print(f"   Test R²:  {elastic_test_r2:.4f}")
    print(f"   MAE: {elastic_mae:.4f} ({elastic_mae*100:.2f}%)")
    print(f"   Overfitting: {abs(elastic_train_r2 - elastic_test_r2):.4f}")
    
    # Ensemble simple
    ensemble_pred = (rf_pred_test + ridge_pred_test + elastic_pred_test) / 3
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    
    print(f"\n🧠 Ensemble (3 modelos):")
    print(f"   R²: {ensemble_r2:.4f}")
    print(f"   MAE: {ensemble_mae:.4f} ({ensemble_mae*100:.2f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 FEATURE IMPORTANCE:")
    for i, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Guardar modelos
    print(f"\n💾 Guardando modelos crypto...")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf_model, models_dir / "rf_crypto.joblib")
    joblib.dump(ridge_model, models_dir / "ridge_crypto.joblib")
    joblib.dump(elastic_model, models_dir / "elastic_crypto.joblib")
    joblib.dump(scaler, models_dir / "scaler_crypto.joblib")
    
    # Información del modelo
    model_info = {
        'model_type': 'crypto_optimized',
        'target_type': 'next_return_percentage',
        'feature_columns': feature_cols,
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
            'symbols': featured_data['symbol'].nunique(),
            'timeframes': list(featured_data['timeframe'].unique()),
            'target_range': [float(y.min()), float(y.max())]
        }
    }
    
    with open(models_dir / "crypto_model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Modelos crypto guardados")
    
    # Evaluación final
    print(f"\n🔍 EVALUACIÓN CRYPTO:")
    print("-" * 22)
    
    def check_crypto_model(train_r2, test_r2, mae, model_name):
        overfitting = abs(train_r2 - test_r2)
        mae_pct = mae * 100
        
        print(f"\n{model_name}:")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAE: {mae_pct:.2f}%")
        print(f"  Overfitting: {overfitting:.4f}")
        
        if overfitting < 0.05 and test_r2 > 0.01 and mae_pct < 10:
            print(f"  Status: ✅ ESTABLE")
            return True
        elif overfitting < 0.1 and test_r2 > 0.005:
            print(f"  Status: ⚠️ MODERADO")
            return True
        else:
            print(f"  Status: ❌ INESTABLE")
            return False
    
    rf_ok = check_crypto_model(rf_train_r2, rf_test_r2, rf_mae, "🌲 Random Forest")
    ridge_ok = check_crypto_model(ridge_train_r2, ridge_test_r2, ridge_mae, "🏔️ Ridge")
    elastic_ok = check_crypto_model(elastic_train_r2, elastic_test_r2, elastic_mae, "🎯 ElasticNet")
    
    print(f"\n🧠 Ensemble:")
    print(f"  R²: {ensemble_r2:.4f}")
    print(f"  MAE: {ensemble_mae*100:.2f}%")
    
    # Resultado final
    print(f"\n🎯 RESULTADO CRYPTO:")
    print("-" * 18)
    
    if ensemble_r2 > 0.02 and ensemble_mae < 0.08:
        print(f"🚀 ¡OPTIMIZACIÓN CRYPTO EXITOSA!")
        print(f"📈 Modelo estable con predicciones realistas")
        print(f"🎯 Error promedio: {ensemble_mae*100:.2f}%")
        print(f"✅ Listo para bot crypto optimizado")
        success = True
    elif ensemble_r2 > 0.01:
        print(f"⚡ OPTIMIZACIÓN MODERADA")
        print(f"📈 R² = {ensemble_r2:.4f}")
        print(f"🎯 MAE = {ensemble_mae*100:.2f}%")
        print(f"🔧 Funcional para trading conservador")
        success = True
    else:
        print(f"❌ NECESITA MÁS TRABAJO")
        success = False
    
    return success


if __name__ == "__main__":
    success = train_ultra_conservative_models()
    if success:
        print(f"\n🎉 ¡ENTRENAMIENTO CRYPTO COMPLETADO!")
        print(f"🔧 Modelos ultra-conservadores listos")
        print(f"⚡ Predicciones realistas para crypto")
    else:
        print(f"\n❌ Necesita más ajustes")
        exit(1)
