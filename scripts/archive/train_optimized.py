#!/usr/bin/env python3
"""
🔧 ENTRENAMIENTO OPTIMIZADO
- Normalización de targets (% changes en lugar de precios absolutos)
- Feature scaling mejorado con features más estables
- Validación cruzada temporal para mejor generalización
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
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_and_combine_data():
    """Carga y combina datos de forma optimizada"""
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


def create_stable_features(df):
    """Crea features más estables y robustos"""
    print("🔧 Creando features estables...")
    
    # Ordenar por timestamp para mantener secuencia temporal
    df = df.sort_index()
    
    # 🎯 TARGET OPTIMIZADO: % change en lugar de precio absoluto
    df['target_pct'] = df['close'].pct_change().shift(-1)  # % change futuro
    
    # 📊 FEATURES NORMALIZADOS (más estables)
    
    # 1. Returns normalizados
    df['returns_1'] = df['close'].pct_change()
    df['returns_3'] = df['close'].pct_change(periods=3)
    df['returns_5'] = df['close'].pct_change(periods=5)
    
    # 2. Moving averages como ratios (más estables)
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Ratios estables
    df['ma_ratio_5_20'] = df['sma_5'] / df['sma_20']
    df['ma_ratio_20_50'] = df['sma_20'] / df['sma_50']
    df['price_ma_ratio'] = df['close'] / df['sma_20']
    
    # 3. RSI normalizado (ya está entre 0-100)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = (df['rsi'] - 50) / 50  # Centrar en 0
    
    # 4. Volatilidad como feature estable
    df['volatility_short'] = df['returns_1'].rolling(window=10).std()
    df['volatility_long'] = df['returns_1'].rolling(window=30).std()
    df['volatility_ratio'] = df['volatility_short'] / (df['volatility_long'] + 1e-8)
    
    # 5. Volume features normalizados
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
    df['volume_ratio_log'] = np.log1p(df['volume_ratio'])  # Log para estabilizar
    
    # 6. Momentum features
    df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # 7. High-Low features normalizados
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # 8. Bollinger Bands position
    bb_period = 20
    bb_std = 2
    df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * df['bb_std'])
    df['bb_lower'] = df['bb_middle'] - (bb_std * df['bb_std'])
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    # Features finales - todos normalizados y estables
    feature_cols = [
        'returns_1', 'returns_3', 'returns_5',
        'ma_ratio_5_20', 'ma_ratio_20_50', 'price_ma_ratio',
        'rsi_normalized',
        'volatility_ratio',
        'volume_ratio_log',
        'momentum_3', 'momentum_5', 'momentum_10',
        'hl_ratio', 'close_position', 'bb_position'
    ]
    
    # Limpiar datos
    clean_mask = ~(df[feature_cols].isnull().any(axis=1) | df['target_pct'].isnull())
    df_clean = df[clean_mask].copy()
    
    # Filtrar outliers extremos en target (> 50% change)
    target_outlier_mask = (df_clean['target_pct'].abs() < 0.5)
    df_clean = df_clean[target_outlier_mask].copy()
    
    print(f"✅ Features estables creados: {len(feature_cols)}")
    print(f"📊 Datos limpios: {len(df_clean):,} muestras")
    print(f"🎯 Target range: {df_clean['target_pct'].min():.4f} a {df_clean['target_pct'].max():.4f}")
    
    return df_clean, feature_cols


def cross_validate_models(X, y, feature_cols):
    """Validación cruzada temporal"""
    print("\n🔄 VALIDACIÓN CRUZADA TEMPORAL...")
    
    # TimeSeriesSplit para validación temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = {'rf': [], 'ridge': []}
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"  📈 Fold {fold + 1}/5...")
        
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scaling
        scaler_cv = QuantileTransformer(output_distribution='normal')
        X_train_scaled = scaler_cv.fit_transform(X_train_cv)
        X_val_scaled = scaler_cv.transform(X_val_cv)
        
        # Random Forest
        rf_cv = RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        rf_cv.fit(X_train_scaled, y_train_cv)
        rf_pred = rf_cv.predict(X_val_scaled)
        rf_score = r2_score(y_val_cv, rf_pred)
        cv_scores['rf'].append(rf_score)
        
        # Ridge
        ridge_cv = Ridge(alpha=1.0)
        ridge_cv.fit(X_train_scaled, y_train_cv)
        ridge_pred = ridge_cv.predict(X_val_scaled)
        ridge_score = r2_score(y_val_cv, ridge_pred)
        cv_scores['ridge'].append(ridge_score)
    
    # Resultados de CV
    rf_cv_mean = np.mean(cv_scores['rf'])
    rf_cv_std = np.std(cv_scores['rf'])
    ridge_cv_mean = np.mean(cv_scores['ridge'])
    ridge_cv_std = np.std(cv_scores['ridge'])
    
    print(f"  🌲 RF CV: {rf_cv_mean:.3f} ± {rf_cv_std:.3f}")
    print(f"  🏔️ Ridge CV: {ridge_cv_mean:.3f} ± {ridge_cv_std:.3f}")
    
    return cv_scores


def train_optimized_models():
    """Entrenamiento optimizado con mejores prácticas"""
    print("🎯 ENTRENAMIENTO OPTIMIZADO")
    print("=" * 50)
    
    # Cargar datos
    combined_data = load_and_combine_data()
    if combined_data is None:
        return False
    
    # Crear features estables
    featured_data, feature_cols = create_stable_features(combined_data)
    
    if len(featured_data) < 200:
        print("❌ Datos insuficientes después de limpieza")
        return False
    
    # Preparar datos
    X = featured_data[feature_cols]
    y = featured_data['target_pct']  # Usando % change como target
    
    print(f"\n📊 DATASET OPTIMIZADO:")
    print(f"  📈 Muestras: {len(X):,}")
    print(f"  🔧 Features: {len(feature_cols)}")
    print(f"  💰 Símbolos: {featured_data['symbol'].nunique()}")
    print(f"  ⏰ Timeframes: {list(featured_data['timeframe'].unique())}")
    print(f"  🎯 Target (% change): {y.describe()}")
    
    # Validación cruzada temporal
    cv_scores = cross_validate_models(X, y, feature_cols)
    
    # División temporal (80/20)
    split_point = int(len(X) * 0.8)
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    
    print(f"\n📋 DIVISIÓN TEMPORAL:")
    print(f"  🏃 Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  🧪 Test: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
    
    # 🔧 SCALING OPTIMIZADO: QuantileTransformer
    print(f"\n🔧 SCALING AVANZADO:")
    scaler = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  ✅ QuantileTransformer aplicado (distribución normal)")
    
    print(f"\n🤖 MODELOS OPTIMIZADOS:")
    print("-" * 30)
    
    # Random Forest optimizado
    print("🌲 Random Forest optimizado...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
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
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred_test))
    
    print(f"   Train R²: {rf_train_r2:.4f}")
    print(f"   Test R²:  {rf_test_r2:.4f}")
    print(f"   MAE: {rf_mae:.6f} ({rf_mae*100:.3f}%)")
    print(f"   RMSE: {rf_rmse:.6f} ({rf_rmse*100:.3f}%)")
    print(f"   Overfitting: {abs(rf_train_r2 - rf_test_r2):.4f}")
    
    # Ridge optimizado
    print(f"\n🏔️ Ridge optimizado...")
    ridge_model = Ridge(alpha=0.1, max_iter=2000)
    
    ridge_model.fit(X_train_scaled, y_train)
    ridge_pred_train = ridge_model.predict(X_train_scaled)
    ridge_pred_test = ridge_model.predict(X_test_scaled)
    
    ridge_train_r2 = r2_score(y_train, ridge_pred_train)
    ridge_test_r2 = r2_score(y_test, ridge_pred_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred_test)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred_test))
    
    print(f"   Train R²: {ridge_train_r2:.4f}")
    print(f"   Test R²:  {ridge_test_r2:.4f}")
    print(f"   MAE: {ridge_mae:.6f} ({ridge_mae*100:.3f}%)")
    print(f"   RMSE: {ridge_rmse:.6f} ({ridge_rmse*100:.3f}%)")
    print(f"   Overfitting: {abs(ridge_train_r2 - ridge_test_r2):.4f}")
    
    # Ensemble optimizado
    ensemble_pred = (rf_pred_test + ridge_pred_test) / 2
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    print(f"\n🧠 Ensemble optimizado:")
    print(f"   R²: {ensemble_r2:.4f}")
    print(f"   MAE: {ensemble_mae:.6f} ({ensemble_mae*100:.3f}%)")
    print(f"   RMSE: {ensemble_rmse:.6f} ({ensemble_rmse*100:.3f}%)")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 TOP 5 FEATURES:")
    for i, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Guardar modelos optimizados
    print(f"\n💾 Guardando modelos optimizados...")
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(rf_model, models_dir / "rf_optimized.joblib")
    joblib.dump(ridge_model, models_dir / "ridge_optimized.joblib")
    joblib.dump(scaler, models_dir / "scaler_optimized.joblib")
    
    # Información del modelo optimizado
    model_info = {
        'model_type': 'optimized_percentage_prediction',
        'target_type': 'percentage_change',
        'feature_columns': feature_cols,
        'scaling_method': 'QuantileTransformer',
        'cross_validation': {
            'rf_scores': cv_scores['rf'],
            'ridge_scores': cv_scores['ridge'],
            'rf_mean': np.mean(cv_scores['rf']),
            'ridge_mean': np.mean(cv_scores['ridge'])
        },
        'models': {
            'random_forest': {
                'train_r2': rf_train_r2,
                'test_r2': rf_test_r2,
                'mae': rf_mae,
                'rmse': rf_rmse,
                'overfitting': abs(rf_train_r2 - rf_test_r2)
            },
            'ridge': {
                'train_r2': ridge_train_r2,
                'test_r2': ridge_test_r2,
                'mae': ridge_mae,
                'rmse': ridge_rmse,
                'overfitting': abs(ridge_train_r2 - ridge_test_r2)
            },
            'ensemble': {
                'r2': ensemble_r2,
                'mae': ensemble_mae,
                'rmse': ensemble_rmse
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
    
    with open(models_dir / "optimized_model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Modelos optimizados guardados")
    
    # Evaluación final
    print(f"\n🔍 EVALUACIÓN OPTIMIZADA:")
    print("-" * 25)
    
    def evaluate_optimized_model(test_r2, mae_pct, model_name):
        print(f"\n{model_name}:")
        
        if test_r2 > 0.1:
            r2_status = "🎉 EXCELENTE"
        elif test_r2 > 0.05:
            r2_status = "✅ BUENO"
        elif test_r2 > 0.01:
            r2_status = "⚠️ MODERADO"
        else:
            r2_status = "❌ BAJO"
        
        if mae_pct < 1.0:
            mae_status = "🎯 PRECISIÓN ALTA"
        elif mae_pct < 2.0:
            mae_status = "✅ PRECISIÓN BUENA"
        elif mae_pct < 5.0:
            mae_status = "⚠️ PRECISIÓN MODERADA"
        else:
            mae_status = "❌ PRECISIÓN BAJA"
        
        print(f"  R²: {test_r2:.4f} {r2_status}")
        print(f"  MAE: {mae_pct:.3f}% {mae_status}")
        
        return test_r2 > 0.01 and mae_pct < 5.0
    
    rf_ok = evaluate_optimized_model(rf_test_r2, rf_mae*100, "🌲 Random Forest")
    ridge_ok = evaluate_optimized_model(ridge_test_r2, ridge_mae*100, "🏔️ Ridge")
    ensemble_ok = evaluate_optimized_model(ensemble_r2, ensemble_mae*100, "🧠 Ensemble")
    
    # Resultado final
    print(f"\n🎯 RESULTADO OPTIMIZADO:")
    print("-" * 20)
    
    best_r2 = max(rf_test_r2, ridge_test_r2, ensemble_r2)
    best_mae = min(rf_mae, ridge_mae, ensemble_mae) * 100
    
    if best_r2 > 0.1 and best_mae < 2.0:
        print(f"🚀 ¡OPTIMIZACIÓN EXITOSA!")
        print(f"📈 R² = {best_r2:.4f} (explica {best_r2*100:.1f}% varianza)")
        print(f"🎯 MAE = {best_mae:.2f}% (error promedio)")
        print(f"✨ Predicciones estables y realistas")
        success = True
    elif best_r2 > 0.05:
        print(f"✅ OPTIMIZACIÓN BUENA")
        print(f"📈 R² = {best_r2:.4f}")
        print(f"🎯 MAE = {best_mae:.2f}%")
        print(f"🔧 Listo para producción")
        success = True
    else:
        print(f"⚠️ OPTIMIZACIÓN PARCIAL")
        print(f"📈 R² = {best_r2:.4f}")
        print(f"🎯 MAE = {best_mae:.2f}%")
        print(f"🔄 Considera más datos o features")
        success = False
    
    return success


if __name__ == "__main__":
    success = train_optimized_models()
    if success:
        print(f"\n🎉 ¡ENTRENAMIENTO OPTIMIZADO EXITOSO!")
        print(f"🔧 Predicciones en % change (más realistas)")
        print(f"⚡ Listo para bot optimizado")
    else:
        print(f"\n❌ Necesita más optimización")
        exit(1)
