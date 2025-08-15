#!/usr/bin/env python3
"""
Script de entrenamiento optimizado para datos limitados
Aplicando técnicas anti-overfitting de la guía
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge  # Más estable que LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score


def calculate_simple_features(df):
    """Calcula features básicos para evitar overfitting"""
    print("📊 Calculando features básicos (anti-overfitting)...")
    
    # Solo features básicos y robustos
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    
    # Moving averages simples
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # RSI simplificado
    def simple_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)  # Evitar división por cero
        return 100 - (100 / (1 + rs))
    
    df['rsi'] = simple_rsi(df['close'])
    
    # Volatilidad simple
    df['volatility'] = df['close'].rolling(window=10).std()
    
    # Ratio simple
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Target simple
    df['target'] = df['close'].shift(-1)
    
    print("✅ Features básicos calculados (8 features)")
    return df


def train_conservative_models():
    """Entrenamiento conservador anti-overfitting"""
    print("🎯 NVBOT2 ENTRENAMIENTO CONSERVADOR - Anti-Overfitting")
    print("=" * 55)
    print("🚀 Iniciando entrenamiento optimizado...")
    
    try:
        # Cargar datos
        print("📂 Cargando datos históricos...")
        data_files = list(Path("data/raw").glob("*_1h.csv"))
        
        if not data_files:
            print("❌ No se encontraron archivos de datos en data/raw/")
            return False
        
        # Combinar datos
        all_data = []
        for file in data_files:
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            symbol = file.stem.replace("_1h", "")
            df["symbol"] = symbol
            all_data.append(df)
            print(f"✅ {symbol}: {len(df)} registros")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"📊 Total: {len(combined_data)} registros")
        
        # Feature engineering conservador
        print("\n🔧 FASE: Feature Engineering Conservador")
        print("-" * 40)
        featured_data = calculate_simple_features(combined_data)
        
        # Preparar datos (solo features básicos)
        feature_cols = ['price_change', 'price_change_5', 'sma_5', 'sma_20', 
                       'rsi', 'volatility', 'hl_ratio', 'volume']
        
        # Eliminar NaN
        mask = ~(featured_data[feature_cols].isnull().any(axis=1) | 
                featured_data['target'].isnull())
        
        X = featured_data[feature_cols][mask]
        y = featured_data['target'][mask]
        
        print(f"📊 Datos limpios: {len(X)} muestras, {len(feature_cols)} features")
        
        if len(X) < 50:
            print("❌ Datos insuficientes para entrenamiento")
            return False
        
        # División temporal (importante para series temporales)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📈 Entrenamiento: {len(X_train)} | Testing: {len(X_test)}")
        
        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Crear directorio
        models_dir = Path("data/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🤖 FASE: Entrenamiento Conservador")
        print("-" * 35)
        
        # 1. Random Forest MUY conservador
        print("🎯 Random Forest (Ultra Conservador)...")
        rf_model = RandomForestRegressor(
            n_estimators=20,        # MUY POCOS árboles
            max_depth=3,           # MUY POCA profundidad
            min_samples_split=50,  # MUCHAS muestras para dividir
            min_samples_leaf=20,   # MUCHAS muestras en hojas
            max_features=0.5,      # Solo 50% de features
            random_state=42
        )
        
        rf_model.fit(X_train_scaled, y_train)
        rf_pred_train = rf_model.predict(X_train_scaled)
        rf_pred_test = rf_model.predict(X_test_scaled)
        
        rf_train_score = r2_score(y_train, rf_pred_train)
        rf_test_score = r2_score(y_test, rf_pred_test)
        rf_mae = mean_absolute_error(y_test, rf_pred_test)
        
        print(f"   ✅ Train: {rf_train_score:.3f} ({rf_train_score*100:.1f}%)")
        print(f"   ✅ Test:  {rf_test_score:.3f} ({rf_test_score*100:.1f}%)")
        print(f"   📈 MAE: {rf_mae:.2f}")
        print(f"   📊 Diferencia: {abs(rf_train_score - rf_test_score):.3f}")
        
        # 2. Ridge Regression (muy estable)
        print("\n🎯 Ridge Regression (Regularizado)...")
        ridge_model = Ridge(alpha=10.0)  # Alta regularización
        
        ridge_model.fit(X_train_scaled, y_train)
        ridge_pred_train = ridge_model.predict(X_train_scaled)
        ridge_pred_test = ridge_model.predict(X_test_scaled)
        
        ridge_train_score = r2_score(y_train, ridge_pred_train)
        ridge_test_score = r2_score(y_test, ridge_pred_test)
        ridge_mae = mean_absolute_error(y_test, ridge_pred_test)
        
        print(f"   ✅ Train: {ridge_train_score:.3f} ({ridge_train_score*100:.1f}%)")
        print(f"   ✅ Test:  {ridge_test_score:.3f} ({ridge_test_score*100:.1f}%)")
        print(f"   📈 MAE: {ridge_mae:.2f}")
        print(f"   📊 Diferencia: {abs(ridge_train_score - ridge_test_score):.3f}")
        
        # Cross-validation
        print("\n🔄 Cross-Validation (Series Temporales)...")
        tscv = TimeSeriesSplit(n_splits=3)
        
        rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=tscv, scoring='r2')
        ridge_cv_scores = cross_val_score(ridge_model, X_train_scaled, y_train, cv=tscv, scoring='r2')
        
        print(f"   🔄 RF CV: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")
        print(f"   🔄 Ridge CV: {ridge_cv_scores.mean():.3f} (+/- {ridge_cv_scores.std() * 2:.3f})")
        
        # Ensemble simple
        if rf_test_score > 0 and ridge_test_score > 0:
            ensemble_pred = (rf_pred_test * 0.6 + ridge_pred_test * 0.4)
            ensemble_score = r2_score(y_test, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            print(f"\n🧠 Ensemble Simple:")
            print(f"   ✅ Score: {ensemble_score:.3f} ({ensemble_score*100:.1f}%)")
            print(f"   📈 MAE: {ensemble_mae:.2f}")
        else:
            # Si hay scores negativos, usar solo el mejor
            if rf_test_score > ridge_test_score:
                ensemble_pred = rf_pred_test
                ensemble_score = rf_test_score
                ensemble_mae = rf_mae
                print(f"\n🧠 Usando solo Random Forest (mejor performance)")
            else:
                ensemble_pred = ridge_pred_test
                ensemble_score = ridge_test_score
                ensemble_mae = ridge_mae
                print(f"\n🧠 Usando solo Ridge (mejor performance)")
        
        # Guardar modelos
        print("\n💾 Guardando modelos...")
        joblib.dump(rf_model, models_dir / "rf_conservative.joblib")
        joblib.dump(ridge_model, models_dir / "ridge_model.joblib")
        joblib.dump(scaler, models_dir / "scaler_conservative.joblib")
        
        # Métricas finales
        results = {
            'random_forest': {
                'train_score': rf_train_score,
                'test_score': rf_test_score,
                'mae': rf_mae,
                'cv_mean': rf_cv_scores.mean(),
                'cv_std': rf_cv_scores.std()
            },
            'ridge': {
                'train_score': ridge_train_score,
                'test_score': ridge_test_score,
                'mae': ridge_mae,
                'cv_mean': ridge_cv_scores.mean(),
                'cv_std': ridge_cv_scores.std()
            },
            'ensemble': {
                'score': ensemble_score,
                'mae': ensemble_mae
            },
            'features': feature_cols,
            'training_date': datetime.now().isoformat()
        }
        
        with open(models_dir / "conservative_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Modelos conservadores guardados")
        
        # Análisis final
        print("\n🔍 ANÁLISIS ANTI-OVERFITTING:")
        print("-" * 35)
        
        def check_overfitting(train_score, test_score, model_name):
            diff = abs(train_score - test_score)
            if diff < 0.05:
                status = "✅ EXCELENTE"
            elif diff < 0.15:
                status = "⚠️  ACEPTABLE"
            else:
                status = "🚨 OVERFITTING"
            
            print(f"{model_name}: {status}")
            print(f"  Train: {train_score:.3f} | Test: {test_score:.3f} | Diff: {diff:.3f}")
            return diff < 0.15
        
        rf_ok = check_overfitting(rf_train_score, rf_test_score, "RANDOM FOREST")
        ridge_ok = check_overfitting(ridge_train_score, ridge_test_score, "RIDGE")
        
        if rf_ok or ridge_ok:
            print("\n🎉 ¡ENTRENAMIENTO EXITOSO! Al menos un modelo está saludable.")
        else:
            print("\n⚠️  Aún hay overfitting. Necesitamos más datos o más regularización.")
        
        print("\n📋 RECOMENDACIONES SEGÚN TU GUÍA:")
        print("-" * 35)
        if not rf_ok and not ridge_ok:
            print("1. 📊 Descargar más datos históricos (2-3 años)")
            print("2. 🔧 Usar menos features (máximo 5-6)")
            print("3. 📈 Considerar usar solo Ridge o modelos más simples")
        else:
            print("1. ✅ Los modelos están mejorando")
            print("2. 📈 Continuar con backtesting")
            print("3. 🚀 Probar con paper trading")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = train_conservative_models()
    if not success:
        exit(1)
