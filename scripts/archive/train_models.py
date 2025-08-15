#!/usr/bin/env python3
"""
Script para entrenar modelos de machine learning
"""

import sys
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config_manager import ConfigManager
from utils.logger import setup_logger
from ml_models.feature_engineering import FeatureEngineer
from ml_models.regression_model import RegressionModel
from ml_models.classification_model import ClassificationModel


def main():
    """Función principal para entrenar modelos"""
    logger = setup_logger("train_models", "train_models.log")
    logger.info("Iniciando entrenamiento de modelos...")
    
    try:
        # Cargar configuración
        config = ConfigManager()
        
        # Cargar datos
        logger.info("Cargando datos de entrenamiento...")
        data_files = list(Path("data/raw").glob("*_1h.csv"))
        
        if not data_files:
            logger.error("No se encontraron archivos de datos en data/raw/")
            return
        
        # Combinar todos los datos
        all_data = []
        for file in data_files:
            df = pd.read_csv(file)
            symbol = file.stem.replace("_1h", "").replace("_", "/")
            df["symbol"] = symbol
            all_data.append(df)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Datos cargados: {len(combined_data)} registros")
        
        # Feature engineering
        logger.info("Realizando feature engineering...")
        feature_engineer = FeatureEngineer(config)
        features_df = feature_engineer.create_features(combined_data)
        
        # Preparar datos para modelos
        X, y_reg, y_class = feature_engineer.prepare_model_data(features_df)
        
        # Entrenar modelo de regresión
        logger.info("Entrenando modelo de regresión...")
        regression_model = RegressionModel(config)
        regression_model.train(X, y_reg)
        
        # Guardar modelo de regresión
        model_path = Path("data/models") / "regression_model.joblib"
        joblib.dump(regression_model, model_path)
        logger.info(f"Modelo de regresión guardado: {model_path}")
        
        # Entrenar modelo de clasificación
        logger.info("Entrenando modelo de clasificación...")
        classification_model = ClassificationModel(config)
        classification_model.train(X, y_class)
        
        # Guardar modelo de clasificación
        model_path = Path("data/models") / "classification_model.joblib"
        joblib.dump(classification_model, model_path)
        logger.info(f"Modelo de clasificación guardado: {model_path}")
        
        # Guardar feature engineer
        feature_path = Path("data/models") / "feature_engineer.joblib"
        joblib.dump(feature_engineer, feature_path)
        logger.info(f"Feature engineer guardado: {feature_path}")
        
        logger.info("Entrenamiento de modelos completado")
        
    except Exception as e:
        logger.error(f"Error en entrenamiento de modelos: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
