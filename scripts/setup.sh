#!/bin/bash

# Script de configuración para el proyecto Momentum Predictor Bot

echo "🚀 Configurando Momentum Predictor Bot..."

# Verificar Python
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python encontrado: $python_version"
else
    echo "❌ Python 3 no encontrado. Por favor instala Python 3.8 o superior."
    exit 1
fi

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv venv

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Actualizar pip
echo "🔄 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias principales
echo "📚 Instalando dependencias principales..."
pip install -r requirements.txt

# Instalar dependencias de desarrollo
echo "🛠️ Instalando dependencias de desarrollo..."
pip install -r requirements-dev.txt

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p data/{raw,processed,models}
mkdir -p logs
mkdir -p notebooks/{research,backtesting,analysis}

# Copiar archivo de configuración de entorno
echo "⚙️ Configurando variables de entorno..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Archivo .env creado. Por favor edítalo con tus credenciales."
fi

# Configurar pre-commit hooks
echo "🔗 Configurando pre-commit hooks..."
pre-commit install

echo ""
echo "🎉 ¡Configuración completada!"
echo ""
echo "Próximos pasos:"
echo "1. Edita el archivo .env con tus credenciales de API"
echo "2. Revisa y ajusta config/config.yaml según tus necesidades"
echo "3. Ejecuta 'python scripts/download_data.py' para descargar datos históricos"
echo "4. Ejecuta 'python scripts/train_models.py' para entrenar los modelos"
echo "5. Ejecuta 'python src/main.py' para iniciar el bot"
echo ""
echo "Para activar el entorno virtual en el futuro:"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "  source venv/Scripts/activate"
else
    echo "  source venv/bin/activate"
fi
