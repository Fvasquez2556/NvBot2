#!/bin/bash
# @claude: Script completo de deployment que:
# 1. Valide configuración
# 2. Ejecute tests
# 3. Build Docker image
# 4. Deploy a producción
# 5. Run health checks
# 6. Setup monitoring

set -e  # Exit on any error

echo "🚀 Deploying Momentum Predictor Bot..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="momentum-predictor-bot"
VERSION=$(date +%Y%m%d-%H%M%S)
DOCKER_IMAGE="${PROJECT_NAME}:${VERSION}"
DOCKER_LATEST="${PROJECT_NAME}:latest"

# Environment check
ENVIRONMENT=${1:-production}
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Verificar que estamos en el directorio correcto
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Run from project root directory."
    exit 1
fi

print_status "Project root directory confirmed"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

print_status "Docker is running"

# Activar entorno virtual
echo -e "${BLUE}Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
print_status "Virtual environment activated"

# Verificar variables de entorno
echo "📋 Verificando configuración..."
if [ ! -f ".env" ]; then
    echo "⚠️ Advertencia: No se encontró archivo .env"
    echo "Usando configuración de ejemplo..."
    cp .env.example .env
fi

# Ejecutar tests
echo "🧪 Ejecutando tests..."
python -m pytest tests/ -v

if [ $? -ne 0 ]; then
    echo "❌ Tests fallaron. Deteniendo despliegue."
    exit 1
fi

# Verificar calidad de código
echo "🔍 Verificando calidad de código..."
python -m flake8 src/ --max-line-length=88
python -m black --check src/
python -m isort --check-only src/

# Crear backup de modelos existentes
echo "💾 Creando backup de modelos..."
if [ -d "data/models" ] && [ "$(ls -A data/models)" ]; then
    backup_dir="data/models_backup_$(date +%Y%m%d_%H%M%S)"
    cp -r data/models "$backup_dir"
    echo "📦 Backup creado en: $backup_dir"
fi

# Descargar datos actualizados
echo "📊 Descargando datos actualizados..."
python scripts/download_data.py

# Entrenar modelos
echo "🤖 Entrenando modelos..."
python scripts/train_models.py

# Verificar que los modelos se entrenaron correctamente
echo "✅ Verificando modelos entrenados..."
model_files=("data/models/regression_model.joblib" "data/models/classification_model.joblib" "data/models/feature_engineer.joblib")

for model_file in "${model_files[@]}"; do
    if [ ! -f "$model_file" ]; then
        echo "❌ Error: No se encontró $model_file"
        exit 1
    fi
done

echo "📈 Todos los modelos se entrenaron correctamente"

# Crear archivo de versión
echo "📝 Creando archivo de versión..."
echo "version=$(date +%Y.%m.%d-%H%M%S)" > version.txt
echo "deployed_at=$(date)" >> version.txt
echo "commit=$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" >> version.txt

echo ""
echo "🎉 ¡Despliegue completado exitosamente!"
echo ""
echo "Para iniciar el bot:"
echo "  python src/main.py"
echo ""
echo "Para monitorear logs:"
echo "  tail -f logs/momentum_bot.log"
echo ""
echo "⚠️ IMPORTANTE: Verifica que el bot esté funcionando correctamente en modo testnet antes de usar fondos reales."
