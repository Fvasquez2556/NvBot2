#!/usr/bin/env python3
"""
Script para validar configuración del sistema
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def print_status(message: str, status: bool = True):
    """Imprime status con color"""
    emoji = "✅" if status else "❌"
    print(f"   {emoji} {message}")

def print_warning(message: str):
    """Imprime warning"""
    print(f"   ⚠️  {message}")

def validate_environment_variables() -> bool:
    """
    Valida variables de entorno requeridas
    """
    print("\n📋 Validating Environment Variables")
    print("-" * 40)
    
    required_vars = {
        'BINANCE_API_KEY': 'Binance API key for trading',
        'BINANCE_SECRET_KEY': 'Binance secret key for trading',
        'TELEGRAM_BOT_TOKEN': 'Telegram bot token for notifications',
        'TELEGRAM_CHAT_ID': 'Telegram chat ID for notifications'
    }
    
    optional_vars = {
        'COINBASE_API_KEY': 'Coinbase API key (optional)',
        'EMAIL_ADDRESS': 'Email address for notifications',
        'EMAIL_PASSWORD': 'Email password for notifications',
        'WEBHOOK_URLS': 'Webhook URLs for notifications',
        'DATABASE_URL': 'Database connection URL',
        'REDIS_URL': 'Redis connection URL'
    }
    
    all_valid = True
    
    # Check required variables
    print("Required Variables:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            print_status(f"{var}: {description}")
        else:
            print_status(f"{var}: {description} - NOT SET", False)
            all_valid = False
    
    # Check optional variables
    print("\nOptional Variables:")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            print_status(f"{var}: {description}")
        else:
            print_warning(f"{var}: {description} - not set")
    
    return all_valid

def validate_config_files() -> bool:
    """
    Valida archivos de configuración
    """
    print("\n📄 Validating Configuration Files")
    print("-" * 40)
    
    config_files = {
        'config/config.yaml': 'Main configuration file',
        'config/model_params.yaml': 'ML model parameters',
        'config/trading_pairs.yaml': 'Trading pairs configuration',
        '.env.example': 'Environment variables example',
        'requirements.txt': 'Python dependencies',
        'requirements-dev.txt': 'Development dependencies'
    }
    
    all_valid = True
    
    for file_path, description in config_files.items():
        full_path = Path(file_path)
        if full_path.exists():
            try:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    with open(full_path, 'r') as f:
                        yaml.safe_load(f)
                elif file_path.endswith('.json'):
                    with open(full_path, 'r') as f:
                        json.load(f)
                print_status(f"{file_path}: {description}")
            except Exception as e:
                print_status(f"{file_path}: Invalid format - {e}", False)
                all_valid = False
        else:
            print_status(f"{file_path}: {description} - NOT FOUND", False)
            all_valid = False
    
    return all_valid

def validate_directory_structure() -> bool:
    """
    Valida estructura de directorios
    """
    print("\n📁 Validating Directory Structure")
    print("-" * 40)
    
    required_dirs = [
        'src',
        'src/strategies',
        'src/ml_models',
        'src/data_sources',
        'src/utils',
        'tests',
        'config',
        'data',
        'data/raw',
        'data/processed',
        'data/models',
        'logs',
        'scripts',
        'notebooks'
    ]
    
    all_valid = True
    
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists() and full_path.is_dir():
            print_status(f"{dir_path}/")
        else:
            print_status(f"{dir_path}/ - NOT FOUND", False)
            all_valid = False
    
    return all_valid

def validate_python_files() -> bool:
    """
    Valida archivos Python principales
    """
    print("\n🐍 Validating Python Files")
    print("-" * 40)
    
    required_files = [
        'src/__init__.py',
        'src/main.py',
        'src/strategies/base_strategy.py',
        'src/strategies/momentum_detector.py',
        'src/strategies/momentum_predictor_strategy.py',
        'src/strategies/multi_timeframe_analyzer.py',
        'src/utils/config_manager.py',
        'src/utils/logger.py',
        'src/utils/notification_system.py'
    ]
    
    all_valid = True
    
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            try:
                # Basic syntax check
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    compile(content, file_path, 'exec')
                print_status(f"{file_path}")
            except SyntaxError as e:
                print_status(f"{file_path} - SYNTAX ERROR: {e}", False)
                all_valid = False
            except Exception as e:
                print_status(f"{file_path} - ERROR: {e}", False)
                all_valid = False
        else:
            print_status(f"{file_path} - NOT FOUND", False)
            all_valid = False
    
    return all_valid

def validate_dependencies() -> bool:
    """
    Valida dependencias Python
    """
    print("\n📦 Validating Python Dependencies")
    print("-" * 40)
    
    critical_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'ccxt',
        'python-binance',
        'aiohttp',
        'pyyaml',
        'python-dotenv'
    ]
    
    all_valid = True
    
    for package in critical_packages:
        try:
            __import__(package.replace('-', '_'))
            print_status(f"{package}")
        except ImportError:
            print_status(f"{package} - NOT INSTALLED", False)
            all_valid = False
    
    return all_valid

def validate_docker_setup() -> bool:
    """
    Valida configuración de Docker
    """
    print("\n🐳 Validating Docker Setup")
    print("-" * 40)
    
    docker_files = [
        'Dockerfile',
        'docker/docker-compose.yml',
        '.dockerignore'
    ]
    
    all_valid = True
    
    for file_path in docker_files:
        full_path = Path(file_path)
        if full_path.exists():
            print_status(f"{file_path}")
        else:
            print_status(f"{file_path} - NOT FOUND", False)
            all_valid = False
    
    # Check if Docker is available
    try:
        import subprocess
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print_status("Docker is installed and accessible")
        else:
            print_warning("Docker is not accessible")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_warning("Docker is not installed or not in PATH")
    
    return all_valid

def validate_vscode_setup() -> bool:
    """
    Valida configuración de VS Code
    """
    print("\n💻 Validating VS Code Setup")
    print("-" * 40)
    
    vscode_files = [
        '.vscode/settings.json',
        '.vscode/launch.json',
        '.vscode/tasks.json'
    ]
    
    all_valid = True
    
    for file_path in vscode_files:
        full_path = Path(file_path)
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    json.load(f)
                print_status(f"{file_path}")
            except json.JSONDecodeError as e:
                print_status(f"{file_path} - INVALID JSON: {e}", False)
                all_valid = False
        else:
            print_status(f"{file_path} - NOT FOUND", False)
            all_valid = False
    
    return all_valid

def generate_summary_report(results: Dict[str, bool]) -> None:
    """
    Genera reporte resumen
    """
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY REPORT")
    print("="*60)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    failed_checks = total_checks - passed_checks
    
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks} ✅")
    print(f"Failed: {failed_checks} ❌")
    print(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")
    
    print("\nDetailed Results:")
    for check_name, status in results.items():
        emoji = "✅" if status else "❌"
        print(f"   {emoji} {check_name.replace('_', ' ').title()}")
    
    if failed_checks == 0:
        print(f"\n🎉 All validations passed! System is ready for deployment.")
    else:
        print(f"\n⚠️  {failed_checks} validation(s) failed. Please fix issues before deployment.")
        print("Check the detailed output above for specific issues.")

def main():
    """
    Función principal de validación
    """
    print("🔍 Momentum Bot Configuration Validator")
    print("="*60)
    print("This script validates your bot configuration and setup")
    print("="*60)
    
    # Run all validations
    validation_results = {
        'environment_variables': validate_environment_variables(),
        'config_files': validate_config_files(),
        'directory_structure': validate_directory_structure(),
        'python_files': validate_python_files(),
        'dependencies': validate_dependencies(),
        'docker_setup': validate_docker_setup(),
        'vscode_setup': validate_vscode_setup()
    }
    
    # Generate summary
    generate_summary_report(validation_results)
    
    # Exit with appropriate code
    if all(validation_results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main()
