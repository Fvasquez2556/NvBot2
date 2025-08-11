#!/usr/bin/env python3
"""
Script para probar el sistema de notificaciones
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.notification_system import NotificationSystem, AlertMessage, PerformanceReport
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def test_all_notifications():
    """
    Prueba todos los tipos de notificaciones
    """
    print("🧪 Testing Notification System")
    print("=" * 50)
    
    # Initialize notification system
    notification_system = NotificationSystem()
    
    # Test 1: Basic notification test
    print("\n1️⃣ Testing notification channels...")
    test_results = await notification_system.test_notifications()
    
    for channel, success in test_results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {channel.title()}: {status}")
    
    # Test 2: Momentum signal alert
    print("\n2️⃣ Testing momentum signal alert...")
    momentum_data = {
        'symbol': 'BTCUSDT',
        'price': 45000.0,
        'confidence': 0.87,
        'signal_strength': 'very_strong',
        'dominant_timeframe': '5m',
        'expected_return': 0.085,
        'risk_level': 'medium'
    }
    
    await notification_system.send_momentum_alert(momentum_data)
    print("   ✅ Momentum alert sent")
    
    # Test 3: Trade execution alert
    print("\n3️⃣ Testing trade execution alert...")
    trade_data = {
        'action': 'buy',
        'symbol': 'ETHUSDT',
        'price': 3000.0,
        'quantity': 0.5,
        'position_size': 0.04,
        'stop_loss': 2850.0,
        'take_profit': 3300.0
    }
    
    await notification_system.send_trade_alert(trade_data)
    print("   ✅ Trade alert sent")
    
    # Test 4: Risk warning
    print("\n4️⃣ Testing risk warning...")
    risk_data = {
        'type': 'High Portfolio Risk',
        'severity': 'high',
        'description': 'Portfolio exposure exceeds 80% of available capital',
        'exposure': 0.85,
        'recommendation': 'Reduce position sizes or close some positions'
    }
    
    await notification_system.send_risk_warning(risk_data)
    print("   ✅ Risk warning sent")
    
    # Test 5: System error alert
    print("\n5️⃣ Testing system error alert...")
    error_data = {
        'component': 'Data Feed',
        'error': 'Connection timeout to Binance API',
        'timestamp': datetime.now(),
        'impact': 'Trading temporarily suspended',
        'suggested_fix': 'Check network connection and API credentials'
    }
    
    await notification_system.send_system_error(error_data)
    print("   ✅ System error alert sent")
    
    # Test 6: Performance report
    print("\n6️⃣ Testing performance report...")
    report = await notification_system.generate_daily_report()
    await notification_system.send_performance_report(report)
    print("   ✅ Performance report sent")
    
    # Test 7: Dashboard data
    print("\n7️⃣ Testing dashboard data...")
    dashboard_data = await notification_system.get_dashboard_data()
    print(f"   Dashboard last update: {dashboard_data['last_update']}")
    print(f"   Total alerts in cache: {len(dashboard_data['alerts'])}")
    print(f"   System status: {list(dashboard_data['system_status'].keys())}")
    print("   ✅ Dashboard data retrieved")
    
    return True

async def test_rate_limiting():
    """
    Prueba el rate limiting del sistema
    """
    print("\n🚦 Testing Rate Limiting")
    print("=" * 30)
    
    notification_system = NotificationSystem()
    
    # Send multiple alerts rapidly
    alerts_sent = 0
    for i in range(5):
        alert = AlertMessage(
            type='test_alert',
            priority='low',
            title=f'Test Alert #{i+1}',
            message=f'This is test alert number {i+1}'
        )
        
        try:
            await notification_system._dispatch_alert(alert)
            alerts_sent += 1
            await asyncio.sleep(0.1)  # Small delay
        except Exception as e:
            print(f"   Alert {i+1} failed: {e}")
    
    print(f"   ✅ Successfully sent {alerts_sent}/5 test alerts")
    return True

async def test_custom_alert():
    """
    Permite al usuario enviar una alerta personalizada
    """
    print("\n✨ Custom Alert Test")
    print("=" * 25)
    
    # Get user input
    title = input("Enter alert title (or press Enter for default): ").strip()
    if not title:
        title = "Custom Test Alert"
    
    message = input("Enter alert message (or press Enter for default): ").strip()
    if not message:
        message = "This is a custom test alert from the notification system test script."
    
    priority = input("Enter priority (low/medium/high/critical) [medium]: ").strip().lower()
    if priority not in ['low', 'medium', 'high', 'critical']:
        priority = 'medium'
    
    # Send custom alert
    notification_system = NotificationSystem()
    alert = AlertMessage(
        type='custom_test',
        priority=priority,
        title=title,
        message=message,
        symbol='TEST',
        additional_data={'test_mode': True}
    )
    
    await notification_system._dispatch_alert(alert)
    print(f"   ✅ Custom alert sent with priority: {priority}")
    
    return True

def print_system_info():
    """
    Muestra información del sistema de notificaciones
    """
    print("\n📋 System Information")
    print("=" * 25)
    
    # Check environment variables
    env_vars = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID',
        'EMAIL_ADDRESS',
        'EMAIL_PASSWORD',
        'WEBHOOK_URLS'
    ]
    
    print("Environment Variables:")
    for var in env_vars:
        value = os.getenv(var)
        status = "✅ SET" if value else "❌ NOT SET"
        # Mask sensitive data
        if value and 'TOKEN' in var or 'PASSWORD' in var:
            display_value = f"{value[:8]}***"
        elif value and len(value) > 20:
            display_value = f"{value[:20]}..."
        else:
            display_value = value or "None"
        
        print(f"   {var}: {status} ({display_value})")

async def main():
    """
    Main test function
    """
    print("🧪 Momentum Bot Notification System Tester")
    print("=" * 60)
    
    # Show system info
    print_system_info()
    
    # Interactive menu
    while True:
        print("\n📋 Test Menu:")
        print("1. Test all notification types")
        print("2. Test rate limiting")
        print("3. Send custom alert")
        print("4. Show system information")
        print("0. Exit")
        
        choice = input("\nSelect an option (0-4): ").strip()
        
        try:
            if choice == '1':
                await test_all_notifications()
            elif choice == '2':
                await test_rate_limiting()
            elif choice == '3':
                await test_custom_alert()
            elif choice == '4':
                print_system_info()
            elif choice == '0':
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Test interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            logger.error(f"Test error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
