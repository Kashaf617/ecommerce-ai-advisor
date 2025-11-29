"""
Configuration settings for the E-Commerce Business Automation Platform
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent
BASE_DIR = PROJECT_ROOT  # Alias for compatibility

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
WAREHOUSE_DIR = DATA_DIR / 'warehouse'

# AI/ML models directory
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Create data directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, WAREHOUSE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Platform configurations
PLATFORMS = {
    'amazon': {
        'name': 'Amazon',
        'base_url': 'https://www.amazon.com',
        'enabled': True,
        'fee_structure': {
            'referral_fee': 0.15,  # 15%
            'closing_fee': 0,
            'fulfillment_fee': 'variable'
        }
    },
    'daraz': {
        'name': 'Daraz',
        'base_url': 'https://www.daraz.pk',
        'enabled': True,
        'fee_structure': {
            'commission': 0.05,  # 5%
            'payment_gateway': 0.02  # 2%
        }
    },
    'ebay': {
        'name': 'eBay',
        'base_url': 'https://www.ebay.com',
        'enabled': False,
        'fee_structure': {
            'insertion_fee': 0.35,
            'final_value_fee': 0.125  # 12.5%
        }
    }
}

# Scraper settings
SCRAPER_CONFIG = {
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'timeout': 30,
    'retry_attempts': 3,
    'delay_between_requests': 2,  # seconds
    'use_proxy': False,
    'proxy_list': []
}

# Analysis settings
ANALYSIS_CONFIG = {
    'trend_window_days': 90,
    'forecast_horizon_days': 30,
    'min_data_points': 10,
    'confidence_interval': 0.95
}

# Pricing settings
PRICING_CONFIG = {
    'target_profit_margin': 0.30,  # 30%
    'minimum_roi': 0.20,  # 20%
    'shipping_markup': 0.10,  # 10%
    'currency': 'USD'
}

# Marketing settings
MARKETING_CONFIG = {
    'budget_tiers': {
        'low': 100,
        'medium': 500,
        'high': 2000
    },
    'channels': ['social_media', 'paid_ads', 'influencer', 'content', 'email']
}

# Flask app settings
FLASK_CONFIG = {
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production'),
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': BASE_DIR / 'app.log'
}
