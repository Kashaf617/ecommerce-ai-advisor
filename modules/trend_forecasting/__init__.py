"""Trend and Demand Forecasting Module with AI capabilities"""
from .analyzer import TrendAnalyzer

# Convenience function for AI forecasting
def create_ai_analyzer():
    """Create a TrendAnalyzer with AI enabled"""
    return TrendAnalyzer(use_ai=True)

def create_statistical_analyzer():
    """Create a TrendAnalyzer with only statistical methods"""
    return TrendAnalyzer(use_ai=False)

__all__ = ['TrendAnalyzer', 'create_ai_analyzer', 'create_statistical_analyzer']
