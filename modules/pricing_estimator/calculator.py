"""
Profit, Pricing, and Cost Estimator Module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import WAREHOUSE_DIR, PLATFORMS, PRICING_CONFIG
from utils.logger import get_logger
from utils.helpers import get_timestamp, format_currency

logger = get_logger(__name__)


class PricingCalculator:
    """Calculates optimal pricing and estimates costs and profits with AI enhancement"""
    
    def __init__(self, use_ai: bool = True):
        self.warehouse_dir = WAREHOUSE_DIR
        self.platforms = PLATFORMS
        self.use_ai = use_ai
        self.ai_pricing_model = None
        
        # Initialize AI pricing if enabled
        if use_ai:
            try:
                from .xgboost_pricing import XGBoostPricingModel
                self.ai_pricing_model = XGBoostPricingModel()
                logger.info("AI-powered pricing (XGBoost) enabled")
            except Exception as e:
                logger.warning(f"Could not load AI pricing model: {e}. Using traditional pricing.")
                self.use_ai = False
        
        logger.info(f"PricingCalculator initialized (AI: {self.use_ai})")
    
    def calculate_costs(self, product_cost: float, platform: str, **kwargs) -> Dict[str, Any]:
        """
        Calculate all costs for selling a product
        
        Args:
            product_cost: Base cost of the product
            platform: Marketplace platform (amazon, daraz, etc.)
            **kwargs: Additional params (shipping_cost, packaging_cost, etc.)
        
        Returns:
            Detailed cost breakdown
        """
        logger.info(f"Calculating costs for {platform}")
        
        platform_config = self.platforms.get(platform, {})
        fee_structure = platform_config.get('fee_structure', {})
        
        # Base costs
        costs = {
            'product_cost': product_cost,
            'shipping_cost': kwargs.get('shipping_cost', product_cost * 0.1),
            'packaging_cost': kwargs.get('packaging_cost', product_cost * 0.05),
            'marketing_cost': kwargs.get('marketing_cost', product_cost * 0.15),
        }
        
        # Platform fees (calculated as percentage of selling price estimate)
        # We'll iterate to find the right selling price
        estimated_selling_price = product_cost * (1 + PRICING_CONFIG['target_profit_margin']) / 0.7
        
        if platform == 'amazon':
            costs['referral_fee'] = estimated_selling_price * fee_structure.get('referral_fee', 0.15)
            costs['fulfillment_fee'] = kwargs.get('fulfillment_fee', 3.0)
        elif platform == 'daraz':
            costs['commission'] = estimated_selling_price * fee_structure.get('commission', 0.05)
            costs['payment_gateway_fee'] = estimated_selling_price * fee_structure.get('payment_gateway', 0.02)
        
        # Tax (if applicable)
        costs['tax'] = kwargs.get('tax', 0)
        
        # Calculate total cost
        costs['total_cost'] = sum(costs.values())
        
        logger.info(f"Total cost calculated: ${costs['total_cost']:.2f}")
        return costs
    
    def calculate_pricing(self, product_cost: float, platform: str, 
                         target_margin: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Calculate optimal pricing strategy
        
        Args:
            product_cost: Base cost of the product
            platform: Marketplace platform
            target_margin: Desired profit margin (optional, uses config default)
            **kwargs: Additional parameters (category, quantity, market_data)
        
        Returns:
            Pricing recommendation with metrics
        """
        logger.info(f"Calculating pricing for {platform}")
        
        # Use target margin or default
        if target_margin is None:
            target_margin = PRICING_CONFIG.get('target_profit_margin', 0.30)
        
        # Try AI pricing if enabled
        if self.use_ai and self.ai_pricing_model:
            try:
                product_data = {
                    'cost': product_cost,
                    'quantity': kwargs.get('quantity', 100),
                    'target_margin': target_margin,
                    'category': kwargs.get('category', 'general'),
                    'platform': platform
                }
                
                market_data = kwargs.get('market_data') or self._get_market_pricing(platform, kwargs.get('category'))
                
                # Get AI prediction
                ai_prediction = self.ai_pricing_model.predict_price(product_data, market_data)
                
                if 'predicted_price' in ai_prediction:
                    logger.info(f"Using AI-predicted price: ${ai_prediction['predicted_price']:.2f}")
                    return ai_prediction
            except Exception as e:
                logger.warning(f"AI pricing failed: {e}, using traditional method")
        
        # Traditional pricing calculation
        costs = self.calculate_costs(product_cost, platform, **kwargs)
        total_cost = costs['total_cost']
        
        # Calculate selling price
        recommended_price = total_cost / (1 - target_margin)
        profit = recommended_price - total_cost
        actual_margin = profit / recommended_price if recommended_price > 0 else 0
        
        # Get market data
        market_data = kwargs.get('market_data') or self._get_market_pricing(platform, kwargs.get('category'))
        
        pricing = {
            'recommended_price': round(recommended_price, 2),
            'total_cost': round(total_cost, 2),
            'profit': round(profit, 2),
            'profit_margin': round(actual_margin * 100, 2),
            'platform': platform,
            'target_margin': round(target_margin * 100, 2),
            'market_data': market_data,
            'recommendations': self._generate_pricing_recommendations(
                recommended_price, market_data, actual_margin
            ),
            'accuracy_metrics': self._calculate_pricing_accuracy(market_data, actual_margin),
            'calculated_at': get_timestamp()
        }
        
        logger.info(f"Recommended price: ${pricing['recommended_price']:.2f}, Margin: {pricing['profit_margin']:.1f}%")
        return pricing
    
    def _get_platform_fee_percentage(self, platform: str) -> float:
        """Get total platform fee percentage"""
        fee_structure = self.platforms.get(platform, {}).get('fee_structure', {})
        
        if platform == 'amazon':
            return fee_structure.get('referral_fee', 0.15)
        elif platform == 'daraz':
            return fee_structure.get('commission', 0.05) + fee_structure.get('payment_gateway', 0.02)
        
        return 0.10  # Default 10%
    
    def _get_market_pricing(self, platform: str, category: str) -> Dict[str, Any]:
        """Get market pricing information from warehouse data"""
        try:
            warehouse_file = self.warehouse_dir / f"{platform}_products.csv"
            if not warehouse_file.exists():
                warehouse_file = self.warehouse_dir / "all_platforms_products.csv"
            
            if not warehouse_file.exists():
                return {'message': 'No market data available'}
            
            df = pd.read_csv(warehouse_file)
            
            if category:
                df = df[df['category'].str.lower() == category.lower()]
            
            if df.empty:
                return {'message': 'No market data for this category'}
            
            return {
                'average_price': float(df['price'].mean()),
                'median_price': float(df['price'].median()),
                'price_range': {
                    'min': float(df['price'].min()),
                    'max': float(df['price'].max())
                },
                'percentiles': {
                    '25th': float(df['price'].quantile(0.25)),
                    '75th': float(df['price'].quantile(0.75))
                }
            }
        except Exception as e:
            logger.warning(f"Could not get market pricing: {e}")
            return {'message': 'Market data unavailable'}
    
    def _generate_pricing_recommendations(self, selling_price: float, 
                                          market_data: Dict[str, Any], 
                                          margin: float) -> List[str]:
        """Generate pricing recommendations"""
        recommendations = []
        
        if 'average_price' in market_data:
            avg_price = market_data['average_price']
            
            if selling_price < avg_price * 0.8:
                recommendations.append(f"Your price is {((avg_price - selling_price) / avg_price * 100):.1f}% below market average - competitive advantage")
                recommendations.append("Consider if you can maintain quality at this price point")
            elif selling_price > avg_price * 1.2:
                recommendations.append(f"Your price is {((selling_price - avg_price) / avg_price * 100):.1f}% above market average")
                recommendations.append("Ensure premium features/quality justify higher price")
                recommendations.append("Develop strong brand positioning")
            else:
                recommendations.append("Pricing is competitive with market average")
        
        if margin < 0.20:
            recommendations.append(f"Low profit margin ({margin * 100:.1f}%) - consider cost optimization")
        elif margin > 0.40:
            recommendations.append(f"High profit margin ({margin * 100:.1f}%) - monitor competitor reactions")
        
        recommendations.append("Test pricing with A/B testing if possible")
        recommendations.append("Monitor competitor pricing changes monthly")
        recommendations.append("Consider psychological pricing (e.g., $19.99 vs $20.00)")
        
        return recommendations
    
    def compare_platform_profitability(self, product_cost: float, 
                                       selling_price: float, **kwargs) -> List[Dict[str, Any]]:
        """
        Compare profitability across different platforms
        
        Args:
            product_cost: Base product cost
            selling_price: Intended selling price
            **kwargs: Additional costs
        
        Returns:
            Comparison across platforms
        """
        logger.info("Comparing platform profitability")
        
        comparison = []
        
        for platform_name, platform_config in self.platforms.items():
            if not platform_config.get('enabled', False):
                continue
            
            # Calculate costs for this platform
            costs = self.calculate_costs(product_cost, platform_name, **kwargs)
            profit = selling_price - costs['total_cost']
            margin = (profit / selling_price * 100) if selling_price > 0 else 0
            
            comparison.append({
                'platform': platform_name,
                'platform_display_name': platform_config['name'],
                'total_cost': round(costs['total_cost'], 2),
                'profit': round(profit, 2),
                'profit_margin': round(margin, 2),
                'roi': round((profit / costs['total_cost']) * 100, 2) if costs['total_cost'] > 0 else 0,
                'recommendation': 'Recommended' if margin >= 25 else 'Consider alternatives' if margin >= 15 else 'Low profitability'
            })
        
        # Sort by profit margin
        comparison.sort(key=lambda x: x['profit_margin'], reverse=True)
        
        logger.info(f"Platform comparison complete for {len(comparison)} platforms")
        return comparison
    
    def calculate_dynamic_pricing(self, base_price: float, **factors) -> Dict[str, Any]:
        """
        Calculate dynamic pricing based on various factors
        
        Args:
            base_price: Starting price
            **factors: demand_level, competition_level, inventory_level, seasonality
        
        Returns:
            Dynamic pricing recommendations
        """
        logger.info("Calculating dynamic pricing")
        
        adjustments = {}
        final_price = base_price
        
        # Demand adjustment
        demand = factors.get('demand_level', 'medium')  # high, medium, low
        if demand == 'high':
            adjustments['demand'] = 0.10  # +10%
            final_price *= 1.10
        elif demand == 'low':
            adjustments['demand'] = -0.10  # -10%
            final_price *= 0.90
        else:
            adjustments['demand'] = 0
        
        # Competition adjustment
        competition = factors.get('competition_level', 'medium')  # high, medium, low
        if competition == 'high':
            adjustments['competition'] = -0.05  # -5%
            final_price *= 0.95
        elif competition == 'low':
            adjustments['competition'] = 0.05  # +5%
            final_price *= 1.05
        else:
            adjustments['competition'] = 0
        
        # Inventory adjustment
        inventory = factors.get('inventory_level', 'normal')  # high, normal, low
        if inventory == 'high':
            adjustments['inventory'] = -0.08  # -8% to move inventory
            final_price *= 0.92
        elif inventory == 'low':
            adjustments['inventory'] = 0.05  # +5% due to scarcity
            final_price *= 1.05
        else:
            adjustments['inventory'] = 0
        
        # Seasonality adjustment
        seasonality = factors.get('seasonality', 'neutral')  # peak, neutral, off-peak
        if seasonality == 'peak':
            adjustments['seasonality'] = 0.15  # +15%
            final_price *= 1.15
        elif seasonality == 'off-peak':
            adjustments['seasonality'] = -0.15  # -15%
            final_price *= 0.85
        else:
            adjustments['seasonality'] = 0
        
        total_adjustment = sum(adjustments.values())
        
        return {
            'base_price': round(base_price, 2),
            'dynamic_price': round(final_price, 2),
            'adjustments': {k: f"{v*100:+.1f}%" for k, v in adjustments.items()},
            'total_adjustment': f"{total_adjustment*100:+.1f}%",
            'price_change': round(final_price - base_price, 2),
            'recommendations': [
                f"Adjust price from ${base_price:.2f} to ${final_price:.2f}",
                "Monitor competitor reactions to price changes",
                "Review pricing weekly based on market conditions",
                "A/B test different price points"
            ],
            'accuracy_metrics': self._calculate_dynamic_pricing_accuracy(factors),
            'calculated_at': get_timestamp()
        }
    
    def _calculate_pricing_accuracy(self, market_data: Dict[str, Any], 
                                    margin: float) -> Dict[str, Any]:
        """Calculate accuracy of pricing recommendation"""
        # Base confidence from market data availability
        has_market_data = 'average_price' in market_data
        
        if has_market_data:
            data_confidence = 85
        else:
            data_confidence = 60
        
        # Margin validation
        margin_score = 0
        if 0.15 <= margin <= 0.50:  # Healthy margin range
            margin_score = 90
        elif 0.10 <= margin <= 0.60:
            margin_score = 75
        else:
            margin_score = 60
        
        # Overall accuracy
        overall_accuracy = (data_confidence * 0.6 + margin_score * 0.4)
        
        return {
            'accuracy_score': round(overall_accuracy, 1),
            'confidence_level': 'High' if overall_accuracy >= 75 else 'Medium' if overall_accuracy >= 60 else 'Low',
            'market_data_available': has_market_data,
            'pricing_method': 'Cost-plus with market benchmarking' if has_market_data else 'Cost-plus pricing',
            'reliability': self._get_pricing_reliability(overall_accuracy, has_market_data),
            'recommendation': 'Monitor competitor pricing weekly' if has_market_data else 'Gather more market data for validation'
        }
    
    def _calculate_dynamic_pricing_accuracy(self, factors: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy for dynamic pricing"""
        # Count how many factors were provided
        factors_provided = sum(1 for k, v in factors.items() if v not in ['neutral', 'normal', 'medium', None])
        total_factors = 4  # demand, competition, inventory, seasonality
        
        factor_score = (factors_provided / total_factors) * 100
        
        # Dynamic pricing is inherently less certain
        base_confidence = 70
        confidence = min(base_confidence + (factor_score * 0.2), 85)
        
        return {
            'accuracy_score': round(confidence, 1),
            'confidence_level': 'Medium' if confidence >= 65 else 'Low',
            'factors_considered': factors_provided,
            'total_factors': total_factors,
            'pricing_model': 'Dynamic - market responsive',
            'reliability': 'Adjust based on real-time market feedback',
            'update_frequency': 'Review and adjust weekly'
        }
    
    def _get_pricing_reliability(self, accuracy: float, has_market_data: bool) -> str:
        """Get pricing reliability description"""
        if accuracy >= 80 and has_market_data:
            return "High - Based on comprehensive market analysis"
        elif accuracy >= 65:
            return "Medium - Calculated with available data"
        else:
            return "Low - Limited market data, use as starting point"

