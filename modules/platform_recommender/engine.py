"""
Platform Recommendation Engine - Recommends best platforms for products
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import PLATFORMS, WAREHOUSE_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp

logger = get_logger(__name__)


class PlatformRecommender:
    """Recommends best e-commerce platforms for specific products"""
    
    def __init__(self):
        self.platforms = PLATFORMS
        self.warehouse_dir = WAREHOUSE_DIR
        logger.info("PlatformRecommender initialized")
    
    def recommend_platforms(self, product_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recommend best platforms for a product
        
        Args:
            product_info: Product details (category, price, target_audience, etc.)
        
        Returns:
            Ranked list of platform recommendations
        """
        logger.info(f"Recommending platforms for {product_info.get('category', 'product')}")
        
        category = product_info.get('category', '')
        price = product_info.get('price', 0)
        target_market = product_info.get('target_market', 'pakistan')  # pakistan, international, etc.
        business_model = product_info.get('business_model', 'ecommerce')
        
        recommendations = []
        
        for platform_name, platform_config in self.platforms.items():
            if not platform_config.get('enabled', True):
                continue
            
            # Calculate platform score
            score = self._calculate_platform_score(
                platform_name, category, price, target_market, business_model
            )
            
            # Get platform insights
            insights = self._get_platform_insights(platform_name, category)
            
            recommendation = {
                'platform': platform_name,
                'platform_name': platform_config['name'],
                'recommendation_score': score,
                'suitability_rating': self._get_suitability_rating(score),
                'fee_structure': platform_config.get('fee_structure', {}),
                'pros': self._get_platform_pros(platform_name, target_market),
                'cons': self._get_platform_cons(platform_name, target_market),
                'best_for': self._get_best_use_cases(platform_name),
                'market_insights': insights,
                'setup_difficulty': self._get_setup_difficulty(platform_name),
                'estimated_reach': self._estimate_reach(platform_name, target_market),
                'accuracy_metrics': self._calculate_recommendation_accuracy(insights, score)
            }
            
            recommendations.append(recommendation)
        
        # Sort by score
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} platform recommendations")
        return recommendations
    
    def _calculate_platform_score(self, platform: str, category: str, 
                                   price: float, target_market: str, 
                                   business_model: str) -> float:
        """Calculate platform suitability score (0-100)"""
        score = 50  # Base score
        
        # Platform-specific scoring
        if platform == 'amazon':
            # Amazon is great for international, premium products
            if target_market == 'international':
                score += 20
            if price > 50:
                score += 15
            if category.lower() in ['electronics', 'books', 'home']:
                score += 15
            # Amazon has higher fees
            score -= 5
        
        elif platform == 'daraz':
            # Daraz is best for Pakistan market
            if target_market == 'pakistan':
                score += 25
            if price < 100:  # PKR friendly
                score += 15
            if category.lower() in ['fashion', 'beauty', 'electronics']:
                score += 15
            # Lower fees
            score += 5
        
        elif platform == 'ebay':
            # eBay for auctions and unique items
            if category.lower() in ['collectibles', 'vintage', 'electronics']:
                score += 15
            if business_model == 'auction':
                score += 20
        
        # Ensure score is within 0-100
        return round(min(max(score, 0), 100), 2)
    
    def _get_suitability_rating(self, score: float) -> str:
        """Convert score to rating"""
        if score >= 80:
            return 'Excellent'
        elif score >= 65:
            return 'Very Good'
        elif score >= 50:
            return 'Good'
        elif score >= 35:
            return 'Fair'
        else:
            return 'Poor'
    
    def _get_platform_pros(self, platform: str, target_market: str) -> List[str]:
        """Get platform advantages"""
        pros_map = {
            'amazon': [
                "Massive global reach and customer base",
                "Trusted brand with high conversion rates",
                "FBA (Fulfillment by Amazon) handles logistics",
                "Prime membership drives sales",
                "Advanced analytics and seller tools"
            ],
            'daraz': [
                "Leading platform in Pakistan and South Asia",
                "Lower fees compared to international platforms",
                "Cash on Delivery widely accepted",
                "Local language and currency support",
                "Growing customer base in region"
            ],
            'ebay': [
                "Auction format for unique pricing",
                "International marketplace",
                "Good for vintage/collectible items",
                "Established platform with loyal users"
            ]
        }
        
        return pros_map.get(platform, ["Popular e-commerce platform"])
    
    def _get_platform_cons(self, platform: str, target_market: str) -> List[str]:
        """Get platform disadvantages"""
        cons_map = {
            'amazon': [
                "High fees (15-20% total)",
                "Intense competition",
                "Strict policies and potential account suspension",
                "Complex setup for international sellers",
                "Requires inventory for FBA"
            ],
            'daraz': [
                "Limited to Pakistan and select Asian markets",
                "Smaller customer base than global platforms",
                "Payment processing can be slower",
                "Less sophisticated seller tools"
            ],
            'ebay': [
                "Declining market share",
                "Complex fee structure",
                "Less mobile-friendly than competitors",
                "Lower traffic compared to Amazon"
            ]
        }
        
        return cons_map.get(platform, ["Platform-specific challenges exist"])
    
    def _get_best_use_cases(self, platform: str) -> List[str]:
        """Get best use cases for platform"""
        use_cases = {
            'amazon': [
                "Premium branded products",
                "Electronics and gadgets",
                "Books and media",
                "International market expansion",
                "Products with consistent demand"
            ],
            'daraz': [
                "Pakistan and South Asian markets",
                "Fashion and apparel",
                "Beauty and personal care",
                "Affordable products under Rs. 10,000",
                "Cash on delivery preferred customers"
            ],
            'ebay': [
                "Vintage and collectible items",
                "Unique one-of-a-kind products",
                "Auction-style selling",
                "International buyers for specific niches"
            ]
        }
        
        return use_cases.get(platform, ["General e-commerce"])
    
    def _get_platform_insights(self, platform: str, category: str) -> Dict[str, Any]:
        """Get platform-specific insights from warehouse data"""
        try:
            warehouse_file = self.warehouse_dir / f"{platform}_products.csv"
            if not warehouse_file.exists():
                return {'message': 'No data available'}
            
            df = pd.read_csv(warehouse_file)
            
            # Filter by category if exists
            if category:
                category_df = df[df['category'].str.lower() == category.lower()]
            else:
                category_df = df
            
            if category_df.empty:
                return {'message': 'No data for this category'}
            
            return {
                'total_products': len(category_df),
                'average_price': float(category_df['price'].mean()),
                'average_rating': float(category_df['rating'].mean()) if 'rating' in category_df.columns else 0,
                'competition_level': 'High' if len(category_df) > 100 else 'Medium' if len(category_df) > 30 else 'Low'
            }
        
        except Exception as e:
            logger.warning(f"Could not get platform insights: {e}")
            return {'message': 'Data unavailable'}
    
    def _get_setup_difficulty(self, platform: str) -> Dict[str, Any]:
        """Get setup difficulty information"""
        difficulty_map = {
            'amazon': {
                'level': 'Moderate to Difficult',
                'time_to_setup': '1-2 weeks',
                'requirements': [
                    'Business registration',
                    'Tax information',
                    'Bank account',
                    'Product identification (UPC/EAN)',
                    'Identity verification'
                ]
            },
            'daraz': {
                'level': 'Easy to Moderate',
                'time_to_setup': '3-5 days',
                'requirements': [
                    'CNIC/Business registration',
                    'Bank account',
                    'Contact information',
                    'Product listings'
                ]
            },
            'ebay': {
                'level': 'Easy',
                'time_to_setup': '1-2 days',
                'requirements': [
                    'Email and phone',
                    'PayPal account',
                    'Product photos',
                    'Shipping methods'
                ]
            }
        }
        
        return difficulty_map.get(platform, {'level': 'Moderate', 'time_to_setup': '1 week'})
    
    def _estimate_reach(self, platform: str, target_market: str) -> Dict[str, Any]:
        """Estimate potential customer reach"""
        # These are approximate estimates
        reach_data = {
            'amazon': {
                'global': {'monthly_visitors': '2.5 billion', 'active_customers': '300 million'},
                'pakistan': {'monthly_visitors': 'Limited', 'active_customers': 'Limited'},
                'international': {'monthly_visitors': '2.5 billion', 'active_customers': '300 million'}
            },
            'daraz': {
                'pakistan': {'monthly_visitors': '50 million', 'active_customers': '10 million'},
                'global': {'monthly_visitors': 'Limited', 'active_customers': 'Limited'},
                'international': {'monthly_visitors': '200 million', 'active_customers': '30 million (Asia)'}
            }
        }
        
        return reach_data.get(platform, {}).get(target_market, {'monthly_visitors': 'Unknown', 'active_customers': 'Unknown'})
    
    def create_multi_platform_strategy(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a multi-platform selling strategy
        
        Args:
            product_info: Product information
        
        Returns:
            Multi-platform strategy
        """
        logger.info("Creating multi-platform strategy")
        
        recommendations = self.recommend_platforms(product_info)
        
        # Select top platforms
        primary_platform = recommendations[0] if recommendations else None
        secondary_platforms = recommendations[1:3] if len(recommendations) > 1 else []
        
        strategy = {
            'product_category': product_info.get('category', ''),
            'primary_platform': primary_platform,
            'secondary_platforms': secondary_platforms,
            'launch_sequence': [
                {
                    'phase': 'Phase 1 (Month 1-2)',
                    'action': f"Launch on {primary_platform['platform_name'] if primary_platform else 'primary platform'}",
                    'reason': 'Test product-market fit on the most suitable platform',
                    'focus': 'Build initial reviews and optimize listing'
                },
                {
                    'phase': 'Phase 2 (Month 3-4)',
                    'action': 'Expand to secondary platform if primary shows success',
                    'reason': 'Diversify sales channels',
                    'focus': 'Leverage learnings from primary platform'
                },
                {
                    'phase': 'Phase 3 (Month 5+)',
                    'action': 'Consider additional platforms based on performance',
                    'reason': 'Maximize market coverage',
                    'focus': 'Optimize operations across all platforms'
                }
            ],
            'inventory_strategy': self._get_inventory_strategy(len(recommendations)),
            'pricing_strategy': {
                'recommendation': 'Platform-specific pricing',
                'details': 'Adjust prices based on fees and competition per platform',
                'tips': [
                    'Consider platform fees when setting prices',
                    'Monitor competitor pricing on each platform',
                    'Test different price points',
                    'Account for currency differences'
                ]
            },
            'marketing_allocation': self._get_marketing_allocation(recommendations),
            'generated_at': get_timestamp()
        }
        
        logger.info("Multi-platform strategy created")
        return strategy
    
    def _get_inventory_strategy(self, platform_count: int) -> str:
        """Recommend inventory strategy based on platform count"""
        if platform_count == 1:
            return "Single platform - maintain dedicated inventory for optimal fulfillment"
        elif platform_count <= 3:
            return "Multi-platform - use centralized inventory with clear allocation per platform"
        else:
            return "Omnichannel - implement advanced inventory management system to prevent overselling"
    
    def _get_marketing_allocation(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recommend marketing budget allocation across platforms"""
        if not recommendations:
            return {}
        
        # Allocate based on recommendation scores
        total_score = sum(r['recommendation_score'] for r in recommendations[:3])
        
        allocations = {}
        for i, rec in enumerate(recommendations[:3]):
            percentage = (rec['recommendation_score'] / total_score) * 100
            allocations[rec['platform_name']] = f"{percentage:.1f}%"
        
        return allocations
    
    def _calculate_recommendation_accuracy(self, insights: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Calculate accuracy of platform recommendation"""
        # Check if we have market insights
        has_insights = 'total_products' in insights and insights.get('total_products', 0) > 0
        
        if has_insights:
            data_confidence = 85
            product_count = insights.get('total_products', 0)
            if product_count > 100:
                data_confidence = 90
            elif product_count > 50:
                data_confidence = 85
            else:
                data_confidence = 75
        else:
            data_confidence = 65
        
        # Scoring confidence
        if score >= 75:
            match_confidence = 90
        elif score >= 60:
            match_confidence = 80
        elif score >= 45:
            match_confidence = 70
        else:
            match_confidence = 60
        
        overall_confidence = (data_confidence * 0.5 + match_confidence * 0.5)
        
        return {
            'confidence_score': round(overall_confidence, 1),
            'confidence_level': 'High' if overall_confidence >= 80 else 'Medium' if overall_confidence >= 65 else 'Low',
            'data_backed': has_insights,
            'recommendation_strength': self._get_recommendation_strength(overall_confidence),
            'validation_note': 'Based on market data analysis' if has_insights else 'Based on platform characteristics'
        }
    
    def _get_recommendation_strength(self, confidence: float) -> str:
        """Get recommendation strength description"""
        if confidence >= 85:
            return "Strong - Highly recommended based on comprehensive analysis"
        elif confidence >= 70:
            return "Good - Recommended with available data support"
        else:
            return "Moderate - Consider as one option among alternatives"
