"""
Competitor Analysis Engine - Identifies and analyzes competitors
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import WAREHOUSE_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp, calculate_percentage_change

logger = get_logger(__name__)


class CompetitorAnalyzer:
    """Analyzes competitors in the marketplace"""
    
    def __init__(self):
        self.warehouse_dir = WAREHOUSE_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CompetitorAnalyzer initialized")
    
    def identify_competitors(self, category: str, user_price_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Identify key competitors in a category
        
        Args:
            category: Product category
            user_price_range: Optional tuple (min_price, max_price) to filter competitors
        
        Returns:
            List of competitor profiles
        """
        logger.info(f"Identifying competitors in category: {category}")
        
        try:
            warehouse_file = self.warehouse_dir / "all_platforms_products.csv"
            if not warehouse_file.exists():
                logger.warning("No warehouse data found")
                return []
            
            df = pd.read_csv(warehouse_file)
            category_df = df[df['category'].str.lower() == category.lower()]
            
            if category_df.empty:
                logger.warning(f"No products found for category: {category}")
                return []
            
            # Filter by price range if provided
            if user_price_range:
                min_price, max_price = user_price_range
                category_df = category_df[
                    (category_df['price'] >= min_price) &
                    (category_df['price'] <= max_price)
                ]
            
            # Group by seller and analyze
            competitors = []
            seller_groups = category_df.groupby('seller')
            
            for seller, seller_df in seller_groups:
                if len(seller_df) >= 1:  # Only include sellers with at least 1 product
                    competitor = {
                        'seller_name': seller,
                        'product_count': len(seller_df),
                        'average_price': float(seller_df['price'].mean()),
                        'price_range': {
                            'min': float(seller_df['price'].min()),
                            'max': float(seller_df['price'].max())
                        },
                        'average_rating': float(seller_df['rating'].mean()) if 'rating' in seller_df.columns else 0,
                        'total_reviews': int(seller_df['reviews_count'].sum()) if 'reviews_count' in seller_df.columns else 0,
                        'platforms': seller_df['platform'].unique().tolist(),
                        'average_discount': float(seller_df.get('discount_percent', 0).mean()) if 'discount_percent' in seller_df.columns else 0,
                        'competitive_score': 0  # Will be calculated
                    }
                    
                    # Calculate competitive score
                    competitor['competitive_score'] = self._calculate_competitive_score(competitor)
                    competitors.append(competitor)
            
            # Sort by competitive score
            competitors.sort(key=lambda x: x['competitive_score'], reverse=True)
            
            logger.info(f"Identified {len(competitors)} competitors")
            return competitors[:20]  # Return top 20 competitors
            
        except Exception as e:
            logger.error(f"Error identifying competitors: {e}")
            return []
    
    def _calculate_competitive_score(self, competitor: Dict[str, Any]) -> float:
        """Calculate competitive strength score (0-100)"""
        score = 0
        
        # Product count (0-30 points)
        score += min(competitor['product_count'] * 2, 30)
        
        # Rating (0-25 points)
        score += (competitor['average_rating'] / 5.0) * 25
        
        # Reviews (0-25 points)
        score += min(competitor['total_reviews'] / 100, 1.0) * 25
        
        # Multi-platform presence (0-20 points)
        score += len(competitor['platforms']) * 10
        
        return round(min(score, 100), 2)
    
    def compare_with_competitors(self, user_product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare user's product idea with competitors
        
        Args:
            user_product: Dictionary with user's product details (price, category, etc.)
            competitors: List of competitor profiles
        
        Returns:
            Comparison analysis
        """
        logger.info("Comparing with competitors")
        
        if not competitors:
            return {'message': 'No competitors found for comparison'}
        
        user_price = user_product.get('price', 0)
        category = user_product.get('category', '')
        
        # Calculate market statistics
        competitor_prices = [c['average_price'] for c in competitors]
        competitor_ratings = [c['average_rating'] for c in competitors]
        
        avg_market_price = np.mean(competitor_prices)
        median_market_price = np.median(competitor_prices)
        avg_market_rating = np.mean(competitor_ratings)
        
        # Determine price positioning
        if user_price < avg_market_price * 0.8:
            price_positioning = 'Budget/Value'
        elif user_price > avg_market_price * 1.2:
            price_positioning = 'Premium'
        else:
            price_positioning = 'Mid-range'
        
        comparison = {
            'category': category,
            'user_price': user_price,
            'market_statistics': {
                'average_price': round(avg_market_price, 2),
                'median_price': round(median_market_price, 2),
                'price_range': {
                    'min': round(min(competitor_prices), 2),
                    'max': round(max(competitor_prices), 2)
                },
                'average_rating': round(avg_market_rating, 2)
            },
            'price_positioning': price_positioning,
            'price_competitiveness': self._assess_price_competitiveness(user_price, avg_market_price),
            'top_competitors': competitors[:5],
            'market_saturation': self._assess_market_saturation(len(competitors)),
            'competitive_advantages': self._identify_advantages(user_product, competitors),
            'recommendations': self._generate_competitive_recommendations(
                user_price, avg_market_price, price_positioning, len(competitors)
            ),
            'accuracy_metrics': self._calculate_analysis_accuracy(len(competitors), competitors),
            'analyzed_at': get_timestamp()
        }
        
        logger.info("Competitor comparison complete")
        return comparison
    
    def _assess_price_competitiveness(self, user_price: float, avg_price: float) -> str:
        """Assess how competitive the user's price is"""
        diff_percent = ((user_price - avg_price) / avg_price) * 100
        
        if diff_percent < -20:
            return 'Very Competitive (significantly below market average)'
        elif diff_percent < -5:
            return 'Competitive (below market average)'
        elif diff_percent < 5:
            return 'Market Average'
        elif diff_percent < 20:
            return 'Above Average (premium positioning)'
        else:
            return 'Significantly Above Market (luxury positioning)'
    
    def _assess_market_saturation(self, competitor_count: int) -> Dict[str, Any]:
        """Assess market saturation level"""
        if competitor_count < 5:
            level = 'Low'
            description = 'Few competitors - potential niche opportunity'
        elif competitor_count < 15:
            level = 'Medium'
            description = 'Moderate competition - balanced market'
        else:
            level = 'High'
            description = 'Many competitors - saturated market, differentiation crucial'
        
        return {
            'level': level,
            'competitor_count': competitor_count,
            'description': description
        }
    
    def _identify_advantages(self, user_product: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
        """Identify potential competitive advantages"""
        advantages = []
        user_price = user_product.get('price', 0)
        
        competitor_prices = [c['average_price'] for c in competitors]
        avg_price = np.mean(competitor_prices)
        
        if user_price < avg_price * 0.9:
            advantages.append("Price advantage - Lower than market average")
        
        avg_products = np.mean([c['product_count'] for c in competitors])
        if avg_products > 5:
            advantages.append("Opportunity to focus on quality over quantity")
        
        multi_platform_sellers = sum(1 for c in competitors if len(c['platforms']) > 1)
        if multi_platform_sellers < len(competitors) * 0.5:
            advantages.append("Multi-platform strategy could provide advantage")
        
        avg_discount = np.mean([c.get('average_discount', 0) for c in competitors])
        if avg_discount > 15:
            advantages.append("Market relies heavily on discounts - opportunity for value positioning")
        
        if not advantages:
            advantages.append("Focus on product quality and customer service for differentiation")
        
        return advantages
    
    def _generate_competitive_recommendations(self, user_price: float, avg_price: float, 
                                              positioning: str, competitor_count: int) -> List[str]:
        """Generate recommendations based on competitive analysis"""
        recommendations = []
        
        # Price recommendations
        if user_price > avg_price * 1.3:
            recommendations.append("Price is significantly above market - ensure premium features justify the cost")
            recommendations.append("Develop strong brand story to support premium positioning")
        elif user_price < avg_price * 0.7:
            recommendations.append("Very competitive pricing - ensure profit margins are sustainable")
            recommendations.append("Emphasize value proposition in marketing")
        
        # Market saturation recommendations
        if competitor_count > 15:
            recommendations.append("High competition - focus on niche differentiation")
            recommendations.append("Invest in unique value propositions and customer experience")
            recommendations.append("Consider partnering with influencers to stand out")
        elif competitor_count < 5:
            recommendations.append("Limited competition - validate market demand before scaling")
            recommendations.append("Opportunity to establish brand as category leader")
        
        # Positioning recommendations
        if positioning == 'Premium':
            recommendations.append("Premium positioning - prioritize quality, branding, and customer service")
        elif positioning == 'Budget/Value':
            recommendations.append("Value positioning - optimize costs and emphasize affordability")
        
        recommendations.append("Monitor competitor pricing and adjust strategy quarterly")
        recommendations.append("Differentiate through superior customer service and product quality")
        
        return recommendations
    
    def analyze_competitor_strategies(self, category: str) -> Dict[str, Any]:
        """
        Analyze common strategies used by successful competitors
        
        Args:
            category: Product category
        
        Returns:
            Analysis of competitor strategies
        """
        logger.info(f"Analyzing competitor strategies for {category}")
        
        try:
            warehouse_file = self.warehouse_dir / "all_platforms_products.csv"
            if not warehouse_file.exists():
                return {}
            
            df = pd.read_csv(warehouse_file)
            category_df = df[df['category'].str.lower() == category.lower()]
            
            if category_df.empty:
                return {}
            
            # Analyze top performers (high ratings + many reviews)
            category_df['performance_score'] = (
                (category_df['rating'] / 5.0) * 0.5 +
                (np.log1p(category_df['reviews_count']) / 10) * 0.5
            )
            
            top_performers = category_df.nlargest(10, 'performance_score')
            
            strategies = {
                'pricing_strategy': {
                    'avg_price_top_performers': float(top_performers['price'].mean()),
                    'price_range': {
                        'min': float(top_performers['price'].min()),
                        'max': float(top_performers['price'].max())
                    }
                },
                'discount_strategy': {
                    'avg_discount': float(top_performers.get('discount_percent', 0).mean()) if 'discount_percent' in top_performers.columns else 0,
                    'use_discounts': len(top_performers[top_performers.get('discount_percent', 0) > 0]) > 5
                },
                'common_features': self._extract_common_features(top_performers),
                'platform_preferences': top_performers['platform'].value_counts().to_dict(),
                'analyzed_at': get_timestamp()
            }
            
            logger.info("Competitor strategy analysis complete")
            return strategies
            
        except Exception as e:
            logger.error(f"Error analyzing competitor strategies: {e}")
            return {}
    
    def _extract_common_features(self, df: pd.DataFrame) -> List[str]:
        """Extract common features from top performers"""
        features = []
        
        if 'prime_eligible' in df.columns and df['prime_eligible'].sum() / len(df) > 0.5:
            features.append("Prime/Fast shipping eligibility")
        
        if 'verified_seller' in df.columns and df['verified_seller'].sum() / len(df) > 0.5:
            features.append("Verified seller status")
        
        avg_rating = df['rating'].mean() if 'rating' in df.columns else 0
        if avg_rating >= 4.5:
            features.append(f"High customer ratings (avg {avg_rating:.1f}/5.0)")
        
        if 'shipping' in df.columns:
            free_shipping_count = df['shipping'].str.contains('Free', case=False, na=False).sum()
            if free_shipping_count / len(df) > 0.5:
                features.append("Free shipping offered")
        
        return features
    
    def _calculate_analysis_accuracy(self, competitor_count: int, 
                                     competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate accuracy of competitor analysis"""
        # Confidence based on competitor count
        if competitor_count >= 15:
            data_confidence = 90
        elif competitor_count >= 10:
            data_confidence = 80
        elif competitor_count >= 5:
            data_confidence = 70
        else:
            data_confidence = 55
        
        # Check data completeness
        completeness_score = 0
        if competitors:
            total_fields = 0
            complete_fields = 0
            for comp in competitors[:5]:  # Check top 5
                if comp.get('average_rating', 0) > 0:
                    complete_fields += 1
                if comp.get('total_reviews', 0) > 0:
                    complete_fields += 1
                if comp.get('product_count', 0) > 0:
                    complete_fields += 1
                total_fields += 3
            
            if total_fields > 0:
                completeness_score = (complete_fields / total_fields) * 100
        
        # Overall accuracy
        overall_accuracy = (data_confidence * 0.6 + completeness_score * 0.4)
        
        return {
            'accuracy_score': round(overall_accuracy, 1),
            'confidence_level': 'High' if overall_accuracy >= 75 else 'Medium' if overall_accuracy >= 55 else 'Low',
            'competitors_analyzed': competitor_count,
            'data_completeness': round(completeness_score, 1),
            'analysis_reliability': self._get_analysis_reliability(overall_accuracy),
            'recommendation': self._get_accuracy_recommendation(competitor_count, overall_accuracy)
        }
    
    def _get_analysis_reliability(self, accuracy: float) -> str:
        """Get reliability description"""
        if accuracy >= 80:
            return "High - Comprehensive competitive data available"
        elif accuracy >= 60:
            return "Medium - Adequate competitor information"
        else:
            return "Low - Limited competitor data, use as preliminary insight"
    
    def _get_accuracy_recommendation(self, count: int, accuracy: float) -> str:
        """Get recommendation based on accuracy"""
        if count < 5:
            return "Consider expanding search to related categories for more competitors"
        elif accuracy < 60:
            return "Data is limited - supplement with manual research"
        else:
            return "Analysis is reliable - confidence in competitive insights"
