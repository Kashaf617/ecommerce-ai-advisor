"""
Target Audience Recommender - Profiles and identifies target audiences
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import WAREHOUSE_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp

logger = get_logger(__name__)


class AudienceProfiler:
    """Identifies and profiles target audiences for products"""
    
    def __init__(self):
        self.warehouse_dir = WAREHOUSE_DIR
        logger.info("AudienceProfiler initialized")
    
    def create_audience_profile(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create detailed target audience profile
        
        Args:
            product_info: Product details (category, price, features, etc.)
        
        Returns:
            Comprehensive audience profile
        """
        logger.info(f"Creating audience profile for {product_info.get('category', 'product')}")
        
        category = product_info.get('category', '').lower()
        price = product_info.get('price', 0)
        features = product_info.get('features', [])
        
        # Determine primary demographics
        demographics = self._determine_demographics(category, price)
        
        # Create buyer personas
        personas = self._create_buyer_personas(category, price, demographics)
        
        # Analyze buying behavior
        behavior = self._analyze_buying_behavior(category)
        
        # Geographic targeting
        geography = self._determine_geographic_target(category, price)
        
        # Psychographic profile
        psychographics = self._create_psychographic_profile(category, price)
        
        profile = {
            'product_category': category,
            'price_point': price,
            'primary_demographics': demographics,
            'buyer_personas': personas,
            'buying_behavior': behavior,
            'geographic_targeting': geography,
            'psychographics': psychographics,
            'marketing_channels': self._recommend_marketing_channels(demographics, psychographics),
            'messaging_recommendations': self._get_messaging_recommendations(personas),
            'accuracy_metrics': self._calculate_profiling_accuracy(category, price, personas),
            'generated_at': get_timestamp()
        }
        
        logger.info("Audience profile created")
        return profile
    
    def _determine_demographics(self, category: str, price: float) -> Dict[str, Any]:
        """Determine demographic characteristics"""
        demographics = {
            'age_range': '',
            'gender': '',
            'income_level': '',
            'education': '',
            'occupation': []
        }
        
        # Age based on category and price
        if category in ['electronics', 'gadgets']:
            demographics['age_range'] = '18-45'
            demographics['gender'] = 'All genders (slight male skew)'
        elif category == 'fashion':
            demographics['age_range'] = '18-35'
            demographics['gender'] = 'All genders'
        elif category == 'beauty':
            demographics['age_range'] = '18-45'
            demographics['gender'] = 'Primarily female (70%+)'
        elif category == 'sports':
            demographics['age_range'] = '15-40'
            demographics['gender'] = 'All genders'
        elif category == 'home':
            demographics['age_range'] = '25-55'
            demographics['gender'] = 'All genders (slight female skew)'
        else:
            demographics['age_range'] = '18-50'
            demographics['gender'] = 'All genders'
        
        # Income level based on price
        if price < 20:
            demographics['income_level'] = 'Budget-conscious, Lower to Middle income'
            demographics['education'] = 'High school and above'
        elif price < 100:
            demographics['income_level'] = 'Middle income'
            demographics['education'] = 'College educated or equivalent'
        elif price < 500:
            demographics['income_level'] = 'Upper-middle income'
            demographics['education'] = 'College educated, professionals'
        else:
            demographics['income_level'] = 'High income, affluent'
            demographics['education'] = 'Highly educated professionals'
        
        # Occupation based on category
        occupation_map = {
            'electronics': ['Tech professionals', 'Students', 'Office workers', 'Entrepreneurs'],
            'fashion': ['Young professionals', 'Students', 'Fashion enthusiasts', 'Social media influencers'],
            'beauty': ['Working professionals', 'Homemakers', 'Beauty enthusiasts', 'Students'],
            'sports': ['Athletes', 'Fitness enthusiasts', 'Students', 'Active professionals'],
            'home': ['Homeowners', 'Homemakers', 'Interior design enthusiasts', 'Young families']
        }
        demographics['occupation'] = occupation_map.get(category, ['General consumers'])
        
        return demographics
    
    def _create_buyer_personas(self, category: str, price: float, 
                               demographics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed buyer personas"""
        personas = []
        
        # Primary Persona
        primary = {
            'name': self._get_persona_name(category, 1),
            'type': 'Primary',
            'age': demographics['age_range'].split('-')[0] + ' years old',
            'occupation': demographics['occupation'][0] if demographics['occupation'] else 'Professional',
            'goals': self._get_persona_goals(category, 'primary'),
            'pain_points': self._get_persona_pain_points(category, 'primary'),
            'buying_motivations': self._get_buying_motivations(category, price, 'primary'),
            'online_behavior': self._get_online_behavior(category, 'primary'),
            'preferred_content': self._get_preferred_content(category)
        }
        personas.append(primary)
        
        # Secondary Persona
        if len(demographics['occupation']) > 1:
            secondary = {
                'name': self._get_persona_name(category, 2),
                'type': 'Secondary',
                'age': demographics['age_range'].split('-')[-1] + ' years old',
                'occupation': demographics['occupation'][1],
                'goals': self._get_persona_goals(category, 'secondary'),
                'pain_points': self._get_persona_pain_points(category, 'secondary'),
                'buying_motivations': self._get_buying_motivations(category, price, 'secondary'),
                'online_behavior': self._get_online_behavior(category, 'secondary'),
                'preferred_content': self._get_preferred_content(category)
            }
            personas.append(secondary)
        
        return personas
    
    def _get_persona_name(self, category: str, persona_num: int) -> str:
        """Generate persona name"""
        primary_names = {
            'electronics': 'Tech-Savvy Professional',
            'fashion': 'Trendy Fashionista',
            'beauty': 'Beauty Enthusiast',
            'sports': 'Fitness Focused',
            'home': 'Home Improver'
        }
        
        secondary_names = {
            'electronics': 'Budget-Conscious Student',
            'fashion': 'Classic Dresser',
            'beauty': 'Natural Beauty Seeker',
            'sports': 'Weekend Warrior',
            'home': 'Practical Homeowner'
        }
        
        if persona_num == 1:
            return primary_names.get(category, 'Primary Customer')
        else:
            return secondary_names.get(category, 'Secondary Customer')
    
    def _get_persona_goals(self, category: str, persona_type: str) -> List[str]:
        """Get persona goals"""
        goals_map = {
            'electronics': [
                'Stay updated with latest technology',
                'Improve productivity',
                'Get best value for money',
                'Find reliable products'
            ],
            'fashion': [
                'Express personal style',
                'Stay on-trend',
                'Find quality at reasonable prices',
                'Build versatile wardrobe'
            ],
            'beauty': [
                'Maintain healthy skin',
                'Look presentable',
                'Find effective products',
                'Age gracefully'
            ]
        }
        
        return goals_map.get(category, ['Find quality products', 'Get good value', 'Solve specific needs'])
    
    def _get_persona_pain_points(self, category: str, persona_type: str) -> List[str]:
        """Get persona pain points"""
        pain_points = {
            'electronics': [
                'Overwhelming product choices',
                'Concerned about quality/authenticity',
                'Uncertain about compatibility',
                'Worried about after-sales support'
            ],
            'fashion': [
                'Difficulty finding right size',
                'Uncertain about quality from photos',
                'Concerns about color accuracy',
                'Return/exchange policies'
            ],
            'beauty': [
                'Skin sensitivity/allergies',
                'Unsure which products suit their skin type',
                'Expiration dates and freshness',
                'High prices for quality products'
            ]
        }
        
        return pain_points.get(category, ['Product quality concerns', 'Price concerns', 'Delivery issues'])
    
    def _get_buying_motivations(self, category: str, price: float, persona_type: str) -> List[str]:
        """Get buying motivations"""
        motivations = []
        
        if price < 50:
            motivations.extend(['Value for money', 'Affordability', 'Low risk purchase'])
        else:
            motivations.extend(['Quality', 'Durability', 'Brand reputation'])
        
        category_motivations = {
            'electronics': ['Latest features', 'Performance', 'Brand trust'],
            'fashion': ['Style', 'Fit', 'Versatility'],
            'beauty': ['Effectiveness', 'Natural ingredients', 'Reviews/recommendations'],
            'sports': ['Performance enhancement', 'Durability', 'Comfort']
        }
        
        motivations.extend(category_motivations.get(category, ['Quality', 'Reliability']))
        
        return motivations
    
    def _get_online_behavior(self, category: str, persona_type: str) -> Dict[str, Any]:
        """Get online behavior patterns"""
        return {
            'research_before_buying': 'Yes - reads reviews and compares prices',
            'social_media_usage': 'High - Instagram, Facebook, TikTok',
            'influencer_impact': 'Medium to High',
            'preferred_shopping_time': 'Evenings and weekends',
            'device_preference': 'Mobile-first (70% mobile, 30% desktop)'
        }
    
    def _get_preferred_content(self, category: str) -> List[str]:
        """Get preferred content types"""
        content_map = {
            'electronics': ['Product reviews', 'Comparison videos', 'Tech specs', 'Unboxing videos'],
            'fashion': ['Styling tips', 'Outfit ideas', 'Influencer posts', 'Customer photos'],
            'beauty': ['Tutorials', 'Before/after photos', 'Ingredient analysis', 'Expert reviews'],
            'sports': ['Action videos', 'Performance data', 'Athlete endorsements', 'How-to guides']
        }
        
        return content_map.get(category, ['Product demos', 'Reviews', 'Testimonials', 'Educational content'])
    
    def _analyze_buying_behavior(self, category: str) -> Dict[str, Any]:
        """Analyze buying behavior patterns"""
        return {
            'purchase_frequency': self._get_purchase_frequency(category),
            'average_time_to_purchase': self._get_decision_time(category),
            'price_sensitivity': self._get_price_sensitivity(category),
            'brand_loyalty': self._get_brand_loyalty(category),
            'impulse_buying_tendency': self._get_impulse_tendency(category),
            'influenced_by': ['Reviews', 'Social proof', 'Discounts', 'Free shipping', 'Influencers']
        }
    
    def _get_purchase_frequency(self, category: str) -> str:
        """Get purchase frequency"""
        frequency_map = {
            'electronics': 'Low (1-2 times per year)',
            'fashion': 'Medium to High (Monthly)',
            'beauty': 'High (Weekly to Monthly)',
            'sports': 'Medium (Quarterly)',
            'home': 'Low to Medium (As needed)'
        }
        return frequency_map.get(category, 'Medium (Quarterly)')
    
    def _get_decision_time(self, category: str) -> str:
        """Get average decision-making time"""
        time_map = {
            'electronics': '1-2 weeks (extensive research)',
            'fashion': '1-3 days (moderate research)',
            'beauty': '3-7 days (moderate research)',
            'sports': '1 week (performance focused)',
            'home': '1-2 weeks (practical considerations)'
        }
        return time_map.get(category, '3-7 days')
    
    def _get_price_sensitivity(self, category: str) -> str:
        """Get price sensitivity level"""
        return 'High - actively seeks discounts and compares prices'
    
    def _get_brand_loyalty(self, category: str) -> str:
        """Get brand loyalty level"""
        loyalty_map = {
            'electronics': 'Medium to High - sticks with trusted brands',
            'fashion': 'Low to Medium - willing to try new brands',
            'beauty': 'High - finds what works and sticks with it',
            'sports': 'Medium - brand matters but performance is key'
        }
        return loyalty_map.get(category, 'Medium')
    
    def _get_impulse_tendency(self, category: str) -> str:
        """Get impulse buying tendency"""
        impulse_map = {
            'electronics': 'Low - planned purchases',
            'fashion': 'High - often impulse buys',
            'beauty': 'Medium - mix of planned and impulse',
            'sports': 'Low to Medium - mostly planned'
        }
        return impulse_map.get(category, 'Medium')
    
    def _determine_geographic_target(self, category: str, price: float) -> Dict[str, Any]:
        """Determine geographic targeting"""
        return {
            'primary_markets': ['Urban areas', 'Tier 1 cities', 'Metropolitan regions'],
            'secondary_markets': ['Tier 2 cities', 'Suburban areas'],
            'country_focus': 'Pakistan' if price < 100 else 'Pakistan + International',
            'shipping_considerations': [
                'Offer free shipping for orders above threshold',
                'Partner with local courier services',
                'Consider Cash on Delivery for Pakistan market',
                'International shipping for premium products'
            ]
        }
    
    def _create_psychographic_profile(self, category: str, price: float) -> Dict[str, Any]:
        """Create psychographic profile"""
        return {
            'lifestyle': self._get_lifestyle(category),
            'values': self._get_values(category),
            'interests': self._get_interests(category),
            'personality_traits': self._get_personality(category)
        }
    
    def _get_lifestyle(self, category: str) -> List[str]:
        """Get lifestyle characteristics"""
        lifestyle_map = {
            'electronics': ['Tech-forward', 'Fast-paced', 'Always connected'],
            'fashion': ['Style-conscious', 'Social', 'Trend-aware'],
            'beauty': ['Self-care focused', 'Image-conscious', 'Health aware'],
            'sports': ['Active', 'Health-focused', 'Goal-oriented']
        }
        return lifestyle_map.get(category, ['Modern', 'Busy', 'Value-conscious'])
    
    def _get_values(self, category: str) -> List[str]:
        """Get value system"""
        return ['Quality', 'Value', 'Convenience', 'Authenticity', 'Sustainability']
    
    def _get_interests(self, category: str) -> List[str]:
        """Get interests"""
        interests_map = {
            'electronics': ['Technology', 'Gaming', 'Photography', 'Innovation'],
            'fashion': ['Fashion trends', 'Social media', 'Events', 'Self-expression'],
            'beauty': ['Skincare', 'Makeup', 'Wellness', 'Self-improvement'],
            'sports': ['Fitness', 'Health', 'Outdoor activities', 'Competition']
        }
        return interests_map.get(category, ['Shopping', 'Online browsing', 'Social media'])
    
    def _get_personality(self, category: str) -> List[str]:
        """Get personality traits"""
        return ['Informed decision-maker', 'Value-seeker', 'Quality-conscious', 'Digitally savvy']
    
    def _recommend_marketing_channels(self, demographics: Dict[str, Any], 
                                      psychographics: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend marketing channels based on audience"""
        return {
            'primary_channels': [
                {
                    'channel': 'Instagram',
                    'priority': 'High',
                    'reason': 'Visual platform, high engagement for product categories',
                    'content_type': 'Product photos, Stories, Reels, Influencer partnerships'
                },
                {
                    'channel': 'Facebook',
                    'priority': 'High',
                    'reason': 'Large user base, advanced targeting options',
                    'content_type': 'Sponsored posts, Groups, Marketplace'
                },
                {
                    'channel': 'Google Ads',
                    'priority': 'Medium to High',
                    'reason': 'Capture search intent',
                    'content_type': 'Search ads, Shopping ads, Display ads'
                }
            ],
            'secondary_channels': [
                {'channel': 'TikTok', 'reason': 'Growing platform for younger demographics'},
                {'channel': 'YouTube', 'reason': 'Product reviews and demonstrations'},
                {'channel': 'Email Marketing', 'reason': 'Direct communication with interested customers'}
            ]
        }
    
    def _get_messaging_recommendations(self, personas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get messaging recommendations based on personas"""
        return {
            'key_messages': [
                'Quality you can trust',
                'Best value for your money',
                'Fast and reliable delivery',
                'Customer satisfaction guaranteed'
            ],
            'tone_of_voice': 'Friendly, helpful, trustworthy',
            'communication_style': 'Clear, benefit-focused, solution-oriented',
            'call_to_action_recommendations': [
                'Shop Now',
                'Get Yours Today',
                'Limited Time Offer',
                'Join [X] Happy Customers'
            ]
        }
    
    def _calculate_profiling_accuracy(self, category: str, price: float, personas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate accuracy of audience profiling"""
        # Base confidence on category knowledge
        known_categories = ['electronics', 'fashion', 'beauty', 'sports', 'home']
        
        if category in known_categories:
            category_confidence = 85
        else:
            category_confidence = 70
        
        # Persona completeness
        persona_score = len(personas) * 40  # Max 2 personas = 80
        persona_score = min(persona_score, 80)
        
        overall_confidence = (category_confidence * 0.6 + persona_score * 0.4)
        
        return {
            'confidence_score': round(overall_confidence, 1),
            'confidence_level': 'High' if overall_confidence >= 75 else 'Medium',
            'personas_generated': len(personas),
            'category_familiarity': 'High' if category in known_categories else 'General',
            'data_source': 'Industry research and demographic analysis',
            'recommendation_reliability': self._get_profiling_reliability(overall_confidence)
        }
    
    def _get_profiling_reliability(self, confidence: float) -> str:
        """Get profiling reliability description"""
        if confidence >= 80:
            return "High - Based on comprehensive demographic and psychographic research"
        elif confidence >= 65:
            return "Medium - General audience insights with category patterns"
        else:
            return "Moderate - Use as starting point, validate with audience testing"

