"""
Marketing Strategy Generator - Creates comprehensive marketing strategies
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import MARKETING_CONFIG
from utils.logger import get_logger
from utils.helpers import get_timestamp

logger = get_logger(__name__)


class MarketingStrategyGenerator:
    """Generates comprehensive marketing strategies"""
    
    def __init__(self):
        self.marketing_config = MARKETING_CONFIG
        logger.info("MarketingStrategyGenerator initialized")
    
    def generate_strategy(self, product_info: Dict[str, Any], 
                         audience_profile: Dict[str, Any],
                         platform_recommendations: List[Dict[str, Any]],
                         budget: float = 500) -> Dict[str, Any]:
        """
        Generate comprehensive marketing strategy
        
        Args:
            product_info: Product details
            audience_profile: Target audience profile
            platform_recommendations: Recommended selling platforms
            budget: Marketing budget
        
        Returns:
            Complete marketing strategy
        """
        logger.info(f"Generating marketing strategy with budget ${budget}")
        
        category = product_info.get('category', '')
        price = product_info.get('price', 0)
        
        # Determine budget tier
        budget_tier = self._determine_budget_tier(budget)
        
        # Create strategy
        strategy = {
            'overview': {
                'product_category': category,
                'target_audience': audience_profile.get('primary_demographics', {}),
                'total_budget': budget,
                'budget_tier': budget_tier,
                'strategy_period': '3 months (90 days)',
                'primary_goal': 'Drive sales and build brand awareness'
            },
            'channel_strategy': self._create_channel_strategy(
                audience_profile, platform_recommendations, budget, budget_tier
            ),
            'content_strategy': self._create_content_strategy(
                category, audience_profile
            ),
            'paid_advertising': self._create_paid_ad_strategy(
                budget, budget_tier, category, audience_profile
            ),
            'social_media_strategy': self._create_social_media_strategy(
                category, audience_profile, budget_tier
            ),
            'influencer_marketing': self._create_influencer_strategy(
                budget_tier, category, audience_profile
            ),
            'email_marketing': self._create_email_strategy(
                budget_tier, audience_profile
            ),
            'seo_aso_strategy': self._create_seo_aso_strategy(
                category, platform_recommendations
            ),
            'timeline': self._create_marketing_timeline(),
            'kpis': self._define_kpis(budget_tier),
            'action_items': self._create_action_items(budget_tier),
            'accuracy_metrics': self._calculate_strategy_accuracy(budget, budget_tier, audience_profile),
            'generated_at': get_timestamp()
        }
        
        logger.info("Marketing strategy generated")
        return strategy
    
    def _determine_budget_tier(self, budget: float) -> str:
        """Determine budget tier"""
        tiers = self.marketing_config['budget_tiers']
        
        if budget <= tiers['low']:
            return 'Low'
        elif budget <= tiers['medium']:
            return 'Medium'
        else:
            return 'High'
    
    def _create_channel_strategy(self, audience_profile: Dict[str, Any],
                                 platforms: List[Dict[str, Any]],
                                 budget: float, tier: str) -> Dict[str, Any]:
        """Create marketing channel strategy"""
        channels = []
        
        # Social Media
        channels.append({
            'channel': 'Social Media Marketing',
            'priority': 'High',
            'budget_allocation': self._get_allocation(tier, 'social_media'),
            'platforms': ['Instagram', 'Facebook', 'TikTok'],
            'tactics': [
                'Organic posts (daily)',
                'Paid advertising campaigns',
                'Stories and Reels',
                'Community engagement',
                'User-generated content'
            ]
        })
        
        # Search Advertising
        if tier != 'Low':
            channels.append({
                'channel': 'Search Advertising (Google/Bing)',
                'priority': 'Medium to High',
                'budget_allocation': self._get_allocation(tier, 'search_ads'),
                'tactics': [
                    'Google Shopping ads',
                    'Search keyword campaigns',
                    'Retargeting campaigns',
                    'Display network ads'
                ]
            })
        
        # Content Marketing
        channels.append({
            'channel': 'Content Marketing',
            'priority': 'Medium',
            'budget_allocation': self._get_allocation(tier, 'content'),
            'tactics': [
                'Blog posts and articles',
                'Product reviews',
                'How-to guides',
                'Video content',
                'Customer testimonials'
            ]
        })
        
        # Email Marketing
        channels.append({
            'channel': 'Email Marketing',
            'priority': 'Medium',
            'budget_allocation': self._get_allocation(tier, 'email'),
            'tactics': [
                'Welcome email series',
                'Abandoned cart recovery',
                'Product recommendations',
                'Seasonal promotions',
                'Customer loyalty program'
            ]
        })
        
        return {
            'channels': channels,
            'focus': 'Multi-channel approach with emphasis on social media and paid advertising'
        }
    
    def _get_allocation(self, tier: str, channel: str) -> str:
        """Get budget allocation percentage for channel"""
        allocations = {
            'Low': {
                'social_media': '60%',
                'search_ads': '0%',
                'content': '30%',
                'email': '10%',
                'influencer': '0%'
            },
            'Medium': {
                'social_media': '40%',
                'search_ads': '30%',
                'content': '15%',
                'email': '10%',
                'influencer': '5%'
            },
            'High': {
                'social_media': '35%',
                'search_ads': '30%',
                'content': '15%',
                'email': '10%',
                'influencer': '10%'
            }
        }
        
        return allocations.get(tier, {}).get(channel, '10%')
    
    def _create_content_strategy(self, category: str, 
                                 audience_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create content marketing strategy"""
        personas = audience_profile.get('buyer_personas', [])
        preferred_content = []
        if personas:
            preferred_content = personas[0].get('preferred_content', [])
        
        return {
            'content_pillars': [
                'Product education',
                'Customer success stories',
                'Industry trends',
                'Behind-the-scenes',
                'Promotional content'
            ],
            'content_types': preferred_content or [
                'Product photos and videos',
                'Customer testimonials',
                'How-to guides',
                'Comparison content',
                'User-generated content'
            ],
            'posting_frequency': {
                'Instagram': '1-2 posts per day + Stories',
                'Facebook': '1 post per day',
                'TikTok': '3-5 videos per week',
                'Blog': '2-3 articles per month'
            },
            'content_themes': self._get_content_themes(category)
        }
    
    def _get_content_themes(self, category: str) -> List[str]:
        """Get content themes based on category"""
        themes_map = {
            'electronics': ['Tech tips', 'Product comparisons', 'Setup guides', 'Tech news'],
            'fashion': ['Style tips', 'Outfit ideas', 'Trend reports', 'Fashion hacks'],
            'beauty': ['Beauty tutorials', 'Skincare routines', 'Product reviews', 'Before/after'],
            'sports': ['Workout tips', 'Performance guides', 'Athlete stories', 'Product tests']
        }
        
        return themes_map.get(category, ['Product features', 'Customer stories', 'How-to guides', 'Tips and tricks'])
    
    def _create_paid_ad_strategy(self, budget: float, tier: str, 
                                 category: str, audience: Dict[str, Any]) -> Dict[str, Any]:
        """Create paid advertising strategy"""
        if tier == 'Low':
            daily_budget = budget * 0.6 / 90  # 60% of budget over 90 days
            return {
                'recommendation': 'Focus on organic growth with minimal paid ads',
                'platforms': ['Facebook Ads (boosted posts)'],
                'daily_budget': f"${daily_budget:.2f}",
                'campaign_types': ['Boosted posts', 'Simple conversion ads'],
                'targeting': 'Broad targeting with basic demographics'
            }
        
        paid_budget = budget * 0.35  # 35% for paid ads
        daily_budget = paid_budget / 90
        
        return {
            'total_budget': f"${paid_budget:.2f}",
            'daily_budget': f"${daily_budget:.2f}",
            'platforms': {
                'Facebook/Instagram Ads': {
                    'budget': f"${paid_budget * 0.5:.2f}",
                    'campaign_types': [
                        'Traffic campaigns',
                        'Conversion campaigns',
                        'Dynamic product ads',
                        'Retargeting campaigns'
                    ],
                    'targeting': self._get_ad_targeting(audience)
                },
                'Google Ads': {
                    'budget': f"${paid_budget * 0.5:.2f}",
                    'campaign_types': [
                        'Google Shopping',
                        'Search ads',
                        'Display remarketing'
                    ]
                } if tier != 'Low' else None
            },
            'optimization_tips': [
                'Start with small daily budgets and scale winners',
                'A/B test ad creatives and copy',
                'Use retargeting for abandoned carts',
                'Track ROAS (Return on Ad Spend) closely',
                'Optimize for conversions, not just clicks'
            ]
        }
    
    def _get_ad_targeting(self, audience: Dict[str, Any]) -> Dict[str, Any]:
        """Get ad targeting parameters"""
        demographics = audience.get('primary_demographics', {})
        
        return {
            'age': demographics.get('age_range', '18-65'),
            'location': 'Pakistan (primary), expand based on performance',
            'interests': audience.get('psychographics', {}).get('interests', []),
            'behaviors': 'Online shoppers, engaged shoppers',
            'custom_audiences': [
                'Website visitors',
                'Email subscribers',
                'Cart abandoners',
                'Past purchasers'
            ]
        }
    
    def _create_social_media_strategy(self, category: str, 
                                      audience: Dict[str, Any],
                                      tier: str) -> Dict[str, Any]:
        """Create social media strategy"""
        return {
            'platforms': {
                'Instagram': {
                    'priority': 'Primary',
                    'goals': ['Brand awareness', 'Product discovery', 'Sales'],
                    'tactics': [
                        'Feed posts showcasing products',
                        'Stories with behind-the-scenes content',
                        'Reels for viral reach',
                        'Instagram Shopping integration',
                        'Influencer collaborations',
                        'User-generated content campaigns'
                    ],
                    'hashtag_strategy': 'Mix of branded, category, and trending hashtags (15-20 per post)'
                },
                'Facebook': {
                    'priority': 'Primary',
                    'goals': ['Community building', 'Customer service', 'Sales'],
                    'tactics': [
                        'Create business page with complete info',
                        'Join and engage in relevant groups',
                        'Facebook Shop setup',
                        'Live videos for product launches',
                        'Customer testimonials and reviews',
                        'Facebook Marketplace listings'
                    ]
                },
                'TikTok': {
                    'priority': 'Secondary' if tier != 'Low' else 'Optional',
                    'goals': ['Viral reach', 'Younger audience', 'Brand awareness'],
                    'tactics': [
                        'Entertaining product demonstrations',
                        'Trend participation',
                        'Educational content',
                        'Collaboration with TikTok creators'
                    ]
                }
            },
            'engagement_strategy': [
                'Respond to all comments within 24 hours',
                'Engage with followers by liking/commenting',
                'Run polls and questions in Stories',
                'Host giveaways and contests',
                'Share user-generated content'
            ]
        }
    
    def _create_influencer_strategy(self, tier: str, category: str,
                                    audience: Dict[str, Any]) -> Dict[str, Any]:
        """Create influencer marketing strategy"""
        if tier == 'Low':
            return {
                'recommendation': 'Focus on micro influencers and product seeding',
                'approach': 'Product gifting in exchange for reviews',
                'target_influencers': 'Nano influencers (1K-10K followers)',
                'budget': 'Product costs only'
            }
        
        return {
            'influencer_tiers': [
                {
                    'tier': 'Micro influencers (10K-100K)',
                    'quantity': '5-10 influencers',
                    'approach': 'Paid collaborations + product gifting',
                    'expected_cost': '$50-200 per post',
                    'benefits': 'Higher engagement rates, niche audiences'
                },
                {
                    'tier': 'Nano influencers (1K-10K)',
                    'quantity': '20-50 influencers',
                    'approach': 'Product gifting + affiliate commissions',
                    'expected_cost': 'Product costs + 10% commission',
                    'benefits': 'Authentic reviews, cost-effective'
                }
            ],
            'collaboration_types': [
                'Sponsored posts and Stories',
                'Product reviews',
                'Unboxing videos',
                'Giveaway partnerships',
                'Affiliate partnerships'
            ],
            'selection_criteria': [
                'Audience alignment with target market',
                'Engagement rate > 3%',
                'Authentic content style',
                'Previous brand collaborations',
                'Content quality'
            ]
        }
    
    def _create_email_strategy(self, tier: str, audience: Dict[str, Any]) -> Dict[str, Any]:
        """Create email marketing strategy"""
        return {
            'email_sequences': [
                {
                    'name': 'Welcome Series',
                    'emails': 3,
                    'timeline': 'Day 1, 3, 7',
                    'goals': 'Introduce brand, build trust, convert'
                },
                {
                    'name': 'Abandoned Cart',
                    'emails': 3,
                    'timeline': '1 hour, 24 hours, 72 hours',
                    'goals': 'Recover lost sales'
                },
                {
                    'name': 'Post-Purchase',
                    'emails': 2,
                    'timeline': '1 day, 7 days',
                    'goals': 'Request review, cross-sell'
                }
            ],
            'regular_campaigns': [
                'Weekly newsletter with new products',
                'Exclusive offers for subscribers',
                'Seasonal promotions',
                'Re-engagement campaigns'
            ],
            'list_building_tactics': [
                'Pop-up signup forms with discount offer',
                'Lead magnets (buying guides, checklists)',
                'Exit-intent popups',
                'Social media to email campaigns'
            ],
            'tools_recommended': ['Mailchimp (free tier)', 'Sendinblue', 'ConvertKit'] if tier == 'Low' else ['Klaviyo', 'ActiveCampaign']
        }
    
    def _create_seo_aso_strategy(self, category: str, 
                                 platforms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create SEO and ASO (App Store Optimization) strategy"""
        return {
            'product_listing_optimization': [
                'Use keyword-rich product titles',
                'Write detailed, benefit-focused descriptions',
                'Include high-quality images (minimum 5)',
                'Add video demonstrations',
                'Encourage customer reviews',
                'Use all available product attributes/tags',
                'Optimize for mobile viewing'
            ],
            'keyword_strategy': [
                f'Research top keywords for {category} products',
                'Use long-tail keywords for less competition',
                'Include keywords in title, description, tags',
                'Monitor competitor keywords',
                'Update keywords based on performance'
            ],
            'review_strategy': [
                'Request reviews from satisfied customers',
                'Respond to all reviews (positive and negative)',
                'Address negative reviews professionally',
                'Use review insights for improvement',
                'Showcase positive reviews in marketing'
            ]
        }
    
    def _create_marketing_timeline(self) -> List[Dict[str, Any]]:
        """Create 90-day marketing timeline"""
        return [
            {
                'phase': 'Month 1: Launch & Foundation',
                'weeks': '1-4',
                'objectives': [
                    'Set up all marketing channels',
                    'Create initial content library',
                    'Launch social media presence',
                    'Begin organic posting',
                    'Set up email capture'
                ],
                'key_activities': [
                    'Week 1: Channel setup and branding',
                    'Week 2: Content creation and scheduling',
                    'Week 3: Launch campaigns and paid ads',
                    'Week 4: Monitor and optimize'
                ]
            },
            {
                'phase': 'Month 2: Growth & Optimization',
                'weeks': '5-8',
                'objectives': [
                    'Scale what\'s working',
                    'Optimize underperforming channels',
                    'Build email list',
                    'Increase social following',
                    'Launch influencer partnerships'
                ],
                'key_activities': [
                    'Week 5-6: Launch influencer collaborations',
                    'Week 7: Run first major promotion',
                    'Week 8: Analyze results and adjust strategy'
                ]
            },
            {
                'phase': 'Month 3: Scaling & Retention',
                'weeks': '9-12',
                'objectives': [
                    'Scale successful campaigns',
                    'Focus on customer retention',
                    'Build brand loyalty',
                    'Prepare for next quarter'
                ],
                'key_activities': [
                    'Week 9-10: Scale winning ads',
                    'Week 11: Launch loyalty program',
                    'Week 12: Comprehensive analysis and planning'
                ]
            }
        ]
    
    def _define_kpis(self, tier: str) -> Dict[str, Any]:
        """Define Key Performance Indicators"""
        return {
            'sales_metrics': [
                {'metric': 'Total Revenue', 'target': 'Track baseline, aim for growth'},
                {'metric': 'Conversion Rate', 'target': '2-5% (varies by platform)'},
                {'metric': 'Average Order Value', 'target': 'Track and optimize'},
                {'metric': 'Customer Acquisition Cost (CAC)', 'target': '< 30% of order value'}
            ],
            'marketing_metrics': [
                {'metric': 'Website/Store Traffic', 'target': 'Increase 100% MoM'},
                {'metric': 'Social Media Followers', 'target': '+500-1000 per month'},
                {'metric': 'Email List Growth', 'target': '+200-500 subscribers per month'},
                {'metric': 'Engagement Rate', 'target': '> 3% on social media'},
                {'metric': 'ROAS (Return on Ad Spend)', 'target': '> 3:1'}
            ],
            'content_metrics': [
                {'metric': 'Content Reach', 'target': 'Track and grow'},
                {'metric': 'Click-Through Rate', 'target': '> 2%'},
                {'metric': 'Video Views', 'target': 'Increase monthly'}
            ]
        }
    
    def _create_action_items(self, tier: str) -> List[Dict[str, Any]]:
        """Create immediate action items"""
        return [
            {
                'priority': 'High',
                'action': 'Set up social media business accounts',
                'timeline': 'Week 1',
                'owner': 'Marketing team'
            },
            {
                'priority': 'High',
                'action': 'Create product photography and videos',
                'timeline': 'Week 1-2',
                'owner': 'Content team'
            },
            {
                'priority': 'High',
                'action': 'Set up Facebook/Instagram Ads account',
                'timeline': 'Week 1',
                'owner': 'Marketing team'
            },
            {
                'priority': 'Medium',
                'action': 'Create content calendar for 30 days',
                'timeline': 'Week 2',
                'owner': 'Content team'
            },
            {
                'priority': 'Medium',
                'action': 'Research and contact micro-influencers',
                'timeline': 'Week 2-3',
                'owner': 'Marketing team'
            },
            {
                'priority': 'Medium',
                'action': 'Set up email marketing platform',
                'timeline': 'Week 2',
                'owner': 'Marketing team'
            },
            {
                'priority': 'Low',
                'action': 'Create blog content plan',
                'timeline': 'Week 3-4',
                'owner': 'Content team'
            }
        ]
    
    def _calculate_strategy_accuracy(self, budget: float, tier: str, audience: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy of marketing strategy"""
        # Base confidence on budget adequacy
        if tier == 'High':
            budget_confidence = 90
        elif tier == 'Medium':
            budget_confidence = 80
        else:
            budget_confidence = 70
        
        # Audience data completeness
        has_personas = len(audience.get('buyer_personas', [])) > 0
        has_demographics = 'primary_demographics' in audience
        
        audience_score = 0
        if has_personas:
            audience_score += 50
        if has_demographics:
            audience_score += 50
        
        overall_confidence = (budget_confidence * 0.5 + audience_score * 0.5)
        
        return {
            'confidence_score': round(overall_confidence, 1),
            'confidence_level': 'High' if overall_confidence >= 75 else 'Medium' if overall_confidence >= 60 else 'Low',
            'budget_tier': tier,
            'strategy_completeness': '100%',
            'data_backed_recommendations': has_personas and has_demographics,
            'recommendation_reliability': self._get_strategy_reliability(overall_confidence, tier),
            'implementation_readiness': 'Ready to execute' if tier != 'Low' else 'Budget-conscious approach'
        }
    
    def _get_strategy_reliability(self, confidence: float, tier: str) -> str:
        """Get strategy reliability description"""
        if confidence >= 85:
            return f"High - Comprehensive {tier}-tier strategy with proven tactics"
        elif confidence >= 70:
            return f"Good - Solid {tier}-tier strategy, adjust based on early results"
        else:
            return "Moderate - Foundational strategy, requires close monitoring and iteration"

