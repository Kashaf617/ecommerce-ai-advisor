"""
AI-Powered Marketing Strategy Generator
Uses template-based generation with optional LLM enhancement
"""
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


class AIMarketingStrategyGenerator:
    """AI-enhanced marketing strategy generator"""
    
    def __init__(self, use_llm: bool = False, api_key: str = None):
        self.use_llm = use_llm
        self.api_key = api_key
        self.llm_client = None
        
        if use_llm and api_key:
            self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM (OpenAI GPT or similar)"""
        try:
            import openai
            openai.api_key = self.api_key
            self.llm_client = openai
            logger.info("LLM client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}")
            self.use_llm = False
    
    def generate_strategy(self, product_info: Dict[str, Any], 
                         audience: Dict[str, Any],
                         budget: float) -> Dict[str, Any]:
        """
        Generate AI-powered marketing strategy
        
        Args:
            product_info: Product details
            audience: Target audience profile
            budget: Marketing budget
            
        Returns:
            Complete marketing strategy
        """
        if self.use_llm and self.llm_client:
            return self._generate_llm_strategy(product_info, audience, budget)
        else:
            return self._generate_template_strategy(product_info, audience, budget)
    
    def _generate_llm_strategy(self, product_info, audience, budget) -> Dict[str, Any]:
        """Generate strategy using LLM"""
        try:
            prompt = self._create_prompt(product_info, audience, budget)
            
            response = self.llm_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert e-commerce marketing strategist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            strategy_text = response.choices[0].message.content
            
            return {
                'strategy_text': strategy_text,
                'method': 'GPT-3.5',
                'is_ai_generated': True,
                'confidence': 0.9
            }
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}, using template")
            return self._generate_template_strategy(product_info, audience, budget)
    
    def _create_prompt(self, product_info, audience, budget) -> str:
        """Create LLM prompt"""
        return f"""
Create a concise marketing strategy for:

Product: {product_info.get('product_name', 'N/A')}
Category: {product_info.get('category', 'N/A')}
Price: ${product_info.get('price', 0)}
Target Audience: {audience.get('primary_demographics', {}).get('age_range', 'N/A')}
Budget: ${budget}

Include:
1. Key marketing channels (2-3)
2. Main message/USP
3. Budget allocation
4. Timeline (90 days)
5. Expected ROI

Keep response under 300 words.
"""
    
    def _generate_template_strategy(self, product_info, audience, budget) -> Dict[str, Any]:
        """Generate strategy using templates (AI-enhanced rules)"""
        category = product_info.get('category', 'general').lower()
        price = product_info.get('price', 50)
        
        # AI-like decision tree for channel selection
        channels = self._select_channels(category, price, budget)
        message = self._generate_message(product_info)
        timeline = self._create_timeline()
        
        strategy = {
            'overview': {
                'product': product_info.get('product_name', 'Product'),
                'budget': budget,
                'duration': '90 days',
                'target_roi': '3-5x'
            },
            'channels': channels,
            'key_message': message,
            'timeline': timeline,
            'method': 'AI-Enhanced Templates',
            'is_ai_generated': True,
            'confidence': 0.75,
            'note': 'Using intelligent rule-based generation (add OpenAI API key for GPT)'
        }
        
        # Generate full text
        strategy_text = self._format_strategy_text(strategy)
        strategy['strategy_text'] = strategy_text
        
        return strategy
    
    def _select_channels(self, category, price, budget) -> List[Dict[str, Any]]:
        """AI-like channel selection"""
        channels = []
        
        # Smart channel selection based on product characteristics
        if budget > 5000:
            channels.append({
                'channel': 'Google Ads',
                'allocation': '40%',
                'budget': budget * 0.4,
                'reason': 'High budget allows paid search'
            })
        
        if price < 50 or category in ['fashion', 'beauty']:
            channels.append({
                'channel': 'Instagram/TikTok',
                'allocation': '35%',
                'budget': budget * 0.35,
                'reason': 'Visual appeal, younger audience'
            })
        
        
        return channels[:3]  # Top 3 channels
    
    def _generate_message(self, product_info) -> str:
        """Generate key marketing message"""
        product_name = product_info.get('product_name', 'our product')
        category = product_info.get('category', 'product')
        
        messages = {
            'electronics': f"Cutting-edge {product_name} - Technology that works for you",
            'fashion': f"Style meets comfort - Discover {product_name}",
            'beauty': f"Transform your routine with {product_name}",
            'home': f"Elevate your space with {product_name}",
            'sports': f"Performance and durability - {product_name} delivers"
        }
        
        return messages.get(category, f"Quality {category} - {product_name}")
    
    def _create_timeline(self) -> Dict[str, str]:
        """Create 90-day timeline"""
        return {
            'Days 1-30': 'Launch phase - Build awareness, test channels',
            'Days 31-60': 'Optimization - Scale what works, refine messaging',
            'Days 61-90': 'Growth - Full campaign execution, retargeting'
        }
    
    def _format_strategy_text(self, strategy: Dict[str, Any]) -> str:
        """Format strategy as readable text"""
        text = f"""
MARKETING STRATEGY

Product: {strategy['overview']['product']}
Budget: ${strategy['overview']['budget']:,.0f}
Duration: {strategy['overview']['duration']}
Target ROI: {strategy['overview']['target_roi']}

KEY MESSAGE:
{strategy['key_message']}

MARKETING CHANNELS:
"""
        for i, channel in enumerate(strategy['channels'], 1):
            text += f"\n{i}. {channel['channel']} ({channel['allocation']})\n   - ${channel['budget']:,.0f}\n   - {channel['reason']}\n"
        
        text += "\nTIMELINE:\n"
        for period, activity in strategy['timeline'].items():
            text += f"- {period}: {activity}\n"
        
        return text.strip()
