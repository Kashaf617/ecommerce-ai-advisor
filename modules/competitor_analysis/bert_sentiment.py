"""
AI-Powered Competitor Sentiment Analysis using BERT
Sentiment analysis of competitor reviews and product feedback
"""
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import get_logger
from modules.ai_models import AIModelManager

logger = get_logger(__name__)


class BERTSentimentAnalyzer:
    """BERT-based sentiment analysis for competitor insights"""
    
    def __init__(self):
        self.sentiment_model = None
        self._load_model()
    
    def _load_model(self):
        """Load BERT sentiment analysis model"""
        try:
            from transformers import pipeline
            logger.info("Loading BERT sentiment analysis model...")
            
            # Use a pre-trained sentiment model (smaller, faster)
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            logger.info("BERT sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            self.sentiment_model = None
    
    def analyze_reviews(self, reviews: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of product reviews
        
        Args:
            reviews: List of review texts
            
        Returns:
            Sentiment analysis results
        """
        if not self.sentiment_model or not reviews:
            return self._fallback_sentiment_analysis(reviews)
        
        try:
            logger.info(f"Analyzing sentiment for {len(reviews)} reviews...")
            
            # Analyze each review (truncate to 512 tokens for BERT)
            results = []
            for review in reviews[:50]:  # Limit to 50 reviews for speed
                truncated_review = review[:512]
                sentiment = self.sentiment_model(truncated_review)[0]
                results.append(sentiment)
            
            # Calculate aggregate statistics
            positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
            negative_count = sum(1 for r in results if r['label'] == 'NEGATIVE')
            
            avg_score = sum(r['score'] for r in results) / len(results) if results else 0
            
            sentiment_summary = {
                'total_reviews_analyzed': len(results),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'positive_ratio': positive_count / len(results) if results else 0,
                'negative_ratio': negative_count / len(results) if results else 0,
                'average_confidence': avg_score,
                'overall_sentiment': 'POSITIVE' if positive_count > negative_count else 'NEGATIVE',
                'sentiment_score': (positive_count - negative_count) / len(results) if results else 0,
                'is_ml_analysis': True
            }
            
            logger.info(f"Sentiment: {sentiment_summary['overall_sentiment']} ({sentiment_summary['positive_ratio']:.1%} positive)")
            return sentiment_summary
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._fallback_sentiment_analysis(reviews)
    
    def _fallback_sentiment_analysis(self, reviews: List[str]) -> Dict[str, Any]:
        """Simple rule-based fallback sentiment analysis"""
        if not reviews:
            return {
                'total_reviews_analyzed': 0,
                'positive_count': 0,
                'negative_count': 0,
                'positive_ratio': 0.5,
                'negative_ratio': 0.5,
                'average_confidence': 0.6,
                'overall_sentiment': 'NEUTRAL',
                'sentiment_score': 0,
                'is_ml_analysis': False,
                'note': 'Using rule-based fallback'
            }
        
        # Simple keyword-based analysis
        positive_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect']
        negative_keywords = ['bad', 'terrible', 'worst', 'hate', 'poor', 'awful', 'disappointing']
        
        positive_count = 0
        negative_count = 0
        
        for review in reviews[:50]:
            review_lower = review.lower()
            pos_matches = sum(1 for word in positive_keywords if word in review_lower)
            neg_matches = sum(1 for word in negative_keywords if word in review_lower)
            
            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1
        
        total = positive_count + negative_count if (positive_count + negative_count) > 0 else 1
        
        return {
            'total_reviews_analyzed': len(reviews[:50]),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_count / total,
            'negative_ratio': negative_count / total,
            'average_confidence': 0.6,
            'overall_sentiment': 'POSITIVE' if positive_count > negative_count else 'NEGATIVE',
            'sentiment_score': (positive_count - negative_count) / total,
            'is_ml_analysis': False,
            'note': 'Using keyword-based fallback'
        }
    
    def analyze_competitor_sentiment(self, competitor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze competitor's overall sentiment from various sources
        
        Args:
            competitor_data: Competitor information including reviews
            
        Returns:
            Enhanced competitor data with sentiment insights
        """
        # Generate mock reviews if not available
        reviews = competitor_data.get('reviews', self._generate_mock_reviews(competitor_data))
        
        sentiment_results = self.analyze_reviews(reviews)
        
        # Add sentiment to competitor data
        enhanced_competitor = competitor_data.copy()
        enhanced_competitor['sentiment_analysis'] = sentiment_results
        enhanced_competitor['sentiment_enhanced_score'] = self._calculate_sentiment_score(
            competitor_data.get('competitive_score', 70),
            sentiment_results
        )
        
        return enhanced_competitor
    
    def _generate_mock_reviews(self, competitor_data: Dict[str, Any]) -> List[str]:
        """Generate mock reviews based on competitor rating"""
        rating = competitor_data.get('average_rating', 4.0)
        
        if rating >= 4.5:
            reviews = [
                "Excellent product, highly recommend!",
                "Great quality and fast shipping",
                "Love this purchase, exceeded expectations",
                "Best product in this category",
                "Outstanding quality and service"
            ]
        elif rating >= 4.0:
            reviews = [
                "Good product overall",
                "Decent quality for the price",
                "Happy with this purchase",
                "Works as expected",
                "Satisfied with the quality"
            ]
        elif rating >= 3.0:
            reviews = [
                "It's okay, nothing special",
                "Average product",
                "Could be better",
                "Not bad but not great",
                "Acceptable for the price"
            ]
        else:
            reviews = [
                "Disappointed with quality",
                "Not worth the money",
                "Poor quality control",
                "Would not recommend",
                "Below expectations"
            ]
        
        return reviews
    
    def _calculate_sentiment_score(self, base_score: float, sentiment: Dict[str, Any]) -> float:
        """Calculate enhanced score incorporating sentiment"""
        sentiment_multiplier = 1.0 + (sentiment['sentiment_score'] * 0.2)  # ±20% based on sentiment
        enhanced_score = base_score * sentiment_multiplier
        return min(max(enhanced_score, 0), 100)  # Clamp to 0-100
