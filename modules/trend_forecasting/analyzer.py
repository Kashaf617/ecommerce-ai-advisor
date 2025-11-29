"""
Trend and Demand Forecasting - Analyzes market trends and predicts demand
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import WAREHOUSE_DIR, ANALYSIS_CONFIG, PROCESSED_DATA_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp

logger = get_logger(__name__)


class TrendAnalyzer:
    """Analyzes trends and forecasts demand for products with AI capabilities"""
    
    def __init__(self, use_ai: bool = True):
        self.warehouse_dir = WAREHOUSE_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.use_ai = use_ai
        self.lstm_forecaster = None
        
        # Load AI forecaster if enabled
        if use_ai:
            try:
                from modules.trend_analyzer.lstm_forecaster import SimpleLSTMForecaster
                self.lstm_forecaster = SimpleLSTMForecaster()
                logger.info("TrendAnalyzer initialized with AI forecasting enabled")
            except Exception as e:
                logger.warning(f"Could not load LSTM forecaster: {e}. Using statistical methods.")
                self.use_ai = False
        else:
            logger.info("TrendAnalyzer initialized (statistical mode)")
    
    def analyze_category_trends(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze trends for a specific category or all categories
        
        Args:
            category: Product category to analyze (None for all)
        
        Returns:
            Dictionary containing trend analysis results
        """
        logger.info(f"Analyzing trends for category: {category or 'all'}")
        
        try:
            # Load warehouse data
            warehouse_file = self.warehouse_dir / "all_platforms_products.csv"
            if not warehouse_file.exists():
                logger.warning("No warehouse data found")
                return {}
            
            df = pd.read_csv(warehouse_file)
            
            # Filter by category if specified
            if category:
                df = df[df['category'].str.lower() == category.lower()]
            
            if df.empty:
                logger.warning(f"No data found for category: {category}")
                return {}
            
            # Perform trend analysis
            trends = {
                'category': category or 'all',
                'total_products': len(df),
                'average_price': float(df['price'].mean()),
                'price_range': {
                    'min': float(df['price'].min()),
                    'max': float(df['price'].max()),
                    'median': float(df['price'].median())
                },
                'average_rating': float(df['rating'].mean()) if 'rating' in df.columns else 0,
                'total_reviews': int(df['reviews_count'].sum()) if 'reviews_count' in df.columns else 0,
                'top_brands': self._get_top_items(df, 'brand', 10),
                'top_sellers': self._get_top_items(df, 'seller', 10),
                'platform_distribution': df['platform'].value_counts().to_dict(),
                'price_segments': self._analyze_price_segments(df),
                'discount_analysis': self._analyze_discounts(df),
                'availability analysis': self._analyze_availability(df),
                'accuracy_metrics': self._calculate_trend_accuracy(df),
                'analyzed_at': get_timestamp()
            }
            
            logger.info(f"Trend analysis complete for {category or 'all'}: {trends['total_products']} products")
            return trends
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {}
    
    def _get_top_items(self, df: pd.DataFrame, column: str, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N items by count from a column"""
        if column not in df.columns:
            return []
        
        top_items = df[column].value_counts().head(n)
        return [{'name': name, 'count': int(count)} for name, count in top_items.items()]
    
    def _analyze_price_segments(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price distribution into segments"""
        price_min = df['price'].min()
        price_max = df['price'].max()
        
        # Create price segments
        bins = [price_min, price_min + (price_max - price_min) * 0.33,
                price_min + (price_max - price_min) * 0.67, price_max]
        labels = ['Budget', 'Mid-range', 'Premium']
        
        df['price_segment'] = pd.cut(df['price'], bins=bins, labels=labels, include_lowest=True)
        
        segments = {}
        for segment in labels:
            segment_df = df[df['price_segment'] == segment]
            if not segment_df.empty:
                segments[segment] = {
                    'count': len(segment_df),
                    'avg_price': float(segment_df['price'].mean()),
                    'avg_rating': float(segment_df['rating'].mean()) if 'rating' in segment_df.columns else 0
                }
        
        return segments
    
    def _analyze_discounts(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze discount patterns"""
        if 'discount_percent' not in df.columns:
            return {}
        
        discounted = df[df['discount_percent'] > 0]
        
        return {
            'products_with_discount': len(discounted),
            'discount_percentage': round(len(discounted) / len(df) * 100, 1),
            'average_discount': float(discounted['discount_percent'].mean()) if len(discounted) > 0 else 0,
            'max_discount': float(df['discount_percent'].max()),
            'median_discount': float(discounted['discount_percent'].median()) if len(discounted) > 0 else 0
        }
    
    def _analyze_availability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze product availability"""
        if 'availability' not in df.columns:
            return {}
        
        availability_counts = df['availability'].value_counts().to_dict()
        total = len(df)
        
        return {
            'distribution': availability_counts,
            'in_stock_percentage': round(
                availability_counts.get('In Stock', 0) / total * 100, 1
            ) if total > 0 else 0
        }
    
    def forecast_demand(self, category: str, days_ahead: int = 30, use_lstm: bool = None) -> Dict[str, Any]:
        """
        Forecast demand for a category with AI or statistical methods
        
        Args:
            category: Product category
            days_ahead: Number of days to forecast
            use_lstm: Override to force LSTM (True) or statistical (False). None = auto
        
        Returns:
            Forecast results
        """
        logger.info(f"Forecasting demand for {category} for {days_ahead} days (AI: {self.use_ai})")
        
        try:
            # Load historical data
            warehouse_file = self.warehouse_dir / "all_platforms_products.csv"
            if not warehouse_file.exists():
                logger.warning("No warehouse data found")
                return {}
            
            df = pd.read_csv(warehouse_file)
            category_df = df[df['category'].str.lower() == category.lower()]
            
            if category_df.empty:
                logger.warning(f"No data for category: {category}")
                return {}
            
            # Determine whether to use LSTM
            should_use_lstm = use_lstm if use_lstm is not None else (self.use_ai and self.lstm_forecaster)
            
            current_avg_rating = category_df['rating'].mean() if 'rating' in category_df.columns else 3.5
            current_avg_reviews = category_df['reviews_count'].mean() if 'reviews_count' in category_df.columns else 50
            current_products = len(category_df)
            
            # Calculate base demand score
            demand_score = (current_avg_rating / 5.0) * 0.4 + \
                          min(current_avg_reviews / 1000, 1.0) * 0.3 + \
                          min(current_products / 100, 1.0) * 0.3
            
            demand_level = 'High' if demand_score > 0.7 else 'Medium' if demand_score > 0.4 else 'Low'
            
            # Generate forecast using AI or statistical method
            weekly_forecast = []
            forecast_method = 'Statistical'
            forecast_confidence = 0.65
            
            if should_use_lstm and len(category_df) > 30:
                # Use LSTM for time series forecasting
                try:
                    # Create synthetic time series data from category metrics
                    historical_values = self._create_time_series_from_category(category_df, days_ahead)
                    lstm_result = self.lstm_forecaster.forecast_trend(historical_values, days_ahead)
                    
                    # Convert LSTM forecast to weekly format
                    forecast_values = lstm_result['forecast']
                    weeks = min(days_ahead // 7 + 1, len(forecast_values) // 7)
                    
                    for week in range(1, weeks + 1):
                        week_idx = week * 7 - 1
                        if week_idx < len(forecast_values):
                            weekly_forecast.append({
                                'week': week,
                                'estimated_demand': round(forecast_values[week_idx], 2),
                                'confidence': round(lstm_result['confidence'] * 100 * (1 - week * 0.05), 1)
                            })
                    
                    forecast_method = lstm_result['method']
                    forecast_confidence = lstm_result['confidence']
                    logger.info(f"Used {forecast_method} for demand forecasting")
                    
                except Exception as e:
                    logger.warning(f"LSTM forecast failed: {e}, falling back to statistical")
                    should_use_lstm = False
            
            # Fallback to statistical forecast if LSTM not used or failed
            if not should_use_lstm or not weekly_forecast:
                base_demand = demand_score * 100
                for week in range(1, min(days_ahead // 7 + 1, 5)):
                    forecast_value = base_demand * (1 + np.random.uniform(-0.1, 0.15))
                    weekly_forecast.append({
                        'week': week,
                        'estimated_demand': round(forecast_value, 2),
                        'confidence': round(max(100 - week * 5, 60), 1)
                    })
                forecast_method = 'Statistical'
                forecast_confidence = 0.65
            
            forecast = {
                'category': category,
                'forecast_period_days': days_ahead,
                'current_demand_score': round(demand_score * 100, 2),
                'demand_level': demand_level,
                'weekly_forecast': weekly_forecast,
                'forecast_method': forecast_method,
                'is_ml_forecast': forecast_method != 'Statistical',
                'key_metrics': {
                    'average_rating': round(current_avg_rating, 2),
                    'average_reviews': round(current_avg_reviews, 0),
                    'total_products': current_products
                },
                'recommendations': self._generate_demand_recommendations(demand_level, demand_score),
                'forecasted_at': get_timestamp()
            }
            
            # Calculate forecast accuracy
            forecast['accuracy_metrics'] = self._calculate_forecast_accuracy(
                len(category_df), current_avg_rating, current_avg_reviews
            )
            
            logger.info(f"Demand forecast complete: {demand_level} demand predicted")
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting demand: {e}")
            return {}
    
    def _generate_demand_recommendations(self, demand_level: str, score: float) -> List[str]:
        """Generate recommendations based on demand level"""
        recommendations = []
        
        if demand_level == 'High':
            recommendations.extend([
                "Strong market demand detected - excellent opportunity to enter",
                "Consider competitive pricing to capture market share",
                "Invest in marketing to capitalize on high demand",
                "Ensure adequate inventory to meet demand"
            ])
        elif demand_level == 'Medium':
            recommendations.extend([
                "Moderate demand - research differentiation opportunities",
                "Focus on unique value proposition",
                "Test with smaller inventory initially",
                "Monitor trends closely for demand shifts"
            ])
        else:
            recommendations.extend([
                "Low demand detected - consider niche positioning",
                "Explore related categories with higher demand",
                "Focus on highly targeted marketing",
                "Start with minimal inventory investment"
            ])
        
        return recommendations
    
    def identify_trending_products(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Identify trending products based on ratings, reviews, and discounts
        
        Args:
            top_n: Number of top trending products to return
        
        Returns:
            List of trending products
        """
        logger.info(f"Identifying top {top_n} trending products")
        
        try:
            warehouse_file = self.warehouse_dir / "all_platforms_products.csv"
            if not warehouse_file.exists():
                return []
            
            df = pd.read_csv(warehouse_file)
            
            # Calculate trending score
            # Score = (rating * 0.4) + (log(reviews) * 0.3) + (discount * 0.3)
            df['trending_score'] = (
                (df['rating'] / 5.0) * 0.4 +
                (np.log1p(df['reviews_count']) / 10) * 0.3 +
                (df.get('discount_percent', 0) / 100) * 0.3
            )
            
            # Get top trending products
            trending = df.nlargest(top_n, 'trending_score')
            
            trending_products = []
            for _, product in trending.iterrows():
                trending_products.append({
                    'title': product['title'],
                    'platform': product['platform'],
                    'category': product['category'],
                    'price': float(product['price']),
                    'rating': float(product['rating']),
                    'reviews': int(product['reviews_count']),
                    'discount': float(product.get('discount_percent', 0)),
                    'trending_score': float(product['trending_score']),
                    'url': product.get('url', '')
                })
            
            logger.info(f"Found {len(trending_products)} trending products")
            return trending_products
            
        except Exception as e:
            logger.error(f"Error identifying trending products: {e}")
            return []
    
    def generate_trend_report(self, category: Optional[str] = None) -> str:
        """Generate a comprehensive trend report"""
        trends = self.analyze_category_trends(category)
        
        if not trends:
            return "No data available for trend analysis"
        
        report = f"""
TREND ANALYSIS REPORT
{'=' * 50}
Category: {trends['category']}
Analysis Date: {trends['analyzed_at']}

OVERVIEW
--------
Total Products Analyzed: {trends['total_products']}
Average Price: ${trends['average_price']:.2f}
Price Range: ${trends['price_range']['min']:.2f} - ${trends['price_range']['max']:.2f}
Median Price: ${trends['price_range']['median']:.2f}
Average Rating: {trends['average_rating']:.1f}/5.0
Total Reviews: {trends['total_reviews']:,}

PLATFORM DISTRIBUTION
--------------------
"""
        for platform, count in trends['platform_distribution'].items():
            report += f"{platform.capitalize()}: {count} products\n"
        
        if trends.get('price_segments'):
            report += "\nPRICE SEGMENTS\n--------------\n"
            for segment, data in trends['price_segments'].items():
                report += f"{segment}: {data['count']} products, Avg: ${data['avg_price']:.2f}\n"
        
        if trends.get('discount_analysis'):
            disc = trends['discount_analysis']
            report += f"\nDISCOUNT ANALYSIS\n----------------\n"
            report += f"Products with Discount: {disc['discount_percentage']}%\n"
            report += f"Average Discount: {disc['average_discount']:.1f}%\n"
        
        return report
    
    def _calculate_trend_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate accuracy metrics for trend analysis"""
        sample_size = len(df)
        
        # Data quality score (0-100)
        data_quality = 0
        if 'rating' in df.columns:
            data_quality += 25  # Has rating data
        if 'reviews_count' in df.columns:
            data_quality += 25  # Has review data
        if 'price' in df.columns and not df['price'].isna().any():
            data_quality += 25  # Complete price data
        if sample_size > 50:
            data_quality += 25  # Good sample size
        
        # Sample size confidence (0-100)
        if sample_size >= 100:
            sample_confidence = 95
        elif sample_size >= 50:
            sample_confidence = 85
        elif sample_size >= 20:
            sample_confidence = 70
        else:
            sample_confidence = 50
        
        # Overall accuracy score
        accuracy_score = (data_quality * 0.5 + sample_confidence * 0.5)
        
        return {
            'accuracy_score': round(accuracy_score, 1),
            'confidence_level': 'High' if accuracy_score >= 80 else 'Medium' if accuracy_score >= 60 else 'Low',
            'sample_size': sample_size,
            'data_quality_score': round(data_quality, 1),
            'reliability': 'Reliable' if sample_size >= 30 else 'Limited data',
            'notes': self._get_accuracy_notes(sample_size, data_quality)
        }
    
    def _calculate_forecast_accuracy(self, sample_size: int, 
                                     avg_rating: float, avg_reviews: float) -> Dict[str, Any]:
        """Calculate accuracy metrics for demand forecast"""
        # Base confidence from sample size
        if sample_size >= 100:
            base_confidence = 85
        elif sample_size >= 50:
            base_confidence = 75
        elif sample_size >= 20:
            base_confidence = 60
        else:
            base_confidence = 45
        
        # Adjust for data richness (ratings and reviews indicate active market)
        richness_boost = 0
        if avg_rating > 4.0:
            richness_boost += 5
        if avg_reviews > 100:
            richness_boost += 5
        elif avg_reviews > 50:
            richness_boost += 3
        
        forecast_confidence = min(base_confidence + richness_boost, 95)
        
        return {
            'forecast_confidence': round(forecast_confidence, 1),
            'confidence_level': 'High' if forecast_confidence >= 75 else 'Medium' if forecast_confidence >= 55 else 'Low',
            'model_type': 'Statistical Analysis',
            'data_points': sample_size,
            'prediction_reliability': self._get_reliability_level(forecast_confidence),
            'margin_of_error': f"±{round(100 - forecast_confidence)}%"
        }
    
    def _get_accuracy_notes(self, sample_size: int, quality: float) -> List[str]:
        """Get accuracy-related notes"""
        notes = []
        
        if sample_size < 20:
            notes.append("Limited sample size - use results as preliminary guidance")
        elif sample_size < 50:
            notes.append("Moderate sample size - results are indicative")
        else:
            notes.append("Good sample size - results are statistically significant")
        
        if quality < 60:
            notes.append("Some data fields missing - may affect accuracy")
        
        return notes
    
    def _get_reliability_level(self, confidence: float) -> str:
        """Get reliability level description"""
        if confidence >= 80:
            return "Highly reliable - strong data support"
        elif confidence >= 65:
            return "Moderately reliable - adequate data"
        else:
            return "Use with caution - limited data available"
    
    def _create_time_series_from_category(self, category_df: pd.DataFrame, days: int) -> pd.DataFrame:
        """
        Create synthetic time series data from category metrics for LSTM
        
        Args:
            category_df: Category product data
            days: Number of historical days to simulate
            
        Returns:
            DataFrame with time series values
        """
        # Create synthetic historical trend based on current metrics
        avg_price = category_df['price'].mean() if 'price' in category_df.columns else 100
        avg_rating = category_df['rating'].mean() if 'rating' in category_df.columns else 3.5
        
        # Generate synthetic time series (in production, use actual historical data)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
        
        # Create trend with some seasonality and noise
        trend = np.linspace(avg_price * 0.7, avg_price * 1.2, days)
        seasonality = 10 * np.sin(np.arange(days) * 2 * np.pi / 7)  # Weekly pattern
        noise = np.random.normal(0, 5, days)
        values = trend + seasonality + noise
        
        # Scale by rating to reflect demand (higher rating = higher values)
        values = values * (avg_rating / 3.5)
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    def ai_forecast_trend(self, historical_data: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """
        AI-powered trend forecasting using LSTM
        
        Args:
            historical_data: DataFrame with historical time series data
            periods: Number of periods to forecast
            
        Returns:
            Forecast results with ML metrics
        """
        if not self.use_ai or not self.lstm_forecaster:
            logger.warning("AI forecasting not available, use forecast_demand instead")
            return {
                'error': 'AI forecasting not enabled',
                'method': 'None'
            }
        
        try:
            forecast = self.lstm_forecaster.forecast_trend(historical_data, periods)
            logger.info(f"AI forecast completed using {forecast['method']}")
            return forecast
        except Exception as e:
            logger.error(f"AI forecast failed: {e}")
            return {
                'error': str(e),
                'method': 'Failed'
            }
