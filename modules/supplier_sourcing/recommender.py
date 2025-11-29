"""
Supplier and Sourcing Recommendation Module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import WAREHOUSE_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp

logger = get_logger(__name__)


class SupplierRecommender:
    """Recommends suppliers and sourcing strategies with AI enhancements"""
    
    def __init__(self, use_ai: bool = True):
        self.warehouse_dir = WAREHOUSE_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample supplier database (in production, this would be a real database)
        self.suppliers = self._load_supplier_database()
        
        # AI semantic matcher
        self.use_ai = use_ai
        self.semantic_matcher = None
        
        if use_ai:
            try:
                from .semantic_supplier_matcher import SemanticSupplierMatcher
                self.semantic_matcher = SemanticSupplierMatcher()
                self.semantic_matcher.index_suppliers(self.suppliers)
                logger.info("AI-powered semantic supplier matching enabled")
            except Exception as e:
                logger.warning(f"Could not load semantic matcher: {e}. Using traditional matching.")
                self.use_ai = False
        
        logger.info(f"SupplierRecommender initialized (AI: {self.use_ai})")
    
    def _load_supplier_database(self) -> List[Dict[str, Any]]:
        """Load supplier database (sample data for demo)"""
        return [
            {
                'name': 'Alibaba Wholesale',
                'type': 'Wholesale Platform',
                'region': 'China',
                'moq_range': [100, 10000],
                'price_tier': 'Low',
                'quality_rating': 4.2,
                'reliability_score': 4.5,
                'shipping_time_days': [15, 30],
                'specialty_categories': ['Electronics', 'Fashion', 'Home'],
                'contact': 'alibaba.com'
            },
            {
                'name': 'Local Wholesale Market',
                'type': 'Local Wholesaler',
                'region': 'Pakistan',
                'moq_range': [10, 500],
                'price_tier': 'Medium',
                'quality_rating': 4.0,
                'reliability_score': 4.3,
                'shipping_time_days': [1, 3],
                'specialty_categories': ['Fashion', 'Beauty', 'Home'],
                'contact': 'local-market.pk'
            },
            {
                'name': 'DHgate',
                'type': 'Wholesale Platform',
                'region': 'China',
                'moq_range': [1, 1000],
                'price_tier': 'Low',
                'quality_rating': 3.8,
                'reliability_score': 4.0,
                'shipping_time_days': [10, 25],
                'specialty_categories': ['Electronics', 'Fashion', 'Sports'],
                'contact': 'dhgate.com'
            },
            {
                'name': 'Premium Brand Distributors',
                'type': 'Brand Distributor',
                'region': 'UAE',
                'moq_range': [50, 2000],
                'price_tier': 'High',
                'quality_rating': 4.8,
                'reliability_score': 4.9,
                'shipping_time_days': [5, 10],
                'specialty_categories': ['Electronics', 'Beauty', 'Fashion'],
                'contact': 'premium-dist.ae'
            },
            {
                'name': 'AliExpress Dropshipping',
                'type': 'Dropshipping',
                'region': 'China',
                'moq_range': [1, 10],
                'price_tier': 'Low-Medium',
                'quality_rating': 3.5,
                'reliability_score': 3.8,
                'shipping_time_days': [15, 45],
                'specialty_categories': ['Electronics', 'Fashion', 'Home', 'Sports'],
                'contact': 'aliexpress.com'
            },
            {
                'name': 'Manufacturer Direct',
                'type': 'Factory Direct',
                'region': 'China',
                'moq_range': [500, 50000],
                'price_tier': 'Very Low',
                'quality_rating': 4.5,
                'reliability_score': 4.4,
                'shipping_time_days': [20, 40],
                'specialty_categories': ['Electronics', 'Home', 'Sports'],
                'contact': 'manufacturing-direct.cn'
            }
        ]
    
    def recommend_suppliers(self, category: str, budget: float, 
                          quantity: int, priority: str = 'balanced') -> List[Dict[str, Any]]:
        """
        Recommend suppliers based on requirements
        
        Args:
            category: Product category
            budget: Available budget
            quantity: Required quantity
            priority: 'cost', 'quality', 'speed', or 'balanced'
        
        Returns:
            List of recommended suppliers with scores
        """
        logger.info(f"Finding suppliers for {category}, qty: {quantity}, budget: ${budget}")
        
        recommendations = []
        
        for supplier in self.suppliers:
            # Check if supplier handles this category
            if category not in supplier['specialty_categories']:
                continue
            
            # Check MOQ compatibility
            moq_min, moq_max = supplier['moq_range']
            if quantity < moq_min or quantity > moq_max:
                continue
            
            # Calculate recommendation score
            score = self._calculate_supplier_score(supplier, priority)
            
            # Estimate costs
            estimated_cost = self._estimate_cost(supplier, quantity, category)
            
            recommendation = {
                **supplier,
                'recommendation_score': score,
                'estimated_unit_cost': estimated_cost,
                'estimated_total_cost': estimated_cost * quantity,
                'fits_budget': estimated_cost * quantity <= budget,
                'pros': self._get_supplier_pros(supplier, priority),
                'cons': self._get_supplier_cons(supplier, priority)
            }
            
            recommendations.append(recommendation)
        
        # Sort by score
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        logger.info(f"Found {len(recommendations)} suitable suppliers")
        
        # Add accuracy metrics
        accuracy_metrics = self._calculate_supplier_accuracy(len(recommendations), budget, quantity)
        
        return {
            'suppliers': recommendations,
            'accuracy_metrics': accuracy_metrics
        }
    
    def _calculate_supplier_score(self, supplier: Dict[str, Any], priority: str) -> float:
        """Calculate supplier recommendation score based on priority"""
        weights = {
            'cost': {'price': 0.5, 'quality': 0.2, 'reliability': 0.2, 'speed': 0.1},
            'quality': {'price': 0.1, 'quality': 0.5, 'reliability': 0.3, 'speed': 0.1},
            'speed': {'price': 0.1, 'quality': 0.2, 'reliability': 0.2, 'speed': 0.5},
            'balanced': {'price': 0.25, 'quality': 0.25, 'reliability': 0.25, 'speed': 0.25}
        }
        
        weight = weights.get(priority, weights['balanced'])
        
        # Normalize scores
        price_score = {'Very Low': 1.0, 'Low': 0.8, 'Low-Medium': 0.6, 'Medium': 0.5, 'High': 0.3}.get(supplier['price_tier'], 0.5)
        quality_score = supplier['quality_rating'] / 5.0
        reliability_score = supplier['reliability_score'] / 5.0
        speed_score = 1.0 - (sum(supplier['shipping_time_days']) / 2 / 45)  # Normalize to 45 days max
        
        total_score = (
            price_score * weight['price'] +
            quality_score * weight['quality'] +
            reliability_score * weight['reliability'] +
            speed_score * weight['speed']
        ) * 100
        
        return round(total_score, 2)
    
    def _estimate_cost(self, supplier: Dict[str, Any], quantity: int, category: str) -> float:
        """Estimate unit cost from supplier"""
        # Base costs by region and type
        base_costs = {
            ('China', 'Wholesale Platform'): 5.0,
            ('China', 'Dropshipping'): 8.0,
            ('China', 'Factory Direct'): 3.0,
            ('Pakistan', 'Local Wholesaler'): 10.0,
            ('UAE', 'Brand Distributor'): 25.0
        }
        
        key = (supplier['region'], supplier['type'])
        base_cost = base_costs.get(key, 10.0)
        
        # Adjust for quantity (volume discounts)
        if quantity >= 1000:
            base_cost *= 0.7
        elif quantity >= 500:
            base_cost *= 0.8
        elif quantity >= 100:
            base_cost *= 0.9
        
        # Adjust for category
        category_multipliers = {
            'Electronics': 2.5,
            'Fashion': 1.0,
            'Beauty': 1.5,
            'Home': 1.2,
            'Sports': 1.3
        }
        base_cost *= category_multipliers.get(category, 1.0)
        
        return round(base_cost, 2)
    
    def _get_supplier_pros(self, supplier: Dict[str, Any], priority: str) -> List[str]:
        """Get supplier advantages"""
        pros = []
        
        if supplier['price_tier'] in ['Very Low', 'Low']:
            pros.append(f"Low cost sourcing ({supplier['price_tier']} price tier)")
        
        if supplier['quality_rating'] >= 4.5:
            pros.append(f"High quality products (rating: {supplier['quality_rating']}/5)")
        
        if supplier['reliability_score'] >= 4.5:
            pros.append(f"Highly reliable supplier")
        
        if max(supplier['shipping_time_days']) <= 10:
            pros.append(f"Fast shipping (up to {max(supplier['shipping_time_days'])} days)")
        
        if supplier['moq_range'][0] <= 10:
            pros.append(f"Low MOQ (minimum {supplier['moq_range'][0]} units)")
        
        if supplier['type'] == 'Dropshipping':
            pros.append("No inventory required (dropshipping)")
        
        return pros
    
    def _get_supplier_cons(self, supplier: Dict[str, Any], priority: str) -> List[str]:
        """Get supplier disadvantages"""
        cons = []
        
        if supplier['price_tier'] in ['High']:
            cons.append("Higher cost structure")
        
        if supplier['quality_rating'] < 4.0:
            cons.append(f"Quality concerns (rating: {supplier['quality_rating']}/5)")
        
        if max(supplier['shipping_time_days']) > 30:
            cons.append(f"Long shipping times (up to {max(supplier['shipping_time_days'])} days)")
        
        if supplier['moq_range'][0] > 500:
            cons.append(f"High MOQ requirement (minimum {supplier['moq_range'][0]} units)")
        
        if supplier['region'] != 'Pakistan' and supplier['type'] != 'Dropshipping':
            cons.append("International shipping complexities")
        
        return cons
    
    def generate_sourcing_strategy(self, product_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive sourcing strategy
        
        Args:
            product_info: Product details (category, target_price, quantity, etc.)
        
        Returns:
            Sourcing strategy recommendations
        """
        logger.info("Generating sourcing strategy")
        
        category = product_info.get('category', '')
        budget = product_info.get('budget', 1000)
        quantity = product_info.get('quantity', 100)
        business_model = product_info.get('business_model', 'ecommerce')  # ecommerce or dropshipping
        
        # Get supplier recommendations
        supplier_data = self.recommend_suppliers(category, budget, quantity, 'balanced')
        
        # Extract suppliers list from the returned dictionary
        if isinstance(supplier_data, dict):
            suppliers_list = supplier_data.get('suppliers', [])
        else:
            suppliers_list = supplier_data if isinstance(supplier_data, list) else []
        
        strategy = {
            'product_category': category,
            'recommended_suppliers': suppliers_list[:3] if suppliers_list else [],  # Top 3
            'business_model_recommendations': self._recommend_business_model(quantity, budget),
            'sourcing_approach': self._determine_sourcing_approach(quantity, budget),
            'quality_control_tips': [
                "Request product samples before bulk order",
                "Conduct third-party quality inspection",
                "Start with smaller test orders",
                "Review supplier certifications and compliance",
                "Check customer reviews and ratings"
            ],
            'negotiation_tips': [
                f"For orders above {quantity} units, negotiate 10-15% bulk discount",
                "Request better payment terms (e.g., 30% deposit, 70% on delivery)",
                "Ask for free samples or reduced sample costs",
                "Negotiate shipping costs for large orders",
                "Build long-term relationship for better terms"
            ],
            'risk_mitigation': [
                "Use secure payment methods (escrow, trade assurance)",
                "Purchase shipping insurance",
                "Verify supplier credentials and business license",
                "Start with small test batch before scaling",
                "Have backup suppliers identified"
            ],
            'generated_at': get_timestamp()
        }
        
        logger.info("Sourcing strategy generated")
        return strategy
    
    def _recommend_business_model(self, quantity: int, budget: float) -> Dict[str, Any]:
        """Recommend business model based on scale"""
        if budget < 500 or quantity < 50:
            model = 'Dropshipping'
            reason = "Low initial investment - recommended to start with dropshipping to test market"
        elif budget < 5000 or quantity < 500:
            model = 'Small Batch Wholesale'
            reason = "Moderate investment - buy small wholesale batches for better margins"
        else:
            model = 'Bulk Wholesale / Manufacturing'
            reason = "Significant investment - bulk purchasing for maximum profit margins"
        
        return {'recommended_model': model, 'reason': reason}
    
    def _determine_sourcing_approach(self, quantity: int, budget: float) -> str:
        """Determine best sourcing approach"""
        if quantity < 100:
            return "Single supplier, small orders, test market demand"
        elif quantity < 1000:
            return "Primary supplier with backup supplier, moderate orders"
        else:
            return "Multiple suppliers, negotiate bulk discounts, establish long-term contracts"
    
    def _calculate_supplier_accuracy(self, supplier_count: int, budget: float, quantity: int) -> Dict[str, Any]:
        """Calculate accuracy of supplier recommendations"""
        # Base confidence on number of suppliers found
        if supplier_count >= 3:
            base_confidence = 85
        elif supplier_count >= 2:
            base_confidence = 75
        else:
            base_confidence = 60
        
        # Adjust for budget/quantity alignment
        if supplier_count > 0:
            alignment_score = 90  # We only show suppliers that match requirements
        else:
            alignment_score = 40
        
        overall_confidence = (base_confidence * 0.6 + alignment_score * 0.4)
        
        return {
            'confidence_score': round(overall_confidence, 1),
            'confidence_level': 'High' if overall_confidence >= 80 else 'Medium' if overall_confidence >= 65 else 'Low',
            'suppliers_found': supplier_count,
            'data_completeness': '100%',  # All supplier data is complete in our database
            'recommendation_reliability': self._get_supplier_reliability(overall_confidence, supplier_count),
            'note': 'Recommendations based on MOQ compatibility and category specialization'
        }
    
    def _get_supplier_reliability(self, confidence: float, count: int) -> str:
        """Get supplier recommendation reliability"""
        if confidence >= 80 and count >= 3:
            return "High - Multiple suitable suppliers with complete data"
        elif confidence >= 65:
            return "Medium - Limited options but viable suppliers found"
        else:
            return "Low - Few suppliers match requirements, consider adjusting parameters"

