"""
E-Commerce Business Automation Platform - Streamlit Frontend
Pure Python implementation with interactive UI
"""
import streamlit as st
import json
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Import all modules
from modules.marketplace_scraper import MarketplaceScraper, DataWarehouse
from modules.trend_forecasting import TrendAnalyzer
from modules.competitor_analysis import CompetitorAnalyzer
from modules.supplier_sourcing import SupplierRecommender
from modules.pricing_estimator import PricingCalculator
from modules.platform_recommender import PlatformRecommender
from modules.audience_recommender import AudienceProfiler
from modules.marketing_strategy import MarketingStrategyGenerator

from config import WAREHOUSE_DIR
from utils.logger import get_logger
from utils.helpers import get_timestamp, save_to_json

# Page configuration
st.set_page_config(
    page_title="E-Commerce Business Automation",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #1e293b;
    }
    .accuracy-badge-high {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .accuracy-badge-medium {
        background: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .accuracy-badge-low {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

logger = get_logger(__name__)

# Helper function to convert all prices in results
def convert_currency(results, to_currency="PKR", exchange_rate=277.78):
    """
    Convert all price values in results to target currency
    exchange_rate: How many PKR per 1 USD (default: 277.78)
    """
    if to_currency == "USD":
        return results  # Already in USD
    
    # Convert prices in each module
    try:
        # Module 2: Trends
        if 'module_2_trends' in results:
            trends = results['module_2_trends'].get('category_trends', {})
            if 'average_price' in trends:
                trends['average_price'] *= exchange_rate
            if 'price_range' in trends:
                trends['price_range'] = {
                    'min': trends['price_range'].get('min', 0) * exchange_rate,
                    'max': trends['price_range'].get('max', 0) * exchange_rate
                }
        
        # Module 3: Competitors
        if 'module_3_competitors' in results:
            comp = results['module_3_competitors']
            if 'competitive_analysis' in comp:
                analysis = comp['competitive_analysis']
                if 'market_statistics' in analysis:
                    stats = analysis['market_statistics']
                    for key in ['average_price', 'min_price', 'max_price', 'median_price']:
                        if key in stats:
                            stats[key] *= exchange_rate
        
        # Module 4: Suppliers
        if 'module_4_suppliers' in results:
            suppliers = results['module_4_suppliers'].get('recommended_suppliers', [])
            for supplier in suppliers:
                if 'estimated_unit_cost' in supplier:
                    supplier['estimated_unit_cost'] *= exchange_rate
                if 'estimated_total_cost' in supplier:
                    supplier['estimated_total_cost'] *= exchange_rate
        
        # Module 5: Pricing
        if 'module_5_pricing' in results:
            pricing = results['module_5_pricing']
            if 'product_cost' in pricing:
                pricing['product_cost'] *= exchange_rate
            
            for platform_pricing in ['amazon_pricing', 'daraz_pricing', 'shopify_pricing']:
                if platform_pricing in pricing and isinstance(pricing[platform_pricing], dict):
                    p = pricing[platform_pricing]
                    # Handle AI-predicted prices (they use 'predicted_price' key)
                    for key in ['product_cost', 'recommended_price', 'predicted_price', 'total_cost', 'net_profit', 'profit']:
                        if key in p and p[key] is not None:
                            p[key] = float(p[key]) * exchange_rate
            
            if 'platform_comparison' in pricing:
                for platform in pricing['platform_comparison']:
                    for key in ['recommended_price', 'total_cost', 'net_profit', 'profit']:
                        if key in platform and platform[key] is not None:
                            platform[key] = float(platform[key]) * exchange_rate
        
        # Module 8: Marketing
        if 'module_8_marketing' in results:
            marketing = results['module_8_marketing'].get('marketing_strategy', {})
            if 'overview' in marketing and 'total_budget' in marketing['overview']:
                marketing['overview']['total_budget'] *= exchange_rate
    
    except Exception as e:
        logger.error(f"Currency conversion error: {e}")
    
    return results

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Initialize modules (cached for performance)
@st.cache_resource
def load_modules():
    return {
        'scraper': MarketplaceScraper(),
        'warehouse': DataWarehouse(),
        'trend_analyzer': TrendAnalyzer(),
        'competitor_analyzer': CompetitorAnalyzer(),
        'supplier_recommender': SupplierRecommender(),
        'pricing_calculator': PricingCalculator(),
        'platform_recommender': PlatformRecommender(),
        'audience_profiler': AudienceProfiler(),
        'marketing_generator': MarketingStrategyGenerator()
    }

modules = load_modules()

# Helper function to display accuracy badge
def show_accuracy_badge(accuracy_metrics):
    if not accuracy_metrics or 'confidence_score' not in accuracy_metrics:
        return
    
    score = accuracy_metrics.get('confidence_score', 0)
    level = accuracy_metrics.get('confidence_level', 'Unknown')
    reliability = (
        accuracy_metrics.get('recommendation_reliability') or 
        accuracy_metrics.get('analysis_reliability') or 
        accuracy_metrics.get('reliability', '')
    )
    
    badge_class = f"accuracy-badge-{level.lower()}"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem; background: rgba(0,0,0,0.05); border-radius: 8px; margin-bottom: 1rem;">
        <span style="font-weight: 600; color: #64748b; font-size: 0.875rem;">Accuracy:</span>
        <span class="{badge_class}">{level} ({score}%)</span>
        <span style="color: #64748b; font-size: 0.8rem; flex: 1;">{reliability}</span>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.title("🚀 E-Commerce Business Automation Platform")
    st.markdown("### Comprehensive AI-Powered Business Analysis")
    st.markdown("---")
    
    # Sidebar - Input Form
    with st.sidebar:
        st.header("📝 Product Details")
        
        # Category selection with custom input option
        st.markdown("**Product Category**")
        
        predefined_categories = [
            "Electronics", "Fashion", "Beauty", "Home", "Sports", "Books",
            "Toys", "Automotive", "Health & Wellness", "Food & Beverages",
            "Pet Supplies", "Office Supplies", "Garden & Outdoor"
        ]
        
        category_option = st.radio(
            "Choose category input method:",
            ["Select from popular categories", "Enter custom category"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if category_option == "Select from popular categories":
            category = st.selectbox(
                "Select Category",
                predefined_categories,
                index=0,
                label_visibility="collapsed"
            )
        else:
            category = st.text_input(
                "Enter Custom Category",
                value="Gaming Accessories",
                placeholder="E.g., Musical Instruments, Craft Supplies, Baby Products...",
                help="Enter any product category you want to analyze",
                label_visibility="collapsed"
            )
            
            # Show suggestions
            st.caption("💡 Suggestions: Furniture, Jewelry, Art Supplies, Camping Gear, Smart Home Devices")
        
        # Product suggestions based on category
        product_examples = {
            "Electronics": "Wireless Headphones",
            "Fashion": "Men's Running Shoes",
            "Beauty": "Anti-Aging Face Cream",
            "Home": "Coffee Maker",
            "Sports": "Yoga Mat",
            "Books": "Self-Help Book",
            "Toys": "Educational Board Game",
            "Automotive": "Car Phone Holder",
            "Health & Wellness": "Fitness Tracker",
            "Food & Beverages": "Organic Protein Powder",
            "Pet Supplies": "Interactive Dog Toy",
            "Office Supplies": "Ergonomic Desk Chair",
            "Garden & Outdoor": "Solar Garden Lights"
        }
        
        product_name = st.text_input(
            "Product Name/Description",
            value=product_examples.get(category, "Specific Product Name"),
            help=f"Enter the specific product you want to analyze",
            placeholder=f"E.g., {product_examples.get(category, 'Product Name')}"
        )
        
        st.markdown("---")
        
        # Target Market (affects currency)
        target_market = st.selectbox(
            "Target Market",
            ["pakistan", "international", "asia"],
            index=0
        )
        
        # Auto-select currency based on target market
        if target_market == "pakistan":
            currency = "PKR"
            currency_symbol = "Rs"
            exchange_rate = 1  # Base currency
            st.info("🇵🇰 Currency: Pakistani Rupees (PKR)")
        else:
            currency = "USD"
            currency_symbol = "$"
            exchange_rate = 0.0036  # 1 PKR = 0.0036 USD (approximate)
            st.info("🌐 Currency: US Dollars (USD)")
        
        # Price inputs with dynamic currency
        price = st.number_input(
            f"Selling Price ({currency_symbol})",
            min_value=1.0,
            max_value=1000000.0 if currency == "PKR" else 10000.0,
            value=14000.0 if currency == "PKR" else 50.0,
            step=100.0 if currency == "PKR" else 1.0,
            help=f"Enter price in {currency}"
        )
        
        budget = st.number_input(
            f"Total Budget ({currency_symbol})",
            min_value=100.0,
            max_value=10000000.0 if currency == "PKR" else 100000.0,
            value=280000.0 if currency == "PKR" else 1000.0,
            step=1000.0 if currency == "PKR" else 100.0,
            help=f"Your total investment budget in {currency}"
        )
        
        quantity = st.number_input(
            "Initial Quantity",
            min_value=1,
            max_value=100000,
            value=100,
            step=10,
            help="Number of units to order"
        )
        
        marketing_budget = st.number_input(
            f"Marketing Budget ({currency_symbol})",
            min_value=50.0 if currency == "USD" else 14000.0,
            max_value=50000.0 if currency == "USD" else 14000000.0,
            value=500.0 if currency == "USD" else 140000.0,
            step=50.0 if currency == "USD" else 10000.0,
            help=f"Budget for marketing activities in {currency}"
        )
        
        # Show currency conversion info
        if currency == "PKR":
            st.caption(f"💱 Approx. ${price * exchange_rate:.2f} USD | ${budget * exchange_rate:.2f} USD budget")
        else:
            st.caption(f"💱 Approx. Rs{price / exchange_rate:.0f} PKR | Rs{budget / exchange_rate:.0f} PKR budget")
        
        st.markdown("---")
        analyze_button = st.button("🔍 Start Complete Analysis", type="primary")
    
    # Main content area
    if analyze_button:
        st.session_state.analysis_complete = False
        
        # Progress tracking
        st.markdown("### 📊 Analysis Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {
            'status': 'success',
            'timestamp': get_timestamp(),
            'input': {
                'category': category,
                'product_name': product_name,
                'price': price,
                'budget': budget,
                'quantity': quantity,
                'target_market': target_market,
                'marketing_budget': marketing_budget,
                'currency': currency,
                'currency_symbol': currency_symbol,
                'exchange_rate': exchange_rate
            }
        }
        
        try:
            # Module 1: Marketplace Scraper
            status_text.text("Step 1/8: 🛒 Scraping marketplace data...")
            progress_bar.progress(0.125)
            
            scraped_data = modules['scraper'].scrape_all_platforms(category, max_results_per_platform=50)
            for platform, products in scraped_data.items():
                modules['warehouse'].store_products(products, platform, category)
            
            results['module_1_scraper'] = {
                'total_products_scraped': sum(len(products) for products in scraped_data.values()),
                'platforms': list(scraped_data.keys()),
                'status': 'completed'
            }
            
            # Module 2: Trend Forecasting
            status_text.text("Step 2/8: 📈 Analyzing trends and forecasting demand...")
            progress_bar.progress(0.25)
            
            trends = modules['trend_analyzer'].analyze_category_trends(category)
            demand_forecast = modules['trend_analyzer'].forecast_demand(category, days_ahead=30)
            trending_products = modules['trend_analyzer'].identify_trending_products(top_n=10)
            
            results['module_2_trends'] = {
                'category_trends': trends,
                'demand_forecast': demand_forecast,
                'trending_products': trending_products,
                'status': 'completed'
            }
            
            # Module 3: Competitor Analysis
            status_text.text("Step 3/8: 🎯 Analyzing competitors...")
            progress_bar.progress(0.375)
            
            competitors = modules['competitor_analyzer'].identify_competitors(category, (price * 0.7, price * 1.3))
            user_product = {'category': category, 'price': price}
            comparison = modules['competitor_analyzer'].compare_with_competitors(user_product, competitors)
            
            results['module_3_competitors'] = {
                'total_competitors': len(competitors),
                'top_competitors': competitors[:5],
                'competitive_analysis': comparison,
                'status': 'completed'
            }
            
            # Module 4: Supplier Sourcing
            status_text.text("Step 4/8: 🏭 Finding suppliers...")
            progress_bar.progress(0.5)
            
            supplier_data = modules['supplier_recommender'].recommend_suppliers(category, budget, quantity, priority='balanced')
            
            # Safely extract suppliers list
            if isinstance(supplier_data, dict):
                suppliers = supplier_data.get('suppliers', [])
                accuracy_metrics = supplier_data.get('accuracy_metrics')
            else:
                suppliers = supplier_data if isinstance(supplier_data, list) else []
                accuracy_metrics = None
            
            sourcing_strategy = modules['supplier_recommender'].generate_sourcing_strategy({
                'category': category,
                'budget': budget,
                'quantity': quantity,
                'business_model': 'ecommerce'
            })
            
            results['module_4_suppliers'] = {
                'recommended_suppliers': suppliers[:3] if suppliers else [],
                'sourcing_strategy': sourcing_strategy,
                'accuracy_metrics': accuracy_metrics,
                'status': 'completed'
            }
            
            # Module 5: Pricing Calculator
            status_text.text("Step 5/8: 💰 Calculating optimal pricing...")
            progress_bar.progress(0.625)
            
            # Calculate product cost from supplier or estimate
            if suppliers and len(suppliers) > 0 and 'estimated_unit_cost' in suppliers[0]:
                product_cost = float(suppliers[0]['estimated_unit_cost'])
            else:
                # Fallback: estimate product cost as 40% of selling price
                product_cost = float(price) * 0.4
            
            logger.info(f"Product cost for pricing: {currency_symbol}{product_cost:.2f}")
            
            platform_profitability = modules['pricing_calculator'].compare_platform_profitability(product_cost, price)
            pricing_amazon = modules['pricing_calculator'].calculate_pricing(product_cost, 'amazon', category=category)
            pricing_daraz = modules['pricing_calculator'].calculate_pricing(product_cost, 'daraz', category=category)

            
            results['module_5_pricing'] = {
                'product_cost': product_cost,
                'platform_comparison': platform_profitability,
                'amazon_pricing': pricing_amazon,
                'daraz_pricing': pricing_daraz,
                'status': 'completed'
            }
            
            # Add ROI to amazon_pricing if missing
            if 'roi' not in pricing_amazon and 'total_cost' in pricing_amazon and pricing_amazon['total_cost'] > 0:
                profit = pricing_amazon.get('profit', 0)
                pricing_amazon['roi'] = round((profit / pricing_amazon['total_cost']) * 100, 2)

            
            # Module 6: Platform Recommendations
            status_text.text("Step 6/8: 🛒 Recommending platforms...")
            progress_bar.progress(0.75)
            
            product_info = {
                'category': category,
                'price': price,
                'target_market': target_market,
                'business_model': 'ecommerce'
            }
            
            platform_recommendations = modules['platform_recommender'].recommend_platforms(product_info)
            multi_platform_strategy = modules['platform_recommender'].create_multi_platform_strategy(product_info)
            
            results['module_6_platforms'] = {
                'recommended_platforms': platform_recommendations,
                'multi_platform_strategy': multi_platform_strategy,
                'status': 'completed'
            }
            
            # Module 7: Audience Profiling
            status_text.text("Step 7/8: 👥 Profiling target audience...")
            progress_bar.progress(0.875)
            
            audience_profile = modules['audience_profiler'].create_audience_profile({
                'category': category,
                'price': price,
                'features': []
            })
            
            results['module_7_audience'] = {
                'audience_profile': audience_profile,
                'status': 'completed'
            }
            
            # Module 8: Marketing Strategy
            status_text.text("Step 8/8: 📱 Generating marketing strategy...")
            progress_bar.progress(1.0)
            
            marketing_strategy = modules['marketing_generator'].generate_strategy(
                product_info,
                audience_profile,
                platform_recommendations,
                marketing_budget
            )
            
            results['module_8_marketing'] = {
                'marketing_strategy': marketing_strategy,
                'status': 'completed'
            }
            
            # Convert all prices to selected currency
            if currency == "PKR":
                logger.info("Converting all prices to PKR...")
                results = convert_currency(results, "PKR", 1/exchange_rate)  # exchange_rate is PKR to USD, so invert
            
            # Save results
            results_file = WAREHOUSE_DIR / f"analysis_results_{get_timestamp().replace(':', '-').replace(' ', '_')}.json"
            save_to_json(results, results_file)
            results['results_file'] = str(results_file)
            
            status_text.text("✅ Analysis Complete!")
            st.session_state.results = results
            st.session_state.analysis_complete = True
            
            st.success("🎉 Analysis completed successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Analysis error: {e}")
            return
    
    # Display Results
    if st.session_state.analysis_complete and st.session_state.results:
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        results = st.session_state.results
        
        # Create tabs for different modules
        tabs = st.tabs([
            "📊 Overview",
            "📈 Trends",
            "🎯 Competitors",
            "🏭 Suppliers",
            "💰 Pricing",
            "🛒 Platforms",
            "👥 Audience",
            "📱 Marketing"
        ])
        
        # Tab 1: Overview
        with tabs[0]:
            st.subheader("Analysis Overview")
            
            # Get currency info
            currency_symbol = results['input'].get('currency_symbol', '$')
            currency = results['input'].get('currency', 'USD')
            
            # Display product information
            st.markdown(f"### 📦 Product: {results['input'].get('product_name', 'N/A')}")
            st.markdown(f"**Category:** {results['input']['category']} | **Price:** {currency_symbol}{results['input']['price']:,.0f} | **Quantity:** {results['input']['quantity']} | **Currency:** {currency}")
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Products Scraped", results['module_1_scraper']['total_products_scraped'])
            with col2:
                st.metric("Competitors Found", results['module_3_competitors']['total_competitors'])
            with col3:
                if results['module_4_suppliers']['recommended_suppliers']:
                    st.metric("Suppliers Found", len(results['module_4_suppliers']['recommended_suppliers']))
            with col4:
                if results['module_6_platforms']['recommended_platforms']:
                    st.metric("Platforms Recommended", len(results['module_6_platforms']['recommended_platforms']))
            
            st.markdown("### Quick Summary")
            st.info(f"✅ All 8 modules completed successfully at {results['timestamp']}")
            
            # Currency info box
            if currency == "PKR":
                exchange_rate = results['input'].get('exchange_rate', 0.0036)
                usd_price = results['input']['price'] * exchange_rate
                st.success(f"💱 **Currency:** Pakistani Rupees (PKR) | Approx. ${usd_price:.2f} USD")
            else:
                st.success(f"💱 **Currency:** US Dollars (USD)")
        
        # Tab 2: Trends
        with tabs[1]:
            st.subheader("📈 Market Trends & Demand Forecast")
            
            if 'module_2_trends' in results:
                trends = results['module_2_trends']['category_trends']
                forecast = results['module_2_trends']['demand_forecast']
                currency_symbol = results['input'].get('currency_symbol', '$')
                
                # Show accuracy
                if trends and 'accuracy_metrics' in trends:
                    show_accuracy_badge(trends['accuracy_metrics'])
                elif forecast and 'accuracy_metrics' in forecast:
                    show_accuracy_badge(forecast['accuracy_metrics'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Price", f"{currency_symbol}{trends.get('average_price', 0):,.2f}")
                with col2:
                    st.metric("Average Rating", f"{trends.get('average_rating', 0):.1f}/5.0")
                with col3:
                    st.metric("Demand Level", forecast.get('demand_level', 'N/A'))
                
                if forecast and 'recommendations' in forecast:
                    st.markdown("### Recommendations")
                    for rec in forecast['recommendations']:
                        st.write(f"- {rec}")
        
        # Tab 3: Competitors
        with tabs[2]:
            st.subheader("🎯 Competitor Analysis")
            
            if 'module_3_competitors' in results:
                comp = results['module_3_competitors']
                analysis = comp.get('competitive_analysis', {})
                currency_symbol = results['input'].get('currency_symbol', '$')
                
                # Show accuracy
                if 'accuracy_metrics' in analysis:
                    show_accuracy_badge(analysis['accuracy_metrics'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Competitors", comp['total_competitors'])
                with col2:
                    market_stats = analysis.get('market_statistics', {})
                    st.metric("Market Avg Price", f"{currency_symbol}{market_stats.get('average_price', 0):,.2f}")
                with col3:
                    st.metric("Your Position", analysis.get('price_positioning', 'N/A'))
                
                if comp['top_competitors']:
                    st.markdown("### Top Competitors")
                    df = pd.DataFrame(comp['top_competitors'][:3])
                    st.dataframe(df[['seller_name', 'competitive_score', 'average_rating']], use_container_width=True)
        
        # Tab 4: Suppliers
        with tabs[3]:
            st.subheader("🏭 Recommended Suppliers")
            
            if 'module_4_suppliers' in results:
                suppliers_data = results['module_4_suppliers']
                suppliers = suppliers_data.get('recommended_suppliers', [])
                currency_symbol = results['input'].get('currency_symbol', '$')
                
                # Show accuracy
                if 'accuracy_metrics' in suppliers_data:
                    show_accuracy_badge(suppliers_data['accuracy_metrics'])
                
                if suppliers:
                    for i, supplier in enumerate(suppliers, 1):
                        with st.expander(f"#{i}: {supplier['name']} - Score: {supplier['recommendation_score']}/100"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Type:** {supplier['type']}")
                                st.write(f"**Region:** {supplier['region']}")
                                st.write(f"**Quality Rating:** {supplier['quality_rating']}/5")
                            with col2:
                                st.write(f"**Est. Cost:** {currency_symbol}{supplier['estimated_unit_cost']:,.2f}/unit")
                                st.write(f"**MOQ:** {supplier['moq_range'][0]:,}-{supplier['moq_range'][1]:,} units")
                                st.write(f"**Fits Budget:** {'✅ Yes' if supplier['fits_budget'] else '❌ No'}")
        
        # Tab 5: Pricing
        with tabs[4]:
            st.subheader("💰 Pricing & Profitability")
            
            if 'module_5_pricing' in results:
                pricing = results['module_5_pricing']
                amazon_pricing = pricing.get('amazon_pricing', {})
                currency_symbol = results['input'].get('currency_symbol', '$')
                
                # Show accuracy
                if 'accuracy_metrics' in amazon_pricing:
                    show_accuracy_badge(amazon_pricing['accuracy_metrics'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Product Cost", f"{currency_symbol}{pricing.get('product_cost', 0):,.2f}")
                with col2:
                    # Handle AI-predicted price or traditional recommended price
                    amazon_price = amazon_pricing.get('predicted_price') or amazon_pricing.get('recommended_price', 0)
                    st.metric("Amazon Price", f"{currency_symbol}{amazon_price:,.2f}")
                with col3:
                    # Handle both AI (margin_percent) and traditional (profit_margin) keys
                    margin = amazon_pricing.get('profit_margin') or amazon_pricing.get('margin_percent', 0)
                    st.metric("Amazon Margin", f"{margin:.1f}%")
                with col4:
                    # Handle both AI (roi_percent) and traditional (roi) keys, with fallback calculation
                    roi = amazon_pricing.get('roi') or amazon_pricing.get('roi_percent', 0)
                    if roi == 0 and 'profit' in amazon_pricing and 'total_cost' in amazon_pricing:
                        total_cost = amazon_pricing['total_cost']
                        if total_cost > 0:
                            roi = (amazon_pricing['profit'] / total_cost) * 100
                    st.metric("ROI", f"{roi:.1f}%")
                
                # Platform comparison chart
                if 'platform_comparison' in pricing:
                    st.markdown("### Platform Profitability Comparison")
                    df = pd.DataFrame(pricing['platform_comparison'])
                    
                    fig = px.bar(df, x='platform', y='profit_margin', 
                                 title='Profit Margin by Platform',
                                 labels={'profit_margin': 'Profit Margin (%)', 'platform': 'Platform'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 6: Platforms
        with tabs[5]:
            st.subheader("🛒 Platform Recommendations")
            
            if 'module_6_platforms' in results:
                platforms = results['module_6_platforms'].get('recommended_platforms', [])
                
                if platforms:
                    for i, platform in enumerate(platforms[:3], 1):
                        # Show accuracy for first platform
                        if i == 1 and 'accuracy_metrics' in platform:
                            show_accuracy_badge(platform['accuracy_metrics'])
                        
                        with st.expander(f"#{i}: {platform['platform_name']} - Score: {platform['recommendation_score']}/100"):
                            st.write(f"**Suitability:** {platform['suitability_rating']}")
                            st.write(f"**Setup Difficulty:** {platform.get('setup_difficulty', {}).get('level', 'N/A')}")
                            
                            st.markdown("**Pros:**")
                            for pro in platform.get('pros', [])[:3]:
                                st.write(f"✅ {pro}")
                            
                            st.markdown("**Cons:**")
                            for con in platform.get('cons', [])[:3]:
                                st.write(f"⚠️ {con}")
        
        # Tab 7: Audience
        with tabs[6]:
            st.subheader("👥 Target Audience Profile")
            
            if 'module_7_audience' in results:
                audience = results['module_7_audience'].get('audience_profile', {})
                demographics = audience.get('primary_demographics', {})
                personas = audience.get('buyer_personas', [])
                
                # Show accuracy
                if 'accuracy_metrics' in audience:
                    show_accuracy_badge(audience['accuracy_metrics'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Age Range", demographics.get('age_range', 'N/A'))
                    st.metric("Income Level", demographics.get('income_level', 'N/A'))
                with col2:
                    st.metric("Gender", demographics.get('gender', 'N/A'))
                
                if personas:
                    st.markdown("### Primary Buyer Persona")
                    persona = personas[0]
                    st.write(f"**Name:** {persona['name']}")
                    st.write(f"**Occupation:** {persona['occupation']}")
                    st.write(f"**Goals:** {', '.join(persona.get('goals', [])[:3])}")
        
        # Tab 8: Marketing
        with tabs[7]:
            st.subheader("📱 Marketing Strategy")
            
            if 'module_8_marketing' in results:
                marketing = results['module_8_marketing'].get('marketing_strategy', {})
                overview = marketing.get('overview', {})
                channels = marketing.get('channel_strategy', {}).get('channels', [])
                currency_symbol = results['input'].get('currency_symbol', '$')
                
                # Show accuracy
                if 'accuracy_metrics' in marketing:
                    show_accuracy_badge(marketing['accuracy_metrics'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Budget", f"{currency_symbol}{overview.get('total_budget', 0):,.0f}")
                with col2:
                    st.metric("Budget Tier", overview.get('budget_tier', 'N/A'))
                with col3:
                    st.metric("Strategy Period", overview.get('strategy_period', 'N/A'))
                
                if channels:
                    st.markdown("### Marketing Channels")
                    for channel in channels[:3]:
                        st.write(f"**{channel['channel']}** - Priority: {channel['priority']} ({channel['budget_allocation']})")
        
        # Download buttons
        st.markdown("---")
        st.markdown("### 📥 Download Reports")
        col1, col2 = st.columns(2)
        
        with col1:
            results_json = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📄 Download JSON Report",
                data=results_json,
                file_name=f"analysis_results_{get_timestamp().replace(':', '-').replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Generate DOCX file
            from utils.helpers import export_to_docx
            from pathlib import Path
            import tempfile
            
            # Create temporary DOCX file
            temp_dir = Path(tempfile.gettempdir())
            docx_path = temp_dir / f"analysis_report_{get_timestamp().replace(':', '-').replace(' ', '_')}.docx"
            
            if export_to_docx(results, docx_path):
                with open(docx_path, 'rb') as f:
                    docx_bytes = f.read()
                
                st.download_button(
                    label="📝 Download Word Document",
                    data=docx_bytes,
                    file_name=f"analysis_report_{get_timestamp().replace(':', '-').replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
            else:
                st.error("Error generating Word document")

if __name__ == "__main__":
    main()
