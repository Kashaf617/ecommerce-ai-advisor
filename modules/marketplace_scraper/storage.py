"""
Data Warehouse - Central storage for all scraped marketplace data
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import WAREHOUSE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger
from utils.helpers import save_to_csv, load_from_csv, get_timestamp

logger = get_logger(__name__)


class DataWarehouse:
    """Manages the central data warehouse for marketplace data"""
    
    def __init__(self):
        self.warehouse_dir = WAREHOUSE_DIR
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        
        # Ensure directories exist
        for directory in [self.warehouse_dir, self.raw_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data Warehouse initialized")
    
    def store_products(self, products: List[Dict[str, Any]], platform: str, category: str = 'general') -> bool:
        """
        Store scraped products in the data warehouse
        
        Args:
            products: List of product dictionaries
            platform: Platform name (amazon, daraz, etc.)
            category: Product category
        
        Returns:
            True if successful, False otherwise
        """
        if not products:
            logger.warning("No products to store")
            return False
        
        try:
            # Store in raw data
            raw_file = self.raw_dir / f"{platform}_{category}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df = pd.DataFrame(products)
            df.to_csv(raw_file, index=False)
            logger.info(f"Stored {len(products)} products to {raw_file}")
            
            # Store/update in warehouse (consolidated file)
            warehouse_file = self.warehouse_dir / f"{platform}_products.csv"
            
            if warehouse_file.exists():
                # Append to existing data
                existing_df = pd.read_csv(warehouse_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                
                # Remove duplicates based on product_id and platform
                combined_df = combined_df.drop_duplicates(
                    subset=['platform', 'product_id'],
                    keep='last'
                )
                combined_df.to_csv(warehouse_file, index=False)
                logger.info(f"Updated warehouse file: {warehouse_file}")
            else:
                # Create new warehouse file
                df.to_csv(warehouse_file, index=False)
                logger.info(f"Created warehouse file: {warehouse_file}")
            
            # Create consolidated all-platforms file
            self._consolidate_all_platforms()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing products: {e}")
            return False
    
    def _consolidate_all_platforms(self):
        """Consolidate data from all platforms into a single file"""
        try:
            all_files = list(self.warehouse_dir.glob("*_products.csv"))
            if not all_files:
                return
            
            dfs = []
            for file in all_files:
                df = pd.read_csv(file)
                dfs.append(df)
            
            consolidated_df = pd.concat(dfs, ignore_index=True)
            consolidated_file = self.warehouse_dir / "all_platforms_products.csv"
            consolidated_df.to_csv(consolidated_file, index=False)
            logger.info(f"Consolidated {len(consolidated_df)} products from all platforms")
            
        except Exception as e:
            logger.error(f"Error consolidating platforms: {e}")
    
    def get_products(self, platform: Optional[str] = None, category: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve products from the warehouse
        
        Args:
            platform: Filter by platform (optional)
            category: Filter by category (optional)
        
        Returns:
            DataFrame containing products
        """
        try:
            if platform:
                warehouse_file = self.warehouse_dir / f"{platform}_products.csv"
            else:
                warehouse_file = self.warehouse_dir / "all_platforms_products.csv"
            
            if not warehouse_file.exists():
                logger.warning(f"Warehouse file not found: {warehouse_file}")
                return pd.DataFrame()
            
            df = pd.read_csv(warehouse_file)
            
            # Filter by category if specified
            if category and 'category' in df.columns:
                df = df[df['category'].str.lower() == category.lower()]
            
            logger.info(f"Retrieved {len(df)} products from warehouse")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving products: {e}")
            return pd.DataFrame()
    
    def get_product_by_id(self, product_id: str, platform: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by ID and platform"""
        try:
            df = self.get_products(platform=platform)
            if df.empty:
                return None
            
            product_df = df[df['product_id'] == product_id]
            if product_df.empty:
                return None
            
            return product_df.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error getting product by ID: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get warehouse statistics"""
        try:
            stats = {
                'total_products': 0,
                'products_by_platform': {},
                'products_by_category': {},
                'last_updated': None
            }
            
            all_products_file = self.warehouse_dir / "all_platforms_products.csv"
            if all_products_file.exists():
                df = pd.read_csv(all_products_file)
                stats['total_products'] = len(df)
                
                # Products by platform
                if 'platform' in df.columns:
                    stats['products_by_platform'] = df['platform'].value_counts().to_dict()
                
                # Products by category
                if 'category' in df.columns:
                    stats['products_by_category'] = df['category'].value_counts().to_dict()
                
                # Last updated
                if 'scraped_at' in df.columns:
                    stats['last_updated'] = df['scraped_at'].max()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def clean_old_data(self, days: int = 30):
        """Remove data older than specified days"""
        try:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            
            for file in self.warehouse_dir.glob("*.csv"):
                df = pd.read_csv(file)
                
                if 'scraped_at' in df.columns:
                    df['scraped_at'] = pd.to_datetime(df['scraped_at'])
                    df = df[df['scraped_at'] >= cutoff_date]
                    df.to_csv(file, index=False)
                    logger.info(f"Cleaned old data from {file}")
            
        except Exception as e:
            logger.error(f"Error cleaning old data: {e}")
    
    def export_to_excel(self, output_file: Path):
        """Export warehouse data to Excel for analysis"""
        try:
            all_products_file = self.warehouse_dir / "all_platforms_products.csv"
            if not all_products_file.exists():
                logger.warning("No data to export")
                return False
            
            df = pd.read_csv(all_products_file)
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # All products
                df.to_excel(writer, sheet_name='All Products', index=False)
                
                # Products by platform
                for platform in df['platform'].unique():
                    platform_df = df[df['platform'] == platform]
                    platform_df.to_excel(writer, sheet_name=platform.capitalize(), index=False)
            
            logger.info(f"Exported data to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False
