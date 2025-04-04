"""
Transaction simulator for the inventory management system.
This module provides utilities for simulating transaction streams.
"""

import random
import time
import datetime
import logging
import threading
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TransactionGenerator:
    """
    Generate realistic transaction data for testing.
    This class provides more extensive simulation capabilities than the basic
    TransactionSimulator in stream_counter.py.
    """

    def __init__(self, data_manager=None, db_path=None):
        """
        Initialize the transaction generator.

        Args:
            data_manager: Optional DataManager instance
            db_path: Path to the SQLite database file
        """
        # Initialize data manager
        if data_manager:
            self.data_manager = data_manager
        else:
            self.data_manager = DataManager(db_path=db_path)

        # Load products and customers
        self.products = self.data_manager.load_products()
        self.customers = []  # Will load as needed

        # Create product distribution
        self._create_product_distribution()

        # Simulation parameters
        self.simulate_seasonal = True
        self.simulate_promotions = True
        self.simulate_out_of_stock = True

        # Seasonal factors (monthly)
        self.seasonal_factors = [
            0.8,  # January
            0.85,  # February
            0.9,  # March
            0.95,  # April
            1.0,  # May
            1.05,  # June
            1.1,  # July
            1.15,  # August
            1.1,  # September
            1.05,  # October
            1.2,  # November (holiday season)
            1.3  # December (holiday season)
        ]

        # Daily factors (day of week)
        self.daily_factors = [
            0.7,  # Monday
            0.8,  # Tuesday
            0.9,  # Wednesday
            1.0,  # Thursday
            1.2,  # Friday
            1.5,  # Saturday
            1.1  # Sunday
        ]

        # Hourly factors (24 hours)
        self.hourly_factors = [
            0.2, 0.1, 0.05, 0.05, 0.1, 0.2,  # 12am-6am
            0.4, 0.6, 0.8, 0.9, 1.0, 1.1,  # 6am-12pm
            1.2, 1.15, 1.1, 1.05, 1.1, 1.2,  # 12pm-6pm
            1.3, 1.4, 1.2, 0.8, 0.5, 0.3  # 6pm-12am
        ]

        # Active promotions (product_id -> discount_factor)
        self.active_promotions = {}

        # Out-of-stock products
        self.out_of_stock = set()

    def _create_product_distribution(self):
        """Create a realistic distribution of product popularity"""
        # Sort products by ID for deterministic results
        self.products = sorted(self.products, key=lambda p: p['product_id'])

        # Create popularity scores (power law distribution)
        n_products = len(self.products)
        popularity = [1.0 / (i + 1) ** 0.8 for i in range(n_products)]

        # Normalize
        total = sum(popularity)
        self.product_weights = [p / total for p in popularity]

        # Create category mapping
        self.categories = {}
        for product in self.products:
            category = product.get('category', 'Unknown')
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(product['product_id'])

        # Create product mapping for quick lookup
        self.product_map = {p['product_id']: p for p in self.products}

        # Create time-based affinity groups (products that sell well together at certain times)
        self.time_affinity = {
            'morning': [p['product_id'] for p in self.products if 'Breakfast' in p.get('name', '')],
            'lunch': [p['product_id'] for p in self.products if 'Lunch' in p.get('name', '')],
            'evening': [p['product_id'] for p in self.products if 'Dinner' in p.get('name', '')]
        }

        # Create regular affinity groups (products commonly bought together)
        self.affinity_groups = []

        # Add 5-10 affinity groups
        num_groups = random.randint(5, 10)

        for _ in range(num_groups):
            # Select a random category
            category = random.choice(list(self.categories.keys()))
            products = self.categories[category]

            # Skip if not enough products
            if len(products) < 2:
                continue

            # Select 2-4 products from this category
            group_size = min(random.randint(2, 4), len(products))
            group = random.sample(products, group_size)

            self.affinity_groups.append({
                'products': group,
                'probability': random.uniform(0.2, 0.6)  # Probability of using this group
            })

    def add_promotion(self, product_id, discount_factor=0.8, duration_hours=24):
        """
        Add a promotional discount for a product.

        Args:
            product_id: ID of the product to discount
            discount_factor: Price multiplier (e.g., 0.8 = 20% off)
            duration_hours: How long the promotion lasts

        Returns:
            bool: Whether the promotion was added successfully
        """
        if product_id not in self.product_map:
            logger.warning(f"Product ID {product_id} not found, can't add promotion")
            return False

        self.active_promotions[product_id] = {
            'discount_factor': discount_factor,
            'end_time': datetime.datetime.now() + datetime.timedelta(hours=duration_hours)
        }

        logger.info(f"Added promotion for {self.product_map[product_id]['name']}: "
                    f"{(1 - discount_factor) * 100:.0f}% off for {duration_hours} hours")
        return True

    def set_out_of_stock(self, product_id, is_out_of_stock=True):
        """
        Set a product as out of stock.

        Args:
            product_id: ID of the product
            is_out_of_stock: Whether the product is out of stock

        Returns:
            bool: Whether the operation was successful
        """
        if product_id not in self.product_map:
            logger.warning(f"Product ID {product_id} not found, can't set stock status")
            return False

        if is_out_of_stock:
            self.out_of_stock.add(product_id)
            logger.info(f"Set {self.product_map[product_id]['name']} as out of stock")
        else:
            self.out_of_stock.discard(product_id)
            logger.info(f"Set {self.product_map[product_id]['name']} as in stock")

        return True

    def _get_time_factor(self, timestamp):
        """
        Calculate combined time factors for the given timestamp.

        Args:
            timestamp: Datetime object

        Returns:
            float: Combined time factor
        """
        month = timestamp.month - 1  # 0-11
        day_of_week = timestamp.weekday()  # 0-6
        hour = timestamp.hour  # 0-23

        # Apply seasonal, daily, and hourly factors
        factor = 1.0

        if self.simulate_seasonal:
            factor *= self.seasonal_factors[month]

        factor *= self.daily_factors[day_of_week]
        factor *= self.hourly_factors[hour]

        return factor

    def _select_products_for_transaction(self, timestamp, n_items=None):
        """
        Select products for a transaction based on various factors.

        Args:
            timestamp: Transaction timestamp
            n_items: Number of items (None = random)

        Returns:
            list: List of selected product IDs
        """
        if n_items is None:
            # Random number of items (1-5), weighted towards fewer items
            n_items = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]

        # Filter out out-of-stock products
        available_products = [
            (i, p['product_id'], w)
            for i, (p, w) in enumerate(zip(self.products, self.product_weights))
            if p['product_id'] not in self.out_of_stock
        ]

        if not available_products:
            logger.warning("No products available for transaction")
            return []

        # Extract indices, product IDs, and weights
        indices, product_ids, weights = zip(*available_products)

        # Determine selection method
        use_affinity = random.random() < 0.4 and self.affinity_groups
        use_time_affinity = random.random() < 0.3

        selected_products = []

        # Time-based selection
        if use_time_affinity:
            hour = timestamp.hour
            if 5 <= hour < 10:  # Morning
                affinity_key = 'morning'
            elif 11 <= hour < 14:  # Lunch
                affinity_key = 'lunch'
            elif 17 <= hour < 21:  # Evening
                affinity_key = 'evening'
            else:
                affinity_key = None

            if affinity_key and self.time_affinity.get(affinity_key):
                # Select from time-based affinity group
                time_products = [p for p in self.time_affinity[affinity_key] if p not in self.out_of_stock]
                if time_products:
                    selected_products.extend(random.sample(
                        time_products,
                        min(n_items, len(time_products))
                    ))

        # Affinity-based selection
        elif use_affinity:
            # Select an affinity group based on probabilities
            group_weights = [group['probability'] for group in self.affinity_groups]
            selected_group = random.choices(self.affinity_groups, weights=group_weights)[0]

            # Filter out out-of-stock products
            group_products = [p for p in selected_group['products'] if p not in self.out_of_stock]

            if group_products:
                # Limit to the desired number of items
                selected_products.extend(random.sample(
                    group_products,
                    min(n_items, len(group_products))
                ))

        # Random selection for remaining items
        remaining = n_items - len(selected_products)
        if remaining > 0:
            additional_products = random.choices(product_ids, weights=weights, k=remaining * 2)
            # Remove duplicates and limit to remaining count
            for p in additional_products:
                if p not in selected_products and len(selected_products) < n_items:
                    selected_products.append(p)

        return selected_products

    def generate_transaction(self, timestamp=None, customer_id=None, n_items=None):
        """
        Generate a single transaction.

        Args:
            timestamp: Transaction timestamp (None = current time)
            customer_id: Customer ID (None = random)
            n_items: Number of items (None = random)

        Returns:
            dict: Transaction dictionary
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()

        if customer_id is None:
            customer_id = random.randint(1, 100)

        # Select products based on various factors
        product_ids = self._select_products_for_transaction(timestamp, n_items)

        if not product_ids:
            return None

        # Create transaction
        transaction_id = int(timestamp.timestamp() * 1000)  # Use timestamp as ID

        # Add items
        items = []
        transaction_total = 0

        for product_id in product_ids:
            # Get product info
            product = self.product_map[product_id]

            # Determine quantity (1-3)
            quantity = random.randint(1, 3)

            # Get price and apply promotions if active
            price = product['price']
            if self.simulate_promotions and product_id in self.active_promotions:
                promotion = self.active_promotions[product_id]

                # Check if promotion is still valid
                if timestamp <= promotion['end_time']:
                    price *= promotion['discount_factor']
                else:
                    # Remove expired promotion
                    del self.active_promotions[product_id]

            # Calculate item total
            item_total = price * quantity
            transaction_total += item_total

            # Add item to transaction
            items.append({
                'product_id': product_id,
                'product_name': product['name'],
                'category': product.get('category', 'Unknown'),
                'quantity': quantity,
                'price': price,
                'total': item_total,
                'promotion_applied': product_id in self.active_promotions
            })

        # Create transaction object
        transaction = {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'timestamp': timestamp,
            'items': items,
            'total': transaction_total,
            'n_items': len(items)
        }

        return transaction

    def generate_batch(self, start_time, end_time, base_rate=10, random_seed=None):
        """
        Generate a batch of transactions for a time period.

        Args:
            start_time: Start datetime
            end_time: End datetime
            base_rate: Base transactions per hour
            random_seed: Random seed for reproducibility

        Returns:
            list: List of transaction dictionaries
        """
        if random_seed is not None:
            random.seed(random_seed)

        # Calculate duration in hours
        duration_hours = (end_time - start_time).total_seconds() / 3600

        # Estimate number of transactions based on base rate and duration
        n_transactions = int(base_rate * duration_hours)

        # Generate timestamps
        timestamps = []
        current_time = start_time

        while current_time < end_time:
            # Get time factor
            time_factor = self._get_time_factor(current_time)

            # Adjust rate based on time factor
            adjusted_rate = base_rate * time_factor

            # Calculate expected transactions in this hour
            expected_txns = adjusted_rate

            # Add Poisson random number of transactions
            n_txns = np.random.poisson(expected_txns)

            # Generate n_txns random timestamps within this hour
            for _ in range(n_txns):
                seconds = random.randint(0, 3599)
                txn_time = current_time + datetime.timedelta(seconds=seconds)
                if txn_time < end_time:  # Ensure we don't exceed end_time
                    timestamps.append(txn_time)

            # Move to next hour
            current_time += datetime.timedelta(hours=1)

        # Sort timestamps
        timestamps.sort()

        # Generate transactions for each timestamp
        transactions = []
        for timestamp in timestamps:
            transaction = self.generate_transaction(timestamp)
            if transaction:
                transactions.append(transaction)

        logger.info(f"Generated {len(transactions)} transactions for period {start_time} to {end_time}")
        return transactions

    def save_transactions_to_file(self, transactions, filename):
        """
        Save transactions to a JSON file.

        Args:
            transactions: List of transaction dictionaries
            filename: Output filename

        Returns:
            str: Path to saved file
        """
        # Convert datetime objects to strings
        serializable_transactions = []
        for txn in transactions:
            txn_copy = txn.copy()
            txn_copy['timestamp'] = txn_copy['timestamp'].isoformat()
            serializable_transactions.append(txn_copy)

        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_transactions, f, indent=2)

        logger.info(f"Saved {len(transactions)} transactions to {filename}")
        return filename


# For numpy imports
try:
    import numpy as np
except ImportError:
    logger.error("NumPy not found. Install with: pip install numpy")
    np = None


def run_simulator():
    """Run the transaction simulator as a standalone script"""
    if np is None:
        print("Error: NumPy is required for this script.")
        return

    print("Inventory Transaction Simulator")
    print("===============================")

    # Create data manager
    data_manager = DataManager()

    # Check if database exists with data
    db_path = os.path.join('data', 'inventory.db')
    if not os.path.exists(db_path) or os.path.getsize(db_path) < 1000:
        print("Generating synthetic data...")
        data_manager.generate_synthetic_data(
            num_products=100,
            num_customers=200,
            num_transactions=1000
        )
        print("Synthetic data generated.")

    # Create transaction generator
    generator = TransactionGenerator(data_manager)

    # Generate batch for last 7 days
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=7)

    print(f"Generating transactions from {start_time} to {end_time}...")
    transactions = generator.generate_batch(start_time, end_time, base_rate=20)

    # Save to file
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, 'simulated_transactions.json')
    generator.save_transactions_to_file(transactions, filename)

    print(f"Generated {len(transactions)} transactions and saved to {filename}")

    # Generate some real-time transactions
    print("\nGenerating real-time transactions (Ctrl+C to stop):")
    try:
        while True:
            transaction = generator.generate_transaction()
            if transaction:
                print(f"Transaction: {transaction['transaction_id']}, "
                      f"Items: {transaction['n_items']}, "
                      f"Total: ${transaction['total']:.2f}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped transaction generation.")


if __name__ == "__main__":
    run_simulator()