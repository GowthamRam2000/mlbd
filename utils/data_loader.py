"""
Data loader and generator for the inventory management system.
This module handles loading data from files and generating synthetic data.
"""

import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import sqlite3
import logging
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for synthetic data generation
PRODUCT_CATEGORIES = [
    "Electronics", "Home Appliances", "Kitchen", "Furniture", "Office Supplies",
    "Clothing", "Sports & Outdoors", "Books", "Toys & Games", "Beauty & Personal Care"
]

SUPPLIERS = [
    "TechHub Inc.", "HomeGoods Corp.", "KitchenWares Ltd.", "FurnishCo", "OfficeSupply Corp",
    "ClothingTrends Inc.", "SportLife Corp.", "BookWorm Ltd.", "ToyJoy Inc.", "BeautyEssentials Co."
]


class DataManager:
    def __init__(self, data_dir='data', db_path='data/inventory.db'):
        self.data_dir = data_dir
        self.db_path = db_path

        # Ensure data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Thread-local storage for database connections
        self.local = threading.local()

    def _get_connection(self):
        # Create a new connection if one doesn't exist for this thread
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            self.local.conn.execute("PRAGMA foreign_keys = ON")
            # Initialize database if needed
            self._initialize_database()
        return self.local.conn

    def _get_cursor(self):
        conn = self._get_connection()
        return conn.cursor()

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        cursor = self._get_cursor()

        # Products table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            supplier TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Inventory table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            inventory_id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
        ''')

        # Transactions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY,
            transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            customer_id INTEGER
        )
        ''')

        # Transaction items table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transaction_items (
            item_id INTEGER PRIMARY KEY,
            transaction_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            price_per_unit REAL NOT NULL,
            FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id),
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
        ''')

        # Customers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        self._get_connection().commit()

    def generate_synthetic_data(self, num_products=100, num_customers=200, num_transactions=100000):
        """Generate synthetic data for the inventory system"""
        logger.info("Generating synthetic data...")
        conn = self._get_connection()
        cursor = self._get_cursor()

        # Generate products
        logger.info(f"Generating {num_products} products...")
        for i in range(1, num_products + 1):
            category = random.choice(PRODUCT_CATEGORIES)
            supplier = random.choice(SUPPLIERS)

            # Generate product name based on category
            product_descriptors = {
                "Electronics": ["Smart", "Wireless", "Digital", "HD", "Portable"],
                "Home Appliances": ["Automatic", "Energy-efficient", "Smart", "Compact", "Deluxe"],
                "Kitchen": ["Stainless Steel", "Non-stick", "Electric", "Ceramic", "Premium"],
                "Furniture": ["Wooden", "Modern", "Ergonomic", "Luxury", "Compact"],
                "Office Supplies": ["Professional", "Premium", "Durable", "Recycled", "Executive"],
                "Clothing": ["Casual", "Formal", "Athletic", "Vintage", "Designer"],
                "Sports & Outdoors": ["Professional", "Lightweight", "All-Weather", "Portable", "Durable"],
                "Books": ["Hardcover", "Bestselling", "Illustrated", "Collector's", "Limited Edition"],
                "Toys & Games": ["Interactive", "Educational", "Classic", "Electronic", "Creative"],
                "Beauty & Personal Care": ["Organic", "Hypoallergenic", "Professional", "Luxury", "Natural"]
            }

            product_items = {
                "Electronics": ["Smartphone", "Laptop", "Headphones", "Tablet", "Camera", "TV", "Smartwatch"],
                "Home Appliances": ["Vacuum", "Air Purifier", "Humidifier", "Heater", "Fan", "AC"],
                "Kitchen": ["Blender", "Mixer", "Toaster", "Oven", "Knife Set", "Cookware Set"],
                "Furniture": ["Chair", "Desk", "Table", "Sofa", "Bookshelf", "Bed", "Cabinet"],
                "Office Supplies": ["Stapler", "Pen Set", "Notebook", "Organizer", "Desk Lamp"],
                "Clothing": ["T-shirt", "Jeans", "Sweater", "Jacket", "Dress", "Shirt", "Socks"],
                "Sports & Outdoors": ["Bicycle", "Tent", "Backpack", "Ball", "Fitness Tracker"],
                "Books": ["Novel", "Cookbook", "Biography", "Self-Help Book", "Textbook"],
                "Toys & Games": ["Building Blocks", "Puzzle", "Action Figure", "Board Game", "Doll"],
                "Beauty & Personal Care": ["Shampoo", "Moisturizer", "Makeup Kit", "Perfume", "Hair Dryer"]
            }

            descriptor = random.choice(product_descriptors[category])
            item = random.choice(product_items[category])

            name = f"{descriptor} {item}"
            price = round(random.uniform(9.99, 999.99), 2)
            description = f"A {descriptor.lower()} {item.lower()} for all your {category.lower()} needs."

            cursor.execute('''
            INSERT INTO products (product_id, name, category, price, supplier, description)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (i, name, category, price, supplier, description))

            # Add inventory for this product
            quantity = random.randint(10, 200)
            cursor.execute('''
            INSERT INTO inventory (product_id, quantity)
            VALUES (?, ?)
            ''', (i, quantity))

        # Generate customers
        logger.info(f"Generating {num_customers} customers...")
        first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William",
                       "Elizabeth",
                       "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Lisa"]
        last_names = ["Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
                      "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez",
                      "Robinson"]

        for i in range(1, num_customers + 1):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            name = f"{first_name} {last_name}"
            email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 99)}@example.com"

            cursor.execute('''
            INSERT INTO customers (customer_id, name, email)
            VALUES (?, ?, ?)
            ''', (i, name, email))

        # Commit before retrieval to ensure products are in the database
        conn.commit()

        # Retrieve products for creating affinities
        cursor.execute("SELECT product_id, name FROM products")
        all_products = {row[0]: row[1] for row in cursor.fetchall()}

        # Create some product affinity groups for more realistic transaction data
        # This will help with market basket analysis later
        affinity_groups = [
            # Electronics group (smartphones, headphones, smartwatches)
            [p_id for p_id, name in all_products.items()
             if "Smartphone" in name
             or "Headphones" in name
             or "Smartwatch" in name],

            # Kitchen group (blender, mixer, toaster)
            [p_id for p_id, name in all_products.items()
             if "Blender" in name
             or "Mixer" in name
             or "Toaster" in name],

            # Office group (stapler, pen set, notebook)
            [p_id for p_id, name in all_products.items()
             if "Stapler" in name
             or "Pen Set" in name
             or "Notebook" in name],
        ]

        # Fill with random products if needed (could happen with small product counts)
        all_product_ids = list(all_products.keys())
        for group in affinity_groups:
            while len(group) < 3:
                rand_id = random.choice(all_product_ids)
                if rand_id not in group:
                    group.append(rand_id)

        # Generate transactions with batched inserts for better performance
        logger.info(f"Generating {num_transactions} transactions...")

        # Generate transactions over the past 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        # Prepare to batch inserts for performance
        transaction_batch = []
        transaction_items_batch = []
        batch_size = 1000  # Commit every 1000 transactions
        transaction_counter = 0
        item_id_counter = 0

        for i in range(1, num_transactions + 1):
            # Random date within the past 30 days
            transaction_date = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )

            customer_id = random.randint(1, num_customers)

            transaction_batch.append((i, transaction_date, customer_id))

            # Decide how many items in this transaction (1-5)
            num_items = random.choices([1, 2, 3, 4, 5], weights=[20, 30, 25, 15, 10])[0]

            # Decide whether to use affinity group or random products
            if random.random() < 0.4:  # 40% chance of using affinity group
                # Select a random affinity group
                selected_group = random.choice(affinity_groups)
                # Select random products from this group
                selected_products = random.sample(selected_group, min(num_items, len(selected_group)))

                # If we need more products, add random ones
                if len(selected_products) < num_items:
                    additional_products = []
                    while len(additional_products) < (num_items - len(selected_products)):
                        rand_id = random.choice(all_product_ids)
                        if rand_id not in selected_products and rand_id not in additional_products:
                            additional_products.append(rand_id)
                    selected_products.extend(additional_products)
            else:
                # Select random products
                selected_products = random.sample(all_product_ids, min(num_items, len(all_product_ids)))

            # Get product prices
            product_prices = {}
            for product_id in selected_products:
                cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
                product_prices[product_id] = cursor.fetchone()[0]

            # Add transaction items
            for product_id in selected_products:
                quantity = random.randint(1, 3)
                price_per_unit = product_prices[product_id]

                item_id_counter += 1
                transaction_items_batch.append((
                    item_id_counter, i, product_id, quantity, price_per_unit
                ))

            transaction_counter += 1

            # Commit in batches for better performance
            if transaction_counter % batch_size == 0:
                cursor.executemany('''
                INSERT INTO transactions (transaction_id, transaction_date, customer_id)
                VALUES (?, ?, ?)
                ''', transaction_batch)

                cursor.executemany('''
                INSERT INTO transaction_items (item_id, transaction_id, product_id, quantity, price_per_unit)
                VALUES (?, ?, ?, ?, ?)
                ''', transaction_items_batch)

                # Update inventory (less precise but much faster)
                cursor.execute('''
                UPDATE inventory 
                SET quantity = quantity - (
                    SELECT SUM(ti.quantity) 
                    FROM transaction_items ti 
                    WHERE ti.product_id = inventory.product_id
                    AND ti.item_id > ?
                )
                WHERE EXISTS (
                    SELECT 1 FROM transaction_items ti 
                    WHERE ti.product_id = inventory.product_id
                    AND ti.item_id > ?
                )
                ''', (item_id_counter - len(transaction_items_batch), item_id_counter - len(transaction_items_batch)))

                conn.commit()
                logger.info(f"Committed {transaction_counter} transactions...")

                # Clear batches
                transaction_batch = []
                transaction_items_batch = []

        # Commit any remaining transactions
        if transaction_batch:
            cursor.executemany('''
            INSERT INTO transactions (transaction_id, transaction_date, customer_id)
            VALUES (?, ?, ?)
            ''', transaction_batch)

            cursor.executemany('''
            INSERT INTO transaction_items (item_id, transaction_id, product_id, quantity, price_per_unit)
            VALUES (?, ?, ?, ?, ?)
            ''', transaction_items_batch)

            # Update inventory
            cursor.execute('''
            UPDATE inventory 
            SET quantity = quantity - (
                SELECT SUM(ti.quantity) 
                FROM transaction_items ti 
                WHERE ti.product_id = inventory.product_id
                AND ti.item_id > ?
            )
            WHERE EXISTS (
                SELECT 1 FROM transaction_items ti 
                WHERE ti.product_id = inventory.product_id
                AND ti.item_id > ?
            )
            ''', (item_id_counter - len(transaction_items_batch), item_id_counter - len(transaction_items_batch)))

            conn.commit()

        # Ensure minimum inventory levels
        cursor.execute('''
        UPDATE inventory SET quantity = 10 WHERE quantity < 10
        ''')

        conn.commit()
        logger.info("Synthetic data generation complete.")

    def get_product(self, product_id):
        """Get product details by ID"""
        cursor = self._get_cursor()
        cursor.execute("SELECT * FROM products WHERE product_id = ?", (product_id,))
        row = cursor.fetchone()
        if row:
            return {
                'product_id': row[0],
                'name': row[1],
                'category': row[2],
                'price': row[3],
                'supplier': row[4],
                'description': row[5],
                'created_at': row[6]
            }
        return None

    def load_products(self):
        """Load all products from database"""
        cursor = self._get_cursor()
        cursor.execute("SELECT * FROM products")
        columns = [description[0] for description in cursor.description]
        products = []
        for row in cursor.fetchall():
            products.append(dict(zip(columns, row)))
        return products

    def load_inventory(self):
        """Load inventory with product details"""
        cursor = self._get_cursor()
        cursor.execute("""
        SELECT i.inventory_id, i.product_id, p.name, p.category, p.supplier, i.quantity, i.last_updated
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        """)
        columns = ['inventory_id', 'product_id', 'product_name', 'category', 'supplier', 'quantity', 'last_updated']
        inventory = []
        for row in cursor.fetchall():
            inventory.append(dict(zip(columns, row)))
        return inventory

    def load_transactions(self):
        """Load transactions from the past 30 days"""
        cursor = self._get_cursor()
        cutoff_date = datetime.now() - timedelta(days=30)
        cursor.execute("""
        SELECT t.transaction_id, t.transaction_date, t.customer_id, c.name as customer_name
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        WHERE t.transaction_date >= ?
        ORDER BY t.transaction_date DESC
        LIMIT 1000
        """, (cutoff_date,))

        columns = ['transaction_id', 'transaction_date', 'customer_id', 'customer_name']
        transactions = []
        for row in cursor.fetchall():
            transactions.append(dict(zip(columns, row)))
        return transactions

    def load_transaction_items(self, transaction_id=None):
        """Load transaction items, optionally filtered by transaction_id"""
        cursor = self._get_cursor()
        if transaction_id:
            cursor.execute("""
            SELECT ti.item_id, ti.transaction_id, ti.product_id, p.name as product_name, 
                   ti.quantity, ti.price_per_unit, (ti.quantity * ti.price_per_unit) as total_price
            FROM transaction_items ti
            JOIN products p ON ti.product_id = p.product_id
            WHERE ti.transaction_id = ?
            """, (transaction_id,))
        else:
            cursor.execute("""
            SELECT ti.item_id, ti.transaction_id, ti.product_id, p.name as product_name, 
                   ti.quantity, ti.price_per_unit, (ti.quantity * ti.price_per_unit) as total_price
            FROM transaction_items ti
            JOIN products p ON ti.product_id = p.product_id
            LIMIT 1000
            """)

        columns = ['item_id', 'transaction_id', 'product_id', 'product_name', 'quantity', 'price_per_unit',
                   'total_price']
        items = []
        for row in cursor.fetchall():
            items.append(dict(zip(columns, row)))
        return items

    def get_transaction_baskets(self):
        """
        Get transaction baskets for market basket analysis.
        Returns a list of sets, where each set contains product IDs in that transaction.
        """
        cursor = self._get_cursor()
        cursor.execute("""
        SELECT transaction_id, product_id
        FROM transaction_items
        ORDER BY transaction_id
        LIMIT 100000
        """)

        baskets = {}
        for transaction_id, product_id in cursor.fetchall():
            if transaction_id not in baskets:
                baskets[transaction_id] = set()
            baskets[transaction_id].add(product_id)

        return list(baskets.values())

    def get_transaction_df(self):
        """
        Get transaction data as a pandas DataFrame.
        Returns a DataFrame with transaction_id, product_id, product_name, and quantity.
        """
        cursor = self._get_cursor()
        cursor.execute("""
        SELECT ti.transaction_id, t.transaction_date, ti.product_id, p.name as product_name, 
               p.category, ti.quantity, ti.price_per_unit, t.customer_id
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.product_id
        JOIN transactions t ON ti.transaction_id = t.transaction_id
        ORDER BY t.transaction_date
        LIMIT 50000
        """)

        columns = ['transaction_id', 'transaction_date', 'product_id', 'product_name',
                   'category', 'quantity', 'price_per_unit', 'customer_id']
        data = cursor.fetchall()

        return pd.DataFrame(data, columns=columns)

    def export_to_csv(self):
        """Export database tables to CSV files"""
        conn = self._get_connection()

        # Export products
        products_df = pd.read_sql_query("SELECT * FROM products", conn)
        products_df.to_csv(os.path.join(self.data_dir, 'products.csv'), index=False)

        # Export inventory
        inventory_df = pd.read_sql_query("""
        SELECT i.*, p.name as product_name, p.category
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        """, conn)
        inventory_df.to_csv(os.path.join(self.data_dir, 'inventory.csv'), index=False)

        # Export transactions (limited to 10,000)
        transactions_df = pd.read_sql_query("""
        SELECT t.*, c.name as customer_name
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        LIMIT 10000
        """, conn)
        transactions_df.to_csv(os.path.join(self.data_dir, 'transactions.csv'), index=False)

        # Export transaction items (limited to 50,000)
        items_df = pd.read_sql_query("""
        SELECT ti.*, p.name as product_name, p.category
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.product_id
        LIMIT 50000
        """, conn)
        items_df.to_csv(os.path.join(self.data_dir, 'transaction_items.csv'), index=False)

        logger.info("Data exported to CSV files in the data directory.")

    def close(self):
        """Close database connection"""
        if hasattr(self.local, 'conn') and self.local.conn is not None:
            self.local.conn.close()
            self.local.conn = None


# Testing function
def test_data_manager():
    data_manager = DataManager()
    data_manager.generate_synthetic_data(num_products=70, num_customers=1000, num_transactions=100000)

    # Test data retrieval
    products = data_manager.load_products()
    print(f"Generated {len(products)} products")

    inventory = data_manager.load_inventory()
    print(f"Generated {len(inventory)} inventory records")

    transactions = data_manager.load_transactions()
    print(f"Retrieved {len(transactions)} transactions")

    baskets = data_manager.get_transaction_baskets()
    print(f"Retrieved {len(baskets)} transaction baskets")

    # Export to CSV
    data_manager.export_to_csv()

    # Close connection
    data_manager.close()


if __name__ == "__main__":
    test_data_manager()