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
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.local = threading.local()

    def _get_connection(self):
        if not hasattr(self.local, 'conn') or self.local.conn is None:
            self.local.conn = sqlite3.connect(self.db_path)
            self.local.conn.execute("PRAGMA foreign_keys = ON")
            self._initialize_database()
        return self.local.conn

    def _get_cursor(self):
        conn = self._get_connection()
        return conn.cursor()

    def _initialize_database(self):
        cursor = self._get_cursor()
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
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            inventory_id INTEGER PRIMARY KEY,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (product_id) REFERENCES products (product_id)
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY,
            transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            customer_id INTEGER
        )
        ''')
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
        logger.info("Generating synthetic data")
        conn = self._get_connection()
        cursor = self._get_cursor()
        logger.info(f"Generating {num_products} products")
        for i in range(1, num_products + 1):
            category = random.choice(PRODUCT_CATEGORIES)
            supplier = random.choice(SUPPLIERS)
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
            quantity = random.randint(10, 200)
            cursor.execute('''
            INSERT INTO inventory (product_id, quantity)
            VALUES (?, ?)
            ''', (i, quantity))
        logger.info(f"Generating {num_customers} customers")
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
        conn.commit()
        cursor.execute("SELECT product_id, name FROM products")
        all_products = {row[0]: row[1] for row in cursor.fetchall()}

        affinity_groups = [
            [p_id for p_id, name in all_products.items()
             if "Smartphone" in name
             or "Headphones" in name
             or "Smartwatch" in name],
            [p_id for p_id, name in all_products.items()
             if "Blender" in name
             or "Mixer" in name
             or "Toaster" in name],
            [p_id for p_id, name in all_products.items()
             if "Stapler" in name
             or "Pen Set" in name
             or "Notebook" in name],
        ]
        all_product_ids = list(all_products.keys())
        for group in affinity_groups:
            while len(group) < 3:
                rand_id = random.choice(all_product_ids)
                if rand_id not in group:
                    group.append(rand_id)
        logger.info(f"Generating {num_transactions} transactions")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        transaction_batch = []
        transaction_items_batch = []
        batch_size = 1000
        transaction_counter = 0
        item_id_counter = 0

        for i in range(1, num_transactions + 1):
            transaction_date = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )

            customer_id = random.randint(1, num_customers)

            transaction_batch.append((i, transaction_date, customer_id))
            num_items = random.choices([1, 2, 3, 4, 5], weights=[20, 30, 25, 15, 10])[0]
            if random.random() < 0.4:

                selected_group = random.choice(affinity_groups)
                selected_products = random.sample(selected_group, min(num_items, len(selected_group)))
                if len(selected_products) < num_items:
                    additional_products = []
                    while len(additional_products) < (num_items - len(selected_products)):
                        rand_id = random.choice(all_product_ids)
                        if rand_id not in selected_products and rand_id not in additional_products:
                            additional_products.append(rand_id)
                    selected_products.extend(additional_products)
            else:
                selected_products = random.sample(all_product_ids, min(num_items, len(all_product_ids)))
            product_prices = {}
            for product_id in selected_products:
                cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
                product_prices[product_id] = cursor.fetchone()[0]
            for product_id in selected_products:
                quantity = random.randint(1, 3)
                price_per_unit = product_prices[product_id]

                item_id_counter += 1
                transaction_items_batch.append((
                    item_id_counter, i, product_id, quantity, price_per_unit
                ))

            transaction_counter += 1
            if transaction_counter % batch_size == 0:
                cursor.executemany('''
                INSERT INTO transactions (transaction_id, transaction_date, customer_id)
                VALUES (?, ?, ?)
                ''', transaction_batch)

                cursor.executemany('''
                INSERT INTO transaction_items (item_id, transaction_id, product_id, quantity, price_per_unit)
                VALUES (?, ?, ?, ?, ?)
                ''', transaction_items_batch)
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
                logger.info(f"Committed {transaction_counter} transactions")
                transaction_batch = []
                transaction_items_batch = []
        if transaction_batch:
            cursor.executemany('''
            INSERT INTO transactions (transaction_id, transaction_date, customer_id)
            VALUES (?, ?, ?)
            ''', transaction_batch)

            cursor.executemany('''
            INSERT INTO transaction_items (item_id, transaction_id, product_id, quantity, price_per_unit)
            VALUES (?, ?, ?, ?, ?)
            ''', transaction_items_batch)
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
        cursor.execute('''
        UPDATE inventory SET quantity = 10 WHERE quantity < 10
        ''')

        conn.commit()
        logger.info("Synthetic data generation complete.")

    def get_product(self, product_id):
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
        cursor = self._get_cursor()
        cursor.execute("SELECT * FROM products")
        columns = [description[0] for description in cursor.description]
        products = []
        for row in cursor.fetchall():
            products.append(dict(zip(columns, row)))
        return products

    def load_inventory(self):
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
        conn = self._get_connection()
        products_df = pd.read_sql_query("SELECT * FROM products", conn)
        products_df.to_csv(os.path.join(self.data_dir, 'products.csv'), index=False)
        inventory_df = pd.read_sql_query("""
        SELECT i.*, p.name as product_name, p.category
        FROM inventory i
        JOIN products p ON i.product_id = p.product_id
        """, conn)
        inventory_df.to_csv(os.path.join(self.data_dir, 'inventory.csv'), index=False)
        transactions_df = pd.read_sql_query("""
        SELECT t.*, c.name as customer_name
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        LIMIT 10000
        """, conn)
        transactions_df.to_csv(os.path.join(self.data_dir, 'transactions.csv'), index=False)
        items_df = pd.read_sql_query("""
        SELECT ti.*, p.name as product_name, p.category
        FROM transaction_items ti
        JOIN products p ON ti.product_id = p.product_id
        LIMIT 50000
        """, conn)
        items_df.to_csv(os.path.join(self.data_dir, 'transaction_items.csv'), index=False)

        logger.info("Data exported to CSV files in the data directory.")

    def close(self):
        if hasattr(self.local, 'conn') and self.local.conn is not None:
            self.local.conn.close()
            self.local.conn = None
def test_data_manager():
    data_manager = DataManager()
    data_manager.generate_synthetic_data(num_products=70, num_customers=1000, num_transactions=100000)
    products = data_manager.load_products()
    print(f"Generated {len(products)} products")

    inventory = data_manager.load_inventory()
    print(f"Generated {len(inventory)} inventory records")

    transactions = data_manager.load_transactions()
    print(f"Retrieved {len(transactions)} transactions")

    baskets = data_manager.get_transaction_baskets()
    print(f"Retrieved {len(baskets)} transaction baskets")
    data_manager.export_to_csv()
    data_manager.close()


if __name__ == "__main__":
    test_data_manager()