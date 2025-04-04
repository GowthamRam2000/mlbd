"""
Stream Counter Implementation for Inventory Management System.
This module implements streaming algorithms for counting frequent items and processing
inventory updates in real-time.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import heapq
import random
import time
import datetime
import logging
import threading
import queue

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CountMinSketch:
    """
    Implementation of the Count-Min Sketch algorithm for approximate frequency counting.
    This is a probabilistic data structure that serves as a frequency table of events in a stream of data.
    It uses hash functions to map events to frequencies, but unlike a standard hash table,
    it uses multiple hash functions to reduce the probability of collisions.
    """

    def __init__(self, width=1000, depth=5, seed=42):
        """
        Initialize the Count-Min Sketch.

        Args:
            width (int): The width of the count table (number of columns)
            depth (int): The depth of the count table (number of rows, hash functions)
            seed (int): Random seed for hash function generation
        """
        self.width = width
        self.depth = depth
        self.seed = seed
        self.count = np.zeros((depth, width), dtype=int)

        # Initialize hash functions (simple linear hash)
        random.seed(seed)
        self.hash_params = []
        for i in range(depth):
            # Create parameters for a linear hash function: (a*x + b) % width
            a = random.randint(1, 1000000)
            b = random.randint(0, 1000000)
            self.hash_params.append((a, b))

    def _hash(self, item, index):
        """
        Hash the item using the hash function at the given index.

        Args:
            item: The item to hash
            index (int): The index of the hash function to use

        Returns:
            int: The hash value
        """
        a, b = self.hash_params[index]
        # Use the item's hash value as input to our hash function
        item_hash = hash(item)
        return ((a * item_hash + b) % 1000000) % self.width

    def update(self, item, count=1):
        """
        Update the count for the given item.

        Args:
            item: The item to update
            count (int): The count to add (default: 1)
        """
        for i in range(self.depth):
            self.count[i, self._hash(item, i)] += count

    def estimate(self, item):
        """
        Estimate the count for the given item.

        Args:
            item: The item to estimate

        Returns:
            int: The estimated count
        """
        return min(self.count[i, self._hash(item, i)] for i in range(self.depth))

    def get_heavy_hitters(self, threshold=0):
        """
        Get all items with an estimated count above the threshold.
        This is an approximation since we don't store the items themselves.

        Args:
            threshold (int): The minimum count threshold

        Returns:
            list: List of (item_hash, estimated_count) tuples
        """
        # This is a simplified implementation that returns the counts at each cell
        # above the threshold, which may not directly correspond to individual items
        heavy_hitters = []
        for i in range(self.width):
            min_count = min(self.count[j, i] for j in range(self.depth))
            if min_count > threshold:
                heavy_hitters.append((i, min_count))

        return sorted(heavy_hitters, key=lambda x: -x[1])


class SpaceSaving:
    """
    Implementation of the Space-Saving algorithm for finding the most frequent items in a stream.
    This algorithm maintains a fixed-size summary of the stream, guaranteed to include the most
    frequent items with their approximate frequencies.
    """

    def __init__(self, k=100):
        """
        Initialize the Space-Saving algorithm.

        Args:
            k (int): The number of counters (maximum number of items to track)
        """
        self.k = k
        self.counters = {}  # Maps item to (count, error) pairs
        self.min_heap = []  # Heap for finding the minimum count

    def update(self, item, count=1):
        """
        Update the count for the given item.

        Args:
            item: The item to update
            count (int): The count to add (default: 1)
        """
        if item in self.counters:
            # Item already being tracked, just update its count
            old_count, error = self.counters[item]
            new_count = old_count + count
            self.counters[item] = (new_count, error)

            # Update heap
            for i, (c, it, _) in enumerate(self.min_heap):
                if it == item:
                    self.min_heap[i] = (new_count, it, error)
                    heapq.heapify(self.min_heap)
                    break

        elif len(self.counters) < self.k:
            # Not at capacity, add new item
            self.counters[item] = (count, 0)
            heapq.heappush(self.min_heap, (count, item, 0))

        else:
            # At capacity, replace item with smallest count
            min_count, min_item, min_error = heapq.heappop(self.min_heap)

            # New item gets min_count + new_count, with error = min_count
            new_count = min_count + count
            new_error = min_count

            # Remove the old item
            del self.counters[min_item]

            # Add the new item
            self.counters[item] = (new_count, new_error)
            heapq.heappush(self.min_heap, (new_count, item, new_error))

    def get_frequent_items(self):
        """
        Get all items being tracked, sorted by frequency.

        Returns:
            list: List of (item, count, error) tuples, sorted by count in descending order
        """
        return sorted([(item, count, error) for item, (count, error) in self.counters.items()],
                      key=lambda x: -x[1])


class StreamingTransactionProcessor:
    """
    Process a stream of transactions to identify frequent items and patterns.
    This class combines multiple streaming algorithms to process transaction data.
    """

    def __init__(self, space_saving_k=100, cms_width=1000, cms_depth=5):
        """
        Initialize the streaming transaction processor.

        Args:
            space_saving_k (int): Number of counters for Space-Saving algorithm
            cms_width (int): Width for Count-Min Sketch
            cms_depth (int): Depth for Count-Min Sketch
        """
        # Initialize streaming algorithms
        self.space_saving = SpaceSaving(k=space_saving_k)
        self.count_min_sketch = CountMinSketch(width=cms_width, depth=cms_depth)

        # Initialize simple counters
        self.product_counts = Counter()
        self.category_counts = Counter()
        self.recent_transactions = []
        self.max_recent = 100  # Keep only the most recent transactions

        # Windowed statistics (last hour, day, week)
        self.hourly_stats = defaultdict(Counter)
        self.daily_stats = defaultdict(Counter)
        self.current_hour = datetime.datetime.now().hour
        self.current_day = datetime.datetime.now().day

        # For real-time rate calculation
        self.transaction_times = []
        self.max_rate_history = 1000  # Maximum number of transactions to keep for rate calculation

    def process_transaction(self, transaction):
        """
        Process a single transaction.

        Args:
            transaction (dict): Dictionary with transaction info including:
                - transaction_id: Unique transaction ID
                - items: List of dictionaries with product_id, quantity, etc.
                - timestamp: Transaction timestamp
        """
        # Extract transaction data
        transaction_id = transaction.get('transaction_id')
        items = transaction.get('items', [])
        timestamp = transaction.get('timestamp', datetime.datetime.now())

        # Update recent transactions list
        self.recent_transactions.append(transaction)
        if len(self.recent_transactions) > self.max_recent:
            self.recent_transactions.pop(0)

        # Update transaction times for rate calculation
        self.transaction_times.append(timestamp)
        if len(self.transaction_times) > self.max_rate_history:
            self.transaction_times.pop(0)

        # Process each item in the transaction
        for item in items:
            product_id = item.get('product_id')
            quantity = item.get('quantity', 1)
            category = item.get('category', 'Unknown')

            # Update streaming algorithms
            self.space_saving.update(product_id, quantity)
            self.count_min_sketch.update(product_id, quantity)

            # Update simple counters
            self.product_counts[product_id] += quantity
            self.category_counts[category] += quantity

            # Update windowed statistics
            hour_key = timestamp.strftime('%Y-%m-%d %H:00')
            day_key = timestamp.strftime('%Y-%m-%d')

            self.hourly_stats[hour_key][product_id] += quantity
            self.daily_stats[day_key][product_id] += quantity

            # Check if we need to cleanup old window data
            current_hour = datetime.datetime.now().hour
            current_day = datetime.datetime.now().day

            if current_hour != self.current_hour:
                self._cleanup_hourly_stats()
                self.current_hour = current_hour

            if current_day != self.current_day:
                self._cleanup_daily_stats()
                self.current_day = current_day

    def _cleanup_hourly_stats(self):
        """Clean up hourly stats older than 24 hours"""
        cutoff = (datetime.datetime.now() - datetime.timedelta(hours=24)).strftime('%Y-%m-%d %H:00')
        keys_to_remove = [key for key in self.hourly_stats.keys() if key < cutoff]
        for key in keys_to_remove:
            del self.hourly_stats[key]

    def _cleanup_daily_stats(self):
        """Clean up daily stats older than 30 days"""
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        keys_to_remove = [key for key in self.daily_stats.keys() if key < cutoff]
        for key in keys_to_remove:
            del self.daily_stats[key]

    def get_transaction_rate(self, minutes=5):
        """
        Calculate the transaction rate over the last X minutes.

        Args:
            minutes (int): Number of minutes to consider

        Returns:
            float: Transactions per minute
        """
        if not self.transaction_times:
            return 0

        cutoff = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
        recent_transactions = [t for t in self.transaction_times if t >= cutoff]

        return len(recent_transactions) / minutes

    def get_frequent_products(self, top_n=10, algorithm='space_saving'):
        """
        Get the most frequent products.

        Args:
            top_n (int): Number of top products to return
            algorithm (str): Which algorithm to use ('space_saving', 'count_min_sketch', or 'counter')

        Returns:
            list: List of (product_id, count, [error]) tuples
        """
        if algorithm == 'space_saving':
            frequent_items = self.space_saving.get_frequent_items()
            return frequent_items[:top_n]

        elif algorithm == 'count_min_sketch':
            # This is a simplified implementation
            heavy_hitters = self.count_min_sketch.get_heavy_hitters()
            return heavy_hitters[:top_n]

        else:  # Default to simple counter
            return self.product_counts.most_common(top_n)

    def get_category_stats(self, top_n=None):
        """
        Get sales statistics by category.

        Args:
            top_n (int): Number of top categories to return

        Returns:
            list: List of (category, count) tuples
        """
        if top_n:
            return self.category_counts.most_common(top_n)
        return self.category_counts.most_common()

    def get_hourly_stats(self, hours=24, top_n=10):
        """
        Get hourly sales statistics for the last X hours.

        Args:
            hours (int): Number of hours to include
            top_n (int): Number of top products per hour

        Returns:
            dict: Dictionary mapping hour to list of (product_id, count) tuples
        """
        result = {}

        # Generate hour keys for the last X hours
        now = datetime.datetime.now()
        hour_keys = [(now - datetime.timedelta(hours=i)).strftime('%Y-%m-%d %H:00') for i in range(hours)]

        for hour_key in hour_keys:
            if hour_key in self.hourly_stats:
                result[hour_key] = self.hourly_stats[hour_key].most_common(top_n)
            else:
                result[hour_key] = []

        return result

    def get_daily_stats(self, days=7, top_n=10):
        """
        Get daily sales statistics for the last X days.

        Args:
            days (int): Number of days to include
            top_n (int): Number of top products per day

        Returns:
            dict: Dictionary mapping day to list of (product_id, count) tuples
        """
        result = {}

        # Generate day keys for the last X days
        now = datetime.datetime.now()
        day_keys = [(now - datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]

        for day_key in day_keys:
            if day_key in self.daily_stats:
                result[day_key] = self.daily_stats[day_key].most_common(top_n)
            else:
                result[day_key] = []

        return result


class TransactionSimulator:
    """
    Simulate a stream of transactions for testing the streaming algorithms.
    This class generates synthetic transaction data based on historical patterns.
    """

    def __init__(self, data_manager, products=None, rate_mean=1.0, rate_std=0.5, batch_size=1):
        """
        Initialize the transaction simulator.

        Args:
            data_manager: Data manager instance to query product info
            products (list): List of product dictionaries (optional)
            rate_mean (float): Mean transaction rate (transactions per second)
            rate_std (float): Standard deviation of transaction rate
            batch_size (int): Number of transactions to generate in each batch
        """
        self.data_manager = data_manager
        self.products = products
        self.rate_mean = rate_mean
        self.rate_std = rate_std
        self.batch_size = batch_size
        self.running = False
        self.thread = None
        self.callback = None
        self.transaction_queue = queue.Queue(maxsize=1000)

        # Load products if not provided
        if not self.products:
            try:
                self.products = self.data_manager.load_products()
                logger.info(f"Loaded {len(self.products)} products for simulation")
            except Exception as e:
                logger.error(f"Error loading products: {e}")
                self.products = []

        # Create product distribution for simulation
        self._create_product_distribution()

    def _create_product_distribution(self):
        """Create a weighted distribution of products based on likely purchase patterns"""
        if not self.products:
            logger.warning("No products available for simulation")
            return

        # Get product IDs
        self.product_ids = [p['product_id'] for p in self.products]

        # Create category mapping
        self.categories = {}
        for product in self.products:
            category = product.get('category', 'Unknown')
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(product['product_id'])

        # Create product weights (probabilities)
        # For simplicity, use a power-law distribution (few popular items, many less popular)
        n_products = len(self.product_ids)
        self.product_weights = [1.0 / (i + 1) for i in range(n_products)]

        # Normalize weights
        weight_sum = sum(self.product_weights)
        self.product_weights = [w / weight_sum for w in self.product_weights]

        # Create category weights (for choosing which category to buy from)
        self.category_weights = {}
        for category, products in self.categories.items():
            self.category_weights[category] = random.uniform(0.5, 1.5)

        # Normalize category weights
        weight_sum = sum(self.category_weights.values())
        for category in self.category_weights:
            self.category_weights[category] /= weight_sum

        # Create common product combinations/bundles
        self.bundles = []
        categories = list(self.categories.keys())

        # Create 3-5 bundles
        n_bundles = random.randint(3, 5)
        for _ in range(n_bundles):
            # Select 2-4 products, potentially from the same category
            category = random.choice(categories)
            category_products = self.categories[category]

            if len(category_products) < 2:
                continue

            bundle_size = min(random.randint(2, 4), len(category_products))
            bundle = random.sample(category_products, bundle_size)

            # Add bundle with a probability
            self.bundles.append({
                'products': bundle,
                'probability': random.uniform(0.1, 0.3)  # Probability of choosing this bundle
            })

        logger.info(f"Created {len(self.bundles)} product bundles for simulation")

    def _generate_transaction(self, timestamp=None):
        """
        Generate a single synthetic transaction.

        Args:
            timestamp (datetime): Transaction timestamp (default: current time)

        Returns:
            dict: Transaction dictionary
        """
        if not self.products:
            return None

        if timestamp is None:
            timestamp = datetime.datetime.now()

        # Determine number of items in transaction (1-5, weighted towards fewer items)
        n_items = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]

        # Decide whether to use a bundle
        use_bundle = random.random() < 0.3 and self.bundles

        items = []
        if use_bundle:
            # Select a bundle based on probabilities
            weights = [bundle['probability'] for bundle in self.bundles]
            selected_bundle = random.choices(self.bundles, weights=weights)[0]

            # Add bundle products
            bundle_products = selected_bundle['products']
            for product_id in bundle_products:
                quantity = random.randint(1, 3)

                # Get product info
                product_info = next((p for p in self.products if p['product_id'] == product_id), None)
                if not product_info:
                    continue

                items.append({
                    'product_id': product_id,
                    'product_name': product_info.get('name', f'Product {product_id}'),
                    'category': product_info.get('category', 'Unknown'),
                    'quantity': quantity,
                    'price': product_info.get('price', 10.0),
                    'total': quantity * product_info.get('price', 10.0)
                })

            # If bundle has fewer than n_items, add more random products
            remaining_items = n_items - len(bundle_products)
            if remaining_items > 0:
                # Make sure we don't duplicate products
                available_products = [p for p in self.product_ids if p not in bundle_products]
                if available_products:
                    additional_products = random.sample(available_products,
                                                        min(remaining_items, len(available_products)))
                    for product_id in additional_products:
                        # Add like above
                        quantity = random.randint(1, 3)
                        product_info = next((p for p in self.products if p['product_id'] == product_id), None)
                        if not product_info:
                            continue

                        items.append({
                            'product_id': product_id,
                            'product_name': product_info.get('name', f'Product {product_id}'),
                            'category': product_info.get('category', 'Unknown'),
                            'quantity': quantity,
                            'price': product_info.get('price', 10.0),
                            'total': quantity * product_info.get('price', 10.0)
                        })
        else:
            # Select random products based on product weights
            selected_product_indices = random.choices(range(len(self.product_ids)), weights=self.product_weights,
                                                      k=n_items)
            selected_products = [self.product_ids[i] for i in selected_product_indices]

            # Make sure we don't have duplicates
            selected_products = list(set(selected_products))

            for product_id in selected_products:
                quantity = random.randint(1, 3)

                # Get product info
                product_info = next((p for p in self.products if p['product_id'] == product_id), None)
                if not product_info:
                    continue

                items.append({
                    'product_id': product_id,
                    'product_name': product_info.get('name', f'Product {product_id}'),
                    'category': product_info.get('category', 'Unknown'),
                    'quantity': quantity,
                    'price': product_info.get('price', 10.0),
                    'total': quantity * product_info.get('price', 10.0)
                })

        # Generate customer ID (1-100)
        customer_id = random.randint(1, 100)

        # Calculate transaction total
        transaction_total = sum(item['total'] for item in items)

        # Create transaction
        transaction_id = int(timestamp.timestamp() * 1000)  # Use timestamp as ID
        transaction = {
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'timestamp': timestamp,
            'items': items,
            'total': transaction_total,
            'n_items': len(items)
        }

        return transaction

    def start(self, callback=None):
        """
        Start the transaction simulator.

        Args:
            callback (function): Callback function to call with each transaction
        """
        if self.running:
            logger.warning("Transaction simulator already running")
            return

        self.running = True
        self.callback = callback

        # Start the simulator thread
        self.thread = threading.Thread(target=self._simulate_transactions)
        self.thread.daemon = True
        self.thread.start()

        logger.info("Transaction simulator started")

    def stop(self):
        """Stop the transaction simulator"""
        if not self.running:
            logger.warning("Transaction simulator not running")
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

        logger.info("Transaction simulator stopped")

    def _simulate_transactions(self):
        """Main simulation loop"""
        while self.running:
            # Generate a random delay based on transaction rate
            delay = random.normalvariate(1.0 / self.rate_mean, self.rate_std / self.rate_mean)
            delay = max(0.1, delay)  # Ensure positive delay

            # Wait
            time.sleep(delay)

            # Generate a batch of transactions
            transactions = []
            for _ in range(self.batch_size):
                transaction = self._generate_transaction()
                if transaction:
                    transactions.append(transaction)

                    # Add to queue
                    try:
                        self.transaction_queue.put(transaction, block=False)
                    except queue.Full:
                        # Queue is full, discard oldest transaction
                        try:
                            self.transaction_queue.get(block=False)
                            self.transaction_queue.put(transaction, block=False)
                        except:
                            pass

            # Call callback if provided
            if self.callback and transactions:
                try:
                    self.callback(transactions)
                except Exception as e:
                    logger.error(f"Error in transaction callback: {e}")

    def get_transactions(self, max_transactions=None):
        """
        Get transactions from the queue.

        Args:
            max_transactions (int): Maximum number of transactions to return

        Returns:
            list: List of transaction dictionaries
        """
        transactions = []
        count = 0

        while not self.transaction_queue.empty():
            if max_transactions is not None and count >= max_transactions:
                break

            try:
                transaction = self.transaction_queue.get(block=False)
                transactions.append(transaction)
                count += 1
            except queue.Empty:
                break

        return transactions

    def generate_batch(self, n_transactions, time_range=None):
        """
        Generate a batch of transactions.

        Args:
            n_transactions (int): Number of transactions to generate
            time_range (tuple): (start_time, end_time) tuple for transaction timestamps

        Returns:
            list: List of transaction dictionaries
        """
        if not self.products:
            return []

        transactions = []

        # Set up time range
        if time_range:
            start_time, end_time = time_range
            time_span = (end_time - start_time).total_seconds()
        else:
            # Default to last hour
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=1)
            time_span = 3600  # seconds

        for _ in range(n_transactions):
            # Random timestamp within range
            rand_seconds = random.uniform(0, time_span)
            timestamp = start_time + datetime.timedelta(seconds=rand_seconds)

            transaction = self._generate_transaction(timestamp)
            if transaction:
                transactions.append(transaction)

        # Sort by timestamp
        transactions.sort(key=lambda x: x['timestamp'])

        return transactions


# Testing function
def test_streaming_processor():
    # Create a Stream Processor
    processor = StreamingTransactionProcessor(space_saving_k=20, cms_width=500, cms_depth=5)

    # Create random products
    products = []
    for i in range(1, 51):
        products.append({
            'product_id': i,
            'name': f'Product {i}',
            'category': f'Category {(i - 1) // 10 + 1}',
            'price': random.uniform(10, 100)
        })

    # Create a transaction simulator
    simulator = TransactionSimulator(None, products, rate_mean=10, batch_size=5)

    # Generate a batch of transactions
    start_time = datetime.datetime.now() - datetime.timedelta(hours=12)
    end_time = datetime.datetime.now()
    transactions = simulator.generate_batch(200, (start_time, end_time))

    print(f"Generated {len(transactions)} test transactions")

    # Process transactions
    for transaction in transactions:
        processor.process_transaction(transaction)

    # Print results
    print("\nMost frequent products (Space-Saving):")
    for product_id, count, error in processor.get_frequent_products(top_n=5, algorithm='space_saving'):
        product = next((p for p in products if p['product_id'] == product_id), None)
        if product:
            print(f"{product['name']} - Count: {count}, Error bound: {error}")

    print("\nCategory statistics:")
    for category, count in processor.get_category_stats(top_n=5):
        print(f"{category} - Count: {count}")

    print("\nTransaction rate: {:.2f} transactions per minute".format(processor.get_transaction_rate()))

    print("\nStreaming processor test completed")


if __name__ == "__main__":
    test_streaming_processor()