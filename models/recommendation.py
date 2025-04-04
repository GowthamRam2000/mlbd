"""
Recommendation System Implementation for Inventory Management System.
This module implements PageRank-inspired algorithms for product importance and recommendation.
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductRankRecommender:
    """
    Implementation of a PageRank-inspired recommendation system for products.
    """

    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1.0e-6):
        """
        Initialize the product rank recommender.

        Args:
            damping_factor (float): Damping factor for PageRank (default: 0.85)
            max_iterations (int): Maximum number of iterations for PageRank (default: 100)
            tolerance (float): Convergence tolerance for PageRank (default: 1.0e-6)
        """
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.product_graph = None
        self.product_ranks = None
        self.product_info = None
        self.content_similarity = None

    def build_product_graph(self, transactions, product_info=None):
        """
        Build the product graph from transaction data.

        Args:
            transactions (list or DataFrame): Transaction data
            product_info (dict): Dictionary mapping product IDs to product information

        Returns:
            self: The trained instance
        """
        start_time = time.time()
        logger.info("Building product graph from transaction data...")

        # Store product info for later
        self.product_info = product_info

        # Create directed graph
        self.product_graph = nx.DiGraph()

        # Process transactions
        if isinstance(transactions, pd.DataFrame):
            # Extract transaction data from DataFrame
            transaction_groups = transactions.groupby('transaction_id')
            n_transactions = len(transaction_groups)

            for transaction_id, group in transaction_groups:
                self._process_transaction_group(group)
        else:
            # Process list of transactions
            n_transactions = len(transactions)

            for transaction in transactions:
                self._process_transaction(transaction)

        logger.info(f"Built product graph with {self.product_graph.number_of_nodes()} nodes "
                    f"and {self.product_graph.number_of_edges()} edges from {n_transactions} transactions "
                    f"in {time.time() - start_time:.2f} seconds")

        # Calculate product ranks
        self._calculate_product_ranks()

        # Build content-based similarity if product info provided
        if product_info:
            self._build_content_similarity()

        return self

    def _process_transaction(self, transaction):
        """
        Process a single transaction to update the product graph.

        Args:
            transaction (dict): Transaction dictionary with items
        """
        items = transaction.get('items', [])

        if len(items) <= 1:
            return

        # Get product IDs from items
        product_ids = []
        for item in items:
            product_id = item.get('product_id')
            if product_id:
                product_ids.append(product_id)

                # Add node if it doesn't exist
                if not self.product_graph.has_node(product_id):
                    product_name = item.get('product_name', f'Product {product_id}')
                    category = item.get('category', 'Unknown')
                    self.product_graph.add_node(product_id, name=product_name, category=category)

        # Connect products in transaction
        for i, source_id in enumerate(product_ids):
            for target_id in product_ids[i + 1:]:
                # Add edge or increment weight
                if self.product_graph.has_edge(source_id, target_id):
                    self.product_graph[source_id][target_id]['weight'] += 1
                else:
                    self.product_graph.add_edge(source_id, target_id, weight=1)

                # Add reverse edge for undirected co-occurrence
                if self.product_graph.has_edge(target_id, source_id):
                    self.product_graph[target_id][source_id]['weight'] += 1
                else:
                    self.product_graph.add_edge(target_id, source_id, weight=1)

    def _process_transaction_group(self, group):
        """
        Process a transaction group from a DataFrame.

        Args:
            group (DataFrame): Group of rows for a single transaction
        """
        if len(group) <= 1:
            return

        # Get product IDs from group
        product_ids = group['product_id'].unique()

        # Add nodes if they don't exist
        for _, row in group.iterrows():
            product_id = row['product_id']
            if not self.product_graph.has_node(product_id):
                product_name = row.get('product_name', f'Product {product_id}')
                category = row.get('category', 'Unknown')
                self.product_graph.add_node(product_id, name=product_name, category=category)

        # Connect products in transaction
        for i, source_id in enumerate(product_ids):
            for target_id in product_ids[i + 1:]:
                # Add edge or increment weight
                if self.product_graph.has_edge(source_id, target_id):
                    self.product_graph[source_id][target_id]['weight'] += 1
                else:
                    self.product_graph.add_edge(source_id, target_id, weight=1)

                # Add reverse edge for undirected co-occurrence
                if self.product_graph.has_edge(target_id, source_id):
                    self.product_graph[target_id][source_id]['weight'] += 1
                else:
                    self.product_graph.add_edge(target_id, source_id, weight=1)

    def _calculate_product_ranks(self):
        """
        Calculate PageRank for the product graph.
        """
        if not self.product_graph:
            raise ValueError("Product graph not built. Call build_product_graph() first.")

        start_time = time.time()
        logger.info("Calculating PageRank for product graph...")

        # Calculate PageRank
        self.product_ranks = nx.pagerank(
            self.product_graph,
            alpha=self.damping_factor,
            max_iter=self.max_iterations,
            tol=self.tolerance,
            weight='weight'
        )

        logger.info(f"Calculated PageRank for {len(self.product_ranks)} products "
                    f"in {time.time() - start_time:.2f} seconds")

    def _build_content_similarity(self):
        """
        Build content-based similarity matrix based on product descriptions.
        """
        if not self.product_info:
            logger.warning("No product info provided. Cannot build content similarity.")
            return

        start_time = time.time()
        logger.info("Building content-based similarity matrix...")

        # Extract product descriptions
        product_ids = []
        descriptions = []

        for product_id, info in self.product_info.items():
            description = info.get('description', '')
            name = info.get('name', '')
            category = info.get('category', '')

            # Combine available text fields
            text = f"{name} {category} {description}".strip()

            product_ids.append(product_id)
            descriptions.append(text)

        if not descriptions:
            logger.warning("No product descriptions found. Cannot build content similarity.")
            return

        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(descriptions)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Store as dictionary mapping product IDs to similar products
        self.content_similarity = {}
        for i, product_id in enumerate(product_ids):
            similar_products = {}
            for j, other_id in enumerate(product_ids):
                if i != j:  # Skip self-similarity
                    similar_products[other_id] = similarity_matrix[i, j]

            # Sort by similarity (descending)
            self.content_similarity[product_id] = dict(
                sorted(similar_products.items(), key=lambda x: x[1], reverse=True)
            )

        logger.info(f"Built content similarity matrix for {len(product_ids)} products "
                    f"in {time.time() - start_time:.2f} seconds")

    def get_product_ranks(self, top_n=None):
        """
        Get product ranks.

        Args:
            top_n (int): Number of top products to return

        Returns:
            list: List of (product_id, rank) tuples, sorted by rank in descending order
        """
        if not self.product_ranks:
            raise ValueError("Product ranks not calculated. Call build_product_graph() first.")

        # Sort by rank (descending)
        sorted_ranks = sorted(self.product_ranks.items(), key=lambda x: x[1], reverse=True)

        # Add product info if available
        result = []
        for product_id, rank in sorted_ranks[:top_n] if top_n else sorted_ranks:
            item = {'product_id': product_id, 'rank': rank}

            # Add product info if available
            if self.product_info and product_id in self.product_info:
                item['product_info'] = self.product_info[product_id]
            elif self.product_graph.has_node(product_id):
                item['product_info'] = {
                    'name': self.product_graph.nodes[product_id].get('name', f'Product {product_id}'),
                    'category': self.product_graph.nodes[product_id].get('category', 'Unknown')
                }

            result.append(item)

        return result

    def get_collaborative_recommendations(self, product_id, top_n=5):
        """
        Get collaborative filtering recommendations for a product.

        Args:
            product_id: Product ID to get recommendations for
            top_n (int): Number of recommendations to return

        Returns:
            list: List of recommended product dictionaries
        """
        if not self.product_graph:
            raise ValueError("Product graph not built. Call build_product_graph() first.")

        if not self.product_graph.has_node(product_id):
            logger.warning(f"Product ID {product_id} not found in graph.")
            return []

        # Get neighbors with weights
        neighbors = []
        for neighbor, edge_data in self.product_graph[product_id].items():
            weight = edge_data.get('weight', 0)
            rank = self.product_ranks.get(neighbor, 0)

            # Score is a combination of weight and rank
            score = weight * rank

            neighbors.append({
                'product_id': neighbor,
                'weight': weight,
                'rank': rank,
                'score': score
            })

        # Sort by score (descending)
        neighbors.sort(key=lambda x: x['score'], reverse=True)

        # Add product info if available
        result = []
        for neighbor in neighbors[:top_n]:
            neighbor_id = neighbor['product_id']

            # Add product info if available
            if self.product_info and neighbor_id in self.product_info:
                neighbor['product_info'] = self.product_info[neighbor_id]
            elif self.product_graph.has_node(neighbor_id):
                neighbor['product_info'] = {
                    'name': self.product_graph.nodes[neighbor_id].get('name', f'Product {neighbor_id}'),
                    'category': self.product_graph.nodes[neighbor_id].get('category', 'Unknown')
                }

            result.append(neighbor)

        return result

    def get_content_recommendations(self, product_id, top_n=5):
        """
        Get content-based recommendations for a product.

        Args:
            product_id: Product ID to get recommendations for
            top_n (int): Number of recommendations to return

        Returns:
            list: List of recommended product dictionaries
        """
        if not self.content_similarity:
            logger.warning("Content similarity not built. Cannot get content recommendations.")
            return []

        if product_id not in self.content_similarity:
            logger.warning(f"Product ID {product_id} not found in content similarity.")
            return []

        # Get similar products
        similar_products = list(self.content_similarity[product_id].items())

        # Convert to recommendation format
        result = []
        for neighbor_id, similarity in similar_products[:top_n]:
            recommendation = {
                'product_id': neighbor_id,
                'similarity': similarity
            }

            # Add product info if available
            if self.product_info and neighbor_id in self.product_info:
                recommendation['product_info'] = self.product_info[neighbor_id]
            elif self.product_graph and self.product_graph.has_node(neighbor_id):
                recommendation['product_info'] = {
                    'name': self.product_graph.nodes[neighbor_id].get('name', f'Product {neighbor_id}'),
                    'category': self.product_graph.nodes[neighbor_id].get('category', 'Unknown')
                }

            result.append(recommendation)

        return result

    def get_hybrid_recommendations(self, product_id, top_n=5, collaborative_weight=0.7):
        """
        Get hybrid recommendations using both collaborative and content-based filtering.

        Args:
            product_id: Product ID to get recommendations for
            top_n (int): Number of recommendations to return
            collaborative_weight (float): Weight for collaborative recommendations (0-1)

        Returns:
            list: List of recommended product dictionaries
        """
        # Get collaborative recommendations
        collaborative_recs = self.get_collaborative_recommendations(product_id, top_n=top_n * 2)

        # Get content recommendations
        content_recs = self.get_content_recommendations(product_id, top_n=top_n * 2)

        # If either is empty, return the other
        if not collaborative_recs:
            return content_recs[:top_n]
        if not content_recs:
            return collaborative_recs[:top_n]

        # Combine scores
        content_weight = 1.0 - collaborative_weight
        hybrid_scores = {}

        # Process collaborative recommendations
        for rec in collaborative_recs:
            product_id = rec['product_id']
            score = rec['score']
            hybrid_scores[product_id] = {
                'hybrid_score': collaborative_weight * score,
                'collaborative_score': score,
                'content_score': 0,
                'product_info': rec.get('product_info')
            }

        # Process content recommendations
        for rec in content_recs:
            product_id = rec['product_id']
            similarity = rec['similarity']

            if product_id in hybrid_scores:
                hybrid_scores[product_id]['content_score'] = similarity
                hybrid_scores[product_id]['hybrid_score'] += content_weight * similarity
            else:
                hybrid_scores[product_id] = {
                    'hybrid_score': content_weight * similarity,
                    'collaborative_score': 0,
                    'content_score': similarity,
                    'product_info': rec.get('product_info')
                }

        # Convert to list and sort by hybrid score
        result = []
        for product_id, scores in hybrid_scores.items():
            result.append({
                'product_id': product_id,
                'hybrid_score': scores['hybrid_score'],
                'collaborative_score': scores['collaborative_score'],
                'content_score': scores['content_score'],
                'product_info': scores['product_info']
            })

        # Sort by hybrid score (descending)
        result.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return result[:top_n]

    def get_basket_recommendations(self, basket, top_n=5):
        """
        Get recommendations based on a basket of products.

        Args:
            basket (list): List of product IDs in the basket
            top_n (int): Number of recommendations to return

        Returns:
            list: List of recommended product dictionaries
        """
        if not self.product_graph:
            raise ValueError("Product graph not built. Call build_product_graph() first.")

        if not basket:
            logger.warning("Empty basket provided. Cannot get recommendations.")
            return []

        # Get recommendations for each product in the basket
        all_recommendations = []
        for product_id in basket:
            recommendations = self.get_collaborative_recommendations(product_id, top_n=top_n * 2)
            all_recommendations.extend(recommendations)

        # Combine scores by product ID
        combined_scores = {}
        for rec in all_recommendations:
            product_id = rec['product_id']
            score = rec['score']

            if product_id in combined_scores:
                combined_scores[product_id]['score'] += score
            else:
                combined_scores[product_id] = {
                    'product_id': product_id,
                    'score': score,
                    'product_info': rec.get('product_info')
                }

        # Remove products already in the basket
        for product_id in basket:
            if product_id in combined_scores:
                del combined_scores[product_id]

        # Convert to list and sort by score
        result = list(combined_scores.values())
        result.sort(key=lambda x: x['score'], reverse=True)

        return result[:top_n]

    def get_category_recommendations(self, category, top_n=5):
        """
        Get top recommendations for a specific category.

        Args:
            category (str): Category to get recommendations for
            top_n (int): Number of recommendations to return

        Returns:
            list: List of recommended product dictionaries
        """
        if not self.product_ranks:
            raise ValueError("Product ranks not calculated. Call build_product_graph() first.")

        # Find all products in this category
        category_products = []

        # Check from product_info first
        if self.product_info:
            for product_id, info in self.product_info.items():
                if info.get('category') == category and product_id in self.product_ranks:
                    category_products.append({
                        'product_id': product_id,
                        'rank': self.product_ranks[product_id],
                        'product_info': info
                    })

        # If not found or no product_info, check graph nodes
        if not category_products and self.product_graph:
            for product_id, node_data in self.product_graph.nodes(data=True):
                if node_data.get('category') == category and product_id in self.product_ranks:
                    category_products.append({
                        'product_id': product_id,
                        'rank': self.product_ranks[product_id],
                        'product_info': {
                            'name': node_data.get('name', f'Product {product_id}'),
                            'category': category
                        }
                    })

        if not category_products:
            logger.warning(f"No products found for category: {category}")
            return []

        # Sort by rank (descending)
        category_products.sort(key=lambda x: x['rank'], reverse=True)

        return category_products[:top_n]

    def export_graph(self, filename=None, format='graphml'):
        """
        Export the product graph to a file.

        Args:
            filename (str): Filename to export to (default: product_graph.graphml)
            format (str): Export format ('graphml', 'gexf', 'pajek', 'gml')

        Returns:
            str: Path to the exported file
        """
        if not self.product_graph:
            raise ValueError("Product graph not built. Call build_product_graph() first.")

        if not filename:
            filename = f"product_graph.{format}"

        if format == 'graphml':
            nx.write_graphml(self.product_graph, filename)
        elif format == 'gexf':
            nx.write_gexf(self.product_graph, filename)
        elif format == 'pajek':
            nx.write_pajek(self.product_graph, filename)
        elif format == 'gml':
            nx.write_gml(self.product_graph, filename)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Exported product graph to {filename}")
        return filename

    def visualize_graph(self, filename=None, layout='spring'):
        """
        Create a visualization of the product graph.

        Args:
            filename (str): Filename to save visualization to (default: product_graph.png)
            layout (str): Graph layout ('spring', 'circular', 'kamada_kawai', 'random')

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        import matplotlib.pyplot as plt

        if not self.product_graph:
            raise ValueError("Product graph not built. Call build_product_graph() first.")

        # Create figure
        plt.figure(figsize=(12, 8))

        # Get layout
        if layout == 'spring':
            pos = nx.spring_layout(self.product_graph)
        elif layout == 'circular':
            pos = nx.circular_layout(self.product_graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.product_graph)
        elif layout == 'random':
            pos = nx.random_layout(self.product_graph)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Get edge weights for width
        edge_weights = [data['weight'] for _, _, data in self.product_graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1

        # Normalize weights for visualization
        edge_widths = [0.1 + 3.0 * weight / max_weight for weight in edge_weights]

        # Get node sizes based on PageRank
        if self.product_ranks:
            node_sizes = [5000 * self.product_ranks.get(node, 0) for node in self.product_graph.nodes()]
        else:
            node_sizes = [300] * len(self.product_graph.nodes())

        # Draw the graph
        nx.draw_networkx_nodes(self.product_graph, pos, node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(self.product_graph, pos, width=edge_widths, alpha=0.5, arrows=True)

        # Add labels to nodes with high PageRank
        if self.product_ranks:
            # Get top nodes by PageRank
            top_nodes = sorted(self.product_ranks.items(), key=lambda x: x[1], reverse=True)[:10]
            top_node_ids = [node_id for node_id, _ in top_nodes]

            # Create labels for top nodes
            labels = {}
            for node_id in top_node_ids:
                if self.product_info and node_id in self.product_info:
                    labels[node_id] = self.product_info[node_id].get('name', f'Product {node_id}')
                elif self.product_graph.has_node(node_id):
                    labels[node_id] = self.product_graph.nodes[node_id].get('name', f'Product {node_id}')
                else:
                    labels[node_id] = f'Product {node_id}'

            nx.draw_networkx_labels(self.product_graph, pos, labels=labels, font_size=10)

        plt.title("Product Recommendation Graph")
        plt.axis('off')

        # Save if filename provided
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            logger.info(f"Saved graph visualization to {filename}")

        return plt.gcf()


# Testing function
def test_product_rank_recommender():
    import random
    from datetime import datetime, timedelta

    # Create sample products
    products = {}
    for i in range(1, 51):
        category = f"Category {(i - 1) // 10 + 1}"
        products[i] = {
            'product_id': i,
            'name': f'Product {i}',
            'category': category,
            'description': f'This is a {random.choice(["great", "fantastic", "awesome", "useful"])} product in the {category} category.',
            'price': round(random.uniform(10, 100), 2)
        }

    # Create sample transactions
    transactions = []
    for i in range(200):
        # Random date in the last 30 days
        days_ago = random.randint(0, 30)
        transaction_date = datetime.now() - timedelta(days=days_ago)

        # Decide how many products in this transaction (1-5)
        n_items = random.randint(1, 5)

        # Select random products, but with some patterns
        if random.random() < 0.4:
            # Product affinity: Products 1, 2, and 3 often bought together
            selected_products = [1, 2, 3][:n_items]
            if n_items > 3:
                additional = random.sample(range(4, 51), n_items - 3)
                selected_products.extend(additional)
        elif random.random() < 0.7:
            # Category affinity: Products from same category
            category = random.randint(1, 5)
            category_products = [p for p in range(1, 51) if products[p]['category'] == f'Category {category}']
            selected_products = random.sample(category_products, min(n_items, len(category_products)))
        else:
            # Random selection
            selected_products = random.sample(range(1, 51), n_items)

        # Create transaction
        transaction = {
            'transaction_id': i + 1,
            'timestamp': transaction_date,
            'items': []
        }

        # Add items
        for product_id in selected_products:
            product = products[product_id]
            quantity = random.randint(1, 3)
            price = product['price']

            transaction['items'].append({
                'product_id': product_id,
                'product_name': product['name'],
                'category': product['category'],
                'quantity': quantity,
                'price': price,
                'total': quantity * price
            })

        transactions.append(transaction)

    # Create and train recommender
    recommender = ProductRankRecommender()
    recommender.build_product_graph(transactions, products)

    # Test product ranks
    top_products = recommender.get_product_ranks(top_n=5)
    print("\nTop 5 products by PageRank:")
    for i, product in enumerate(top_products):
        print(f"{i + 1}. {product['product_info']['name']} - Rank: {product['rank']:.6f}")

    # Test collaborative recommendations
    collab_recs = recommender.get_collaborative_recommendations(1, top_n=3)
    print("\nCollaborative recommendations for Product 1:")
    for i, rec in enumerate(collab_recs):
        print(f"{i + 1}. {rec['product_info']['name']} - Score: {rec['score']:.6f}")

    # Test content recommendations
    content_recs = recommender.get_content_recommendations(1, top_n=3)
    print("\nContent-based recommendations for Product 1:")
    for i, rec in enumerate(content_recs):
        print(f"{i + 1}. {rec['product_info']['name']} - Similarity: {rec['similarity']:.6f}")

    # Test hybrid recommendations
    hybrid_recs = recommender.get_hybrid_recommendations(1, top_n=3)
    print("\nHybrid recommendations for Product 1:")
    for i, rec in enumerate(hybrid_recs):
        print(f"{i + 1}. {rec['product_info']['name']} - Score: {rec['hybrid_score']:.6f}")

    # Test basket recommendations
    basket_recs = recommender.get_basket_recommendations([1, 2], top_n=3)
    print("\nRecommendations for basket containing Products 1 and 2:")
    for i, rec in enumerate(basket_recs):
        print(f"{i + 1}. {rec['product_info']['name']} - Score: {rec['score']:.6f}")

    # Test category recommendations
    category_recs = recommender.get_category_recommendations("Category 1", top_n=3)
    print("\nTop products in Category 1:")
    for i, rec in enumerate(category_recs):
        print(f"{i + 1}. {rec['product_info']['name']} - Rank: {rec['rank']:.6f}")

    print("\nProduct rank recommender test completed")


if __name__ == "__main__":
    test_product_rank_recommender()