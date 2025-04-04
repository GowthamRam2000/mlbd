"""
Visualization utilities for the inventory management system.
This module provides functions for creating various visualizations and charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import io
import base64


def get_img_as_base64(fig):
    """
    Convert a matplotlib figure to a base64-encoded image.

    Args:
        fig: Matplotlib figure object

    Returns:
        str: Base64-encoded image
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return img_str


def plot_product_sales(transaction_data, top_n=10, by_category=False):
    """
    Create a bar chart of product sales.

    Args:
        transaction_data: DataFrame with transaction data
        top_n (int): Number of top products to display
        by_category (bool): Whether to group by category

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Ensure we have a DataFrame
    if not isinstance(transaction_data, pd.DataFrame):
        return None

    # Group by product or category
    if by_category:
        sales_data = transaction_data.groupby('category')['quantity'].sum().sort_values(ascending=False)
        title = f"Top {min(top_n, len(sales_data))} Categories by Sales Volume"
        xlabel = "Category"
    else:
        sales_data = transaction_data.groupby(['product_id', 'product_name'])['quantity'].sum().sort_values(
            ascending=False)
        title = f"Top {min(top_n, len(sales_data))} Products by Sales Volume"
        xlabel = "Product"

    # Take top N
    plot_data = sales_data.head(top_n)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot
    if by_category:
        plot_data.plot(kind='bar', ax=ax, color=sns.color_palette("viridis", len(plot_data)))
    else:
        # For products, we need to use the product name as the label
        labels = [name for _, name in plot_data.index]
        values = plot_data.values
        ax.bar(range(len(labels)), values, color=sns.color_palette("viridis", len(labels)))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # Formatting
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Quantity Sold")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add value labels
    for i, v in enumerate(plot_data.values):
        ax.text(i, v + 0.1, str(int(v)), ha='center')

    plt.tight_layout()

    return fig


def plot_sales_time_series(transaction_data, by='day', top_categories=None):
    """
    Create a time series plot of sales.

    Args:
        transaction_data: DataFrame with transaction data
        by (str): Time grouping ('hour', 'day', 'week', 'month')
        top_categories (list): List of categories to include (None = all)

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    # Ensure we have a DataFrame with datetime
    if not isinstance(transaction_data, pd.DataFrame) or 'transaction_date' not in transaction_data.columns:
        return None

    # Ensure transaction_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(transaction_data['transaction_date']):
        transaction_data['transaction_date'] = pd.to_datetime(transaction_data['transaction_date'])

    # Create time period column
    if by == 'hour':
        period_col = transaction_data['transaction_date'].dt.strftime('%Y-%m-%d %H:00')
    elif by == 'day':
        period_col = transaction_data['transaction_date'].dt.strftime('%Y-%m-%d')
    elif by == 'week':
        period_col = transaction_data['transaction_date'].dt.strftime('%Y-%U')
    elif by == 'month':
        period_col = transaction_data['transaction_date'].dt.strftime('%Y-%m')
    else:
        return None

    # Filter categories if needed
    if top_categories:
        transaction_data = transaction_data[transaction_data['category'].isin(top_categories)]

    # Group by time period and category
    grouped = transaction_data.groupby([period_col, 'category'])['quantity'].sum().unstack(fill_value=0)

    # Sort by time period
    grouped = grouped.sort_index()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each category
    for category in grouped.columns:
        ax.plot(grouped.index, grouped[category], marker='o', linestyle='-', label=category)

    # Formatting
    time_labels = {
        'hour': 'Hours',
        'day': 'Days',
        'week': 'Weeks',
        'month': 'Months'
    }

    ax.set_title(f"Sales by {time_labels.get(by, by)}")
    ax.set_xlabel(time_labels.get(by, by))
    ax.set_ylabel("Quantity Sold")
    ax.grid(linestyle='--', alpha=0.7)
    ax.legend(title="Category")

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    return fig


def plot_market_basket_rules(rules, top_n=10):
    """
    Create a visualization of association rules from market basket analysis.

    Args:
        rules (list): List of association rules
        top_n (int): Number of top rules to display

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not rules:
        return None

    # Take top N rules by lift
    rules = sorted(rules, key=lambda x: x['lift'], reverse=True)[:top_n]

    # Extract rule text and metrics
    labels = []
    lifts = []
    supports = []
    confidences = []

    for rule in rules:
        # Create rule text
        if hasattr(rule['antecedent'], '__iter__') and not isinstance(rule['antecedent'], str):
            antecedent = ', '.join([str(a) for a in rule['antecedent']])
        else:
            antecedent = str(rule['antecedent'])

        if hasattr(rule['consequent'], '__iter__') and not isinstance(rule['consequent'], str):
            consequent = ', '.join([str(c) for c in rule['consequent']])
        else:
            consequent = str(rule['consequent'])

        label = f"{antecedent} → {consequent}"
        labels.append(label)

        # Get metrics
        lifts.append(rule['lift'])
        supports.append(rule['support'])
        confidences.append(rule['confidence'])

    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot lifts
    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, lifts, color=sns.color_palette("viridis", len(lifts)))
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()  # Rules at the top
    ax1.set_title('Association Rules by Lift')
    ax1.set_xlabel('Lift')

    # Add lift values
    for i, v in enumerate(lifts):
        ax1.text(v + 0.1, i, f"{v:.2f}", va='center')

    # Plot support and confidence
    ax2.scatter(supports, confidences, s=100, c=lifts, cmap='viridis')

    # Add rule labels
    for i, label in enumerate(labels):
        ax2.annotate(label, (supports[i], confidences[i]),
                     xytext=(5, 5), textcoords='offset points')

    ax2.set_title('Support vs. Confidence')
    ax2.set_xlabel('Support')
    ax2.set_ylabel('Confidence')
    ax2.grid(linestyle='--', alpha=0.7)

    # Add colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(ax2.collections[0], cax=cax)
    cbar.set_label('Lift')

    plt.tight_layout()

    return fig


def plot_product_recommendations(recommendations):
    """
    Create a visualization of product recommendations.

    Args:
        recommendations (list): List of recommendation dictionaries

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not recommendations:
        return None

    # Extract product names and scores
    products = []
    scores = []
    for rec in recommendations:
        product_name = rec.get('product_info', {}).get('name', f"Product {rec.get('product_id', 'unknown')}")
        products.append(product_name)

        # Get the score (could be 'score', 'hybrid_score', or others)
        score = None
        for key in ['hybrid_score', 'score', 'similarity', 'rank']:
            if key in rec:
                score = rec[key]
                break

        if score is None:
            score = 1.0

        scores.append(score)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot horizontal bar chart
    y_pos = np.arange(len(products))
    ax.barh(y_pos, scores, color=sns.color_palette("viridis", len(scores)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(products)
    ax.invert_yaxis()  # Top recommendations at the top

    ax.set_title('Product Recommendations')
    ax.set_xlabel('Score')

    # Add score values
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')

    plt.tight_layout()

    return fig


def plot_product_rank_graph(product_ranks, product_graph=None, top_n=20):
    """
    Create a network visualization of product ranks and relationships.

    Args:
        product_ranks (dict): Dictionary mapping product IDs to ranks
        product_graph (networkx.DiGraph): Product graph
        top_n (int): Number of top products to include

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not product_ranks:
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Get top products by rank
    top_products = sorted(product_ranks.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_product_ids = [p[0] for p in top_products]

    if product_graph:
        # Create subgraph with only top products
        subgraph = product_graph.subgraph(top_product_ids)

        # Get node positions using spring layout
        pos = nx.spring_layout(subgraph, seed=42)

        # Get edge weights for width
        edge_weights = [data.get('weight', 1) for _, _, data in subgraph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1

        # Normalize weights for visualization
        edge_widths = [0.5 + 4.0 * weight / max_weight for weight in edge_weights]

        # Get node sizes based on PageRank
        node_sizes = [5000 * product_ranks[node] for node in subgraph.nodes()]

        # Get node colors based on degree
        node_degrees = dict(subgraph.degree())
        node_colors = [node_degrees[node] for node in subgraph.nodes()]

        # Draw the graph
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors,
                               cmap=plt.cm.viridis, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.5, arrows=True,
                               arrowsize=15, arrowstyle='->', edge_color='gray', ax=ax)

        # Create labels with product names if available
        labels = {}
        for node in subgraph.nodes():
            # Try to get product name from node attributes
            name = subgraph.nodes[node].get('name', f'Product {node}')
            labels[node] = name

        # Draw labels
        nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)

        # Add a colorbar for node degrees
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_degrees.values()),
                                                                           vmax=max(node_degrees.values())))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Node Degree')
    else:
        # No graph available, just plot the ranks
        products = [str(p[0]) for p in top_products]
        ranks = [p[1] for p in top_products]

        # Clear the axis
        ax.clear()

        # Plot horizontal bar chart
        y_pos = np.arange(len(products))
        ax.barh(y_pos, ranks, color=sns.color_palette("viridis", len(ranks)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(products)
        ax.invert_yaxis()  # Top ranks at the top

        ax.set_title('Top Products by PageRank')
        ax.set_xlabel('PageRank Score')

        # Add rank values
        for i, v in enumerate(ranks):
            ax.text(v + 0.001, i, f"{v:.4f}", va='center')

    ax.set_title('Product Relationship Network')
    ax.axis('off')

    plt.tight_layout()

    return fig


def plot_streaming_stats(processor, top_n=10):
    """
    Create visualizations of streaming statistics.

    Args:
        processor: StreamingTransactionProcessor instance
        top_n (int): Number of top items to display

    Returns:
        tuple: Tuple of figure objects
    """
    if not processor:
        return None, None

    # Get frequent products
    frequent_products = processor.get_frequent_products(top_n=top_n)

    # Get category stats
    category_stats = processor.get_category_stats(top_n=top_n)

    # Create figures
    product_fig, ax1 = plt.subplots(figsize=(10, 6))
    category_fig, ax2 = plt.subplots(figsize=(10, 6))

    # Plot frequent products
    products = [f"Product {p[0]}" for p in frequent_products]
    counts = [p[1] for p in frequent_products]

    ax1.barh(np.arange(len(products)), counts, color=sns.color_palette("viridis", len(counts)))
    ax1.set_yticks(np.arange(len(products)))
    ax1.set_yticklabels(products)
    ax1.invert_yaxis()  # Most frequent at the top

    ax1.set_title('Most Frequent Products (Streaming)')
    ax1.set_xlabel('Count')

    # Add count values
    for i, v in enumerate(counts):
        ax1.text(v + 0.1, i, str(int(v)), va='center')

    # Plot category stats
    categories = [c[0] for c in category_stats]
    cat_counts = [c[1] for c in category_stats]

    ax2.barh(np.arange(len(categories)), cat_counts, color=sns.color_palette("viridis", len(cat_counts)))
    ax2.set_yticks(np.arange(len(categories)))
    ax2.set_yticklabels(categories)
    ax2.invert_yaxis()  # Most frequent at the top

    ax2.set_title('Sales by Category (Streaming)')
    ax2.set_xlabel('Count')

    # Add count values
    for i, v in enumerate(cat_counts):
        ax2.text(v + 0.1, i, str(int(v)), va='center')

    plt.tight_layout()

    return product_fig, category_fig


def plot_streaming_time_series(processor, hours=24):
    """
    Create a time series visualization of streaming data.

    Args:
        processor: StreamingTransactionProcessor instance
        hours (int): Number of hours to display

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    if not processor:
        return None

    # Get hourly stats
    hourly_stats = processor.get_hourly_stats(hours=hours)

    # Create a pandas DataFrame from hourly stats
    data = []
    for hour, products in hourly_stats.items():
        for product_id, count in products:
            data.append({
                'hour': hour,
                'product_id': product_id,
                'count': count
            })

    if not data:
        return None

    df = pd.DataFrame(data)

    # Get top 5 products by total count
    top_products = df.groupby('product_id')['count'].sum().sort_values(ascending=False).head(5).index

    # Filter for top products
    df_top = df[df['product_id'].isin(top_products)]

    # Pivot the data for plotting
    pivot_df = df_top.pivot(index='hour', columns='product_id', values='count').fillna(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each product
    for product_id in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[product_id], marker='o', linestyle='-', label=f'Product {product_id}')

    # Formatting
    ax.set_title(f'Product Sales Over Last {hours} Hours')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Quantity Sold')
    ax.grid(linestyle='--', alpha=0.7)
    ax.legend(title="Product")

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    return fig


# Testing function
def test_visualizations():
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    import random

    # Create sample transaction data
    products = []
    for i in range(1, 21):
        category = f"Category {(i - 1) // 5 + 1}"
        products.append({
            'product_id': i,
            'name': f'Product {i}',
            'category': category,
            'price': round(random.uniform(10, 100), 2)
        })

    # Create transaction data
    n_transactions = 200
    data = []

    for i in range(n_transactions):
        # Random date in the last 30 days
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        transaction_date = datetime.now() - timedelta(days=days_ago, hours=hours_ago)

        # Random number of items (1-5)
        n_items = random.randint(1, 5)

        # Select random products
        selected_products = random.sample(products, n_items)

        for product in selected_products:
            quantity = random.randint(1, 3)
            price = product['price']

            data.append({
                'transaction_id': i + 1,
                'transaction_date': transaction_date,
                'product_id': product['product_id'],
                'product_name': product['name'],
                'category': product['category'],
                'quantity': quantity,
                'price': price,
                'total': quantity * price
            })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Test various visualization functions
    print("Testing product sales visualization...")
    fig1 = plot_product_sales(df, top_n=10)
    if fig1:
        fig1.savefig('product_sales.png')
        print("Saved product_sales.png")

    print("Testing category sales visualization...")
    fig2 = plot_product_sales(df, top_n=5, by_category=True)
    if fig2:
        fig2.savefig('category_sales.png')
        print("Saved category_sales.png")

    print("Testing time series visualization...")
    fig3 = plot_sales_time_series(df, by='day')
    if fig3:
        fig3.savefig('sales_time_series.png')
        print("Saved sales_time_series.png")

    # Test market basket visualization
    print("Testing market basket rules visualization...")
    # Create sample rules
    rules = []
    for i in range(10):
        antecedent = frozenset([random.randint(1, 20)])
        consequent = frozenset([random.randint(1, 20)])

        # Make sure antecedent and consequent are different
        while consequent == antecedent:
            consequent = frozenset([random.randint(1, 20)])

        rules.append({
            'antecedent': antecedent,
            'consequent': consequent,
            'support': random.uniform(0.01, 0.1),
            'confidence': random.uniform(0.3, 0.9),
            'lift': random.uniform(1.1, 3.0)
        })

    fig4 = plot_market_basket_rules(rules, top_n=5)
    if fig4:
        fig4.savefig('market_basket_rules.png')
        print("Saved market_basket_rules.png")

    # Test product recommendation visualization
    print("Testing product recommendation visualization...")
    # Create sample recommendations
    recommendations = []
    for i in range(5):
        product_id = random.randint(1, 20)
        recommendations.append({
            'product_id': product_id,
            'product_info': {'name': f'Product {product_id}'},
            'score': random.uniform(0.5, 1.0)
        })

    fig5 = plot_product_recommendations(recommendations)
    if fig5:
        fig5.savefig('product_recommendations.png')
        print("Saved product_recommendations.png")

    print("Visualization testing completed")


if __name__ == "__main__":
    test_visualizations()