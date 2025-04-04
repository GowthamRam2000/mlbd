#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import base64
from datetime import datetime, timedelta
import threading  # Keep import even if not used for threads here
import logging
import sys
import traceback
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D segmentation plots

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler_file = logging.FileHandler("app_debug.log", mode='a')  # Append mode
log_handler_file.setFormatter(log_formatter)
log_handler_stream = logging.StreamHandler()
log_handler_stream.setFormatter(log_formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(log_handler_file)
root_logger.addHandler(log_handler_stream)
logger = logging.getLogger("inventory_app")
logger.setLevel(logging.INFO)
logger.info("-------------------- Application Starting --------------------")

# --- Add project root to path ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
logger.info(f"Added project root to sys.path: {project_root}")

# --- Import Project Modules ---
try:
    from utils.data_loader import DataManager
    from models.apriori import AprioriAnalysis
    from models.stream_counter import StreamingTransactionProcessor, TransactionSimulator
    from utils.simulator import TransactionGenerator
    from models.recommendation import ProductRankRecommender
    from models.ml_models import InventoryPredictionModel, SpoilageDetectionModel, generate_inventory_report
    from utils.visualizations import (
        plot_product_sales, plot_sales_time_series, plot_market_basket_rules,
        plot_product_recommendations, plot_product_rank_graph, plot_streaming_stats,
        plot_streaming_time_series, get_img_as_base64
    )
    # Import segmentation functionality – assumes you have implemented this module.
    from models.segmentation import CustomerSegmentation
    logger.info("Successfully imported project modules")
except Exception as e:
    logger.critical(f"CRITICAL ERROR importing modules: {e}", exc_info=True)
    st.error(f"Fatal Error: Could not import required modules. Check logs. Error: {e}")
    st.stop()

# --- Global Placeholders (if needed) ---
data_manager = None
apriori_model = None
streaming_processor = None
transaction_simulator = None
recommender = None

# --- Cached Resource Initialization Functions ---
@st.cache_resource
def get_data_manager(db_path='data/inventory.db'):
    logger.info("Attempting to get or create DataManager instance...")
    try:
        dm = DataManager(db_path=db_path)
        logger.info("DataManager instance obtained/created successfully.")
        return dm
    except Exception as e:
        logger.error(f"Error creating/getting DataManager: {e}", exc_info=True)
        st.error(f"Error initializing data manager: {e}")
        st.stop()

def init_data(force_generate=False):
    try:
        db_path = os.path.join('data', 'inventory.db')
        logger.info(f"Checking for database at {db_path}")
        dm = get_data_manager(db_path)
        if force_generate or not os.path.exists(db_path) or os.path.getsize(db_path) < 5000:
            if force_generate:
                logger.info("Forcing synthetic data generation.")
            else:
                logger.info("Database not found or appears empty, generating synthetic data...")
            try:
                num_txns_gen = 10000
                with st.spinner(f'Generating synthetic data ({num_txns_gen:,} transactions). This may take a minute...'):
                    dm.generate_synthetic_data(num_products=100, num_customers=200, num_transactions=num_txns_gen)
                logger.info("Synthetic data generation completed")
                st.success(f'Synthetic data generated ({num_txns_gen:,} transactions)!')
                st.cache_data.clear()
            except Exception as e:
                logger.error(f"Error generating synthetic data: {e}", exc_info=True)
                st.error(f"Error during synthetic data generation: {e}")
        else:
            logger.info(f"Database found ({os.path.getsize(db_path):,} bytes), skipping generation.")
        return dm
    except Exception as e:
        logger.error(f"Error in init_data: {e}", exc_info=True)
        st.error(f"Could not initialize data: {e}")
        st.stop()

@st.cache_data(ttl=600)
def load_products_cached():
    try:
        dm = get_data_manager()
        logger.info("Loading products from database (cached)...")
        products = dm.load_products()
        logger.info(f"Loaded {len(products)} products.")
        return products
    except Exception as e:
        logger.error(f"Error loading products: {e}", exc_info=True)
        st.error("Could not load product data.")
        return []

@st.cache_data(ttl=600)
def load_inventory_cached():
    try:
        dm = get_data_manager()
        logger.info("Loading inventory from database (cached)...")
        inventory = dm.load_inventory()
        logger.info(f"Loaded {len(inventory)} inventory records.")
        return pd.DataFrame(inventory) if inventory else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading inventory: {e}", exc_info=True)
        st.error("Could not load inventory data.")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_transactions_cached():
    try:
        dm = get_data_manager()
        logger.info("Loading transactions from database (cached)...")
        transactions = dm.load_transactions()
        logger.info(f"Loaded {len(transactions)} recent transactions.")
        return transactions
    except Exception as e:
        logger.error(f"Error loading transactions: {e}", exc_info=True)
        st.error("Could not load transaction data.")
        return []

@st.cache_data(ttl=600)
def get_transaction_df_cached():
    try:
        dm = get_data_manager()
        logger.info("Getting transaction DataFrame (cached)...")
        df = dm.get_transaction_df()
        logger.info(f"Transaction DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error getting transaction DataFrame: {e}", exc_info=True)
        st.error("Could not load detailed transaction data.")
        return pd.DataFrame()

def get_total_transaction_count_cached():
    try:
        dm = get_data_manager()
        cursor = dm._get_cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]
        logger.info(f"Total transaction count: {count}")
        return count
    except Exception as e:
        logger.error(f"Error getting total transaction count: {e}", exc_info=True)
        st.error("Could not get total transaction count.")
        return 0

@st.cache_resource
def run_apriori_analysis(min_support, min_confidence):
    try:
        logger.info(f"Running Apriori analysis with min_support={min_support}, min_confidence={min_confidence}")
        dm = get_data_manager()
        baskets = dm.get_transaction_baskets()
        if not baskets:
            st.warning("No transaction baskets found to run Apriori.")
            return None
        products_list = dm.load_products()
        products_dict = {p['product_id']: p for p in products_list}
        model = AprioriAnalysis(min_support=min_support, min_confidence=min_confidence)
        with st.spinner('Running Apriori analysis on transaction data...'):
            model.fit(baskets, products_dict)
        logger.info("Apriori analysis completed")
        return model
    except Exception as e:
        logger.error(f"Error running Apriori analysis: {e}", exc_info=True)
        st.error(f"Apriori analysis failed: {e}")
        return None

def init_streaming_processor():
    try:
        if 'streaming_processor' not in st.session_state:
            logger.info("Initializing streaming processor in session state")
            st.session_state.streaming_processor = StreamingTransactionProcessor(space_saving_k=50, cms_width=1000, cms_depth=5)
        return st.session_state.streaming_processor
    except Exception as e:
        logger.error(f"Error initializing streaming processor: {e}", exc_info=True)
        st.error("Could not initialize streaming processor.")
        return None

def init_transaction_simulator():
    try:
        if 'transaction_simulator' not in st.session_state:
            logger.info("Initializing transaction simulator in session state")
            dm = get_data_manager()
            products = load_products_cached()
            if not products:
                st.error("Cannot initialize simulator without product data.")
                return None
            st.session_state.transaction_simulator = TransactionSimulator(data_manager=dm, products=products, rate_mean=1.0, batch_size=1)
        return st.session_state.transaction_simulator
    except Exception as e:
        logger.error(f"Error initializing transaction simulator: {e}", exc_info=True)
        st.error("Could not initialize transaction simulator.")
        return None

@st.cache_resource
def init_recommender():
    try:
        logger.info("Initializing recommender...")
        recommender_instance = ProductRankRecommender()
        dm = get_data_manager()
        transactions_df = get_transaction_df_cached()
        products_list = load_products_cached()
        products_dict = {p['product_id']: p for p in products_list}
        if transactions_df.empty or not products_dict:
            st.warning("Insufficient data to build recommendation graph.")
            return recommender_instance
        logger.info("Building product recommendation graph")
        with st.spinner('Building product recommendation graph...'):
            recommender_instance.build_product_graph(transactions_df, products_dict)
        logger.info("Recommender initialized successfully.")
        return recommender_instance
    except Exception as e:
        logger.error(f"Error initializing recommender: {e}", exc_info=True)
        st.error(f"Failed to initialize recommendation system: {e}")
        return ProductRankRecommender()

def preload_ml_models():
    try:
        logger.info("Checking and preloading ML models...")
        dm = get_data_manager()
        transaction_df = get_transaction_df_cached()
        inventory_df = load_inventory_cached()
        products_df = pd.DataFrame(load_products_cached())
        if transaction_df.empty or inventory_df.empty:
            logger.warning("Insufficient data loaded, cannot preload ML models.")
            st.warning("Insufficient data found. ML models cannot be pre-trained.")
            return
        prediction_model_path = 'models/inventory_prediction_model.joblib'
        spoilage_model_path = 'models/spoilage_detection_model.joblib'
        os.makedirs('models', exist_ok=True)
        if not os.path.exists(prediction_model_path):
            logger.info(f"Prediction model not found at '{prediction_model_path}'. Training...")
            st.info("First time setup: Training inventory prediction model...")
            with st.spinner("Training inventory prediction model (this might take a moment)..."):
                try:
                    pred_model = InventoryPredictionModel()
                    X, y = pred_model.prepare_data(transaction_df, inventory_df)
                    if X.empty or y.empty:
                        logger.error("Data preparation for prediction model resulted in empty X or y.")
                        st.error("Failed to prepare data for prediction model training.")
                        return
                    metrics = pred_model.train(X, y)
                    logger.info(f"Prediction model training metrics: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.2f}")
                    pred_model.save_model(prediction_model_path)
                    logger.info(f"Saved prediction model to {prediction_model_path}")
                    st.success("Inventory prediction model trained and saved.")
                except Exception as e:
                    logger.error(f"Error during prediction model preloading: {e}", exc_info=True)
                    st.error(f"Failed to train prediction model: {e}")
        else:
            logger.info(f"Prediction model found at {prediction_model_path}. Skipping training.")
        if not os.path.exists(spoilage_model_path):
            logger.info(f"Spoilage model not found at '{spoilage_model_path}'. Training...")
            st.info("First time setup: Training spoilage detection model...")
            with st.spinner("Training spoilage detection model..."):
                try:
                    spoil_model = SpoilageDetectionModel()
                    features = spoil_model.prepare_data(transaction_df, inventory_df, products_df)
                    if features.empty:
                        logger.error("Data preparation for spoilage model resulted in empty features.")
                        st.error("Failed to prepare data for spoilage model training.")
                        return
                    spoil_model.train(features, contamination=0.05)
                    spoil_model.save_model(spoilage_model_path)
                    logger.info(f"Saved spoilage model to {spoilage_model_path}")
                    st.success("Spoilage detection model trained and saved.")
                except Exception as e:
                    logger.error(f"Error during spoilage model preloading: {e}", exc_info=True)
                    st.error(f"Failed to train spoilage model: {e}")
        else:
            logger.info(f"Spoilage model found at {spoilage_model_path}. Skipping training.")
        logger.info("ML models preloading check completed.")
    except Exception as e:
        logger.error(f"General error during ML model preloading check: {e}", exc_info=True)
        st.warning(f"Could not complete ML model preloading: {e}")

# --- UI Rendering Functions ---
def render_header():
    try:
        st.title('📊 Inventory Management System')
        st.subheader('A Demonstration of Big Data Concepts')
        st.markdown("*Explore insights from inventory data using techniques covered in the Big Data course.*")
        logger.debug("Header rendered.")
    except Exception as e:
        logger.error(f"Error rendering header: {e}", exc_info=True)
        st.error("Error displaying header.")

def render_dashboard(dm):
    try:
        st.header('📈 Dashboard Overview')
        products = load_products_cached()
        inventory_df = load_inventory_cached()
        total_transactions_count = get_total_transaction_count_cached()
        transaction_df_detail = get_transaction_df_cached()
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Total Products', f"{len(products):,}")
        with col2:
            total_inventory = int(inventory_df['quantity'].sum()) if not inventory_df.empty else 0
            st.metric('Total Units in Stock', f"{total_inventory:,}")
        with col3:
            st.metric('Total Transactions', f"{total_transactions_count:,}")
        with col4:
            low_stock_items = int((inventory_df['quantity'] < 20).sum()) if not inventory_df.empty else 0
            st.metric('Low Stock Items (<20)', f"{low_stock_items:,}")
        st.markdown("---")
        if not transaction_df_detail.empty:
            if not pd.api.types.is_datetime64_any_dtype(transaction_df_detail['transaction_date']):
                transaction_df_detail['transaction_date'] = pd.to_datetime(transaction_df_detail['transaction_date'], errors='coerce')
            transaction_df_detail = transaction_df_detail.dropna(subset=['transaction_date'])
            st.subheader('Recent Sales Trend (Last 30 Days)')
            fig_ts = plot_sales_time_series(transaction_df_detail, by='day')
            st.pyplot(fig_ts)
            col_prod, col_cat = st.columns(2)
            with col_prod:
                st.subheader('Top 10 Selling Products')
                fig_prod = plot_product_sales(transaction_df_detail, top_n=10)
                st.pyplot(fig_prod)
            with col_cat:
                st.subheader('Top 5 Selling Categories')
                fig_cat = plot_product_sales(transaction_df_detail, top_n=5, by_category=True)
                st.pyplot(fig_cat)
        else:
            st.info('No transaction data available to display charts.')
        logger.debug("Dashboard rendered.")
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}", exc_info=True)
        st.error("Could not render dashboard.")

def render_market_basket_analysis(dm, products_list):
    try:
        st.header('🛒 Market Basket Analysis (Apriori)')
        st.markdown("Discover frequently co-purchased products.")
        st.caption("*Concepts: Frequent Itemsets, Association Rules, Support, Confidence, Lift*")
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider('Min Support', 0.01, 0.5, 0.02, 0.01, format="%.2f", help="Min proportion of transactions.")
        with col2:
            min_confidence = st.slider('Min Confidence', 0.1, 1.0, 0.5, 0.05, format="%.2f", help="Min likelihood Y bought if X bought.")
        if st.button('Run Apriori Analysis', key='run_apriori'):
            apriori_instance = run_apriori_analysis(min_support, min_confidence)
            if apriori_instance:
                st.session_state.frequent_itemsets = apriori_instance.get_frequent_itemsets(with_product_names=True)
                st.session_state.association_rules = apriori_instance.get_association_rules(with_product_names=True, sort_by='lift')
                st.session_state.bundle_opportunities = apriori_instance.find_bundle_opportunities(min_lift=1.2, min_confidence=0.5, top_n=5)
                st.success("Apriori analysis complete.")
            else:
                st.error("Apriori analysis failed.")
                st.session_state.frequent_itemsets = []
                st.session_state.association_rules = []
                st.session_state.bundle_opportunities = []
        if 'frequent_itemsets' in st.session_state:
            st.subheader('Frequent Itemsets Results')
            frequent_itemsets = st.session_state.frequent_itemsets
            if frequent_itemsets:
                itemsets_data = [
                    {
                        'Items': ', '.join([item.get('name', f"ID:{item.get('product_id', '?')}") for item in iset.get('items', [])]),
                        'Length': iset.get('length', '?'),
                        'Support Count': iset.get('support_count', '?')
                    }
                    for iset in frequent_itemsets
                ]
                st.dataframe(pd.DataFrame(itemsets_data))
            else:
                st.info('No frequent itemsets found.')
        if 'association_rules' in st.session_state:
            st.subheader('Association Rules Results')
            association_rules = st.session_state.association_rules
            if association_rules:
                rules_data = [
                    {
                        'If Buy...': ', '.join([item.get('name', f"ID:{item.get('product_id', '?')}") for item in rule.get('antecedent_items', [])]),
                        'Then Buy...': ', '.join([item.get('name', f"ID:{item.get('product_id', '?')}") for item in rule.get('consequent_items', [])]),
                        'Support': f"{rule.get('support', 0):.4f}",
                        'Confidence': f"{rule.get('confidence', 0):.4f}",
                        'Lift': f"{rule.get('lift', 0):.2f}"
                    }
                    for rule in association_rules
                ]
                st.dataframe(pd.DataFrame(rules_data))
                st.subheader('Top 10 Rules Visualization (by Lift)')
                try:
                    fig_rules = plot_market_basket_rules(association_rules, top_n=10)
                    st.pyplot(fig_rules)
                except Exception as plot_e:
                    logger.error(f"Error plotting MBA rules: {plot_e}", exc_info=True)
                    st.warning("Could not generate rule visualization.")
            else:
                st.info('No association rules found.')
        if 'bundle_opportunities' in st.session_state:
            st.subheader('Potential Product Bundles')
            bundles = st.session_state.bundle_opportunities
            if bundles:
                for i, bundle in enumerate(bundles):
                    bundle_names = [p.get('name', f"ID:{p.get('product_id', '?')}") for p in bundle.get('product_info', [])]
                    st.markdown(f"**Bundle {i+1}:** ({', '.join(bundle_names)})")
                    st.markdown(f"&nbsp;&nbsp;*Lift: {bundle.get('lift', 0):.2f}, Conf: {bundle.get('confidence', 0):.2f}*")
            else:
                st.info('No strong bundles found.')
        logger.debug("Market basket tab rendered.")
    except Exception as e:
        logger.error(f"Error rendering MBA: {e}", exc_info=True)
        st.error("Market Basket failed.")

def render_streaming_data():
    try:
        st.header('🔴 Streaming Data Simulation')
        st.markdown("Simulate and analyze a real-time transaction stream.")
        st.caption("*Concepts: Streaming Algorithms, Frequent Items (Space-Saving)*")
        processor = init_streaming_processor()
        simulator = init_transaction_simulator()
        if processor is None or simulator is None:
            st.error("Streaming init failed.")
            return
        if 'simulation_running' not in st.session_state:
            st.session_state.simulation_running = False
            st.session_state.processed_count = 0
        col1, col2 = st.columns(2)
        with col1:
            if st.button('▶️ Start Simulation', key='start_sim_stream', disabled=st.session_state.simulation_running):
                if not simulator.running:
                    simulator.start()
                    st.session_state.simulation_running = True
                    st.session_state.processed_count = 0
                    logger.info("Sim started.")
                    st.rerun()
        with col2:
            if st.button('⏹️ Stop Simulation', key='stop_sim_stream', disabled=not st.session_state.simulation_running):
                if simulator.running:
                    simulator.stop()
                    st.session_state.simulation_running = False
                    logger.info("Sim stopped.")
                    st.rerun()
        if st.session_state.simulation_running:
            st.info("Simulation running...")
            new_transactions = simulator.get_transactions(max_transactions=5)
            if new_transactions:
                for txn in new_transactions:
                    processor.process_transaction(txn)
                st.session_state.processed_count += len(new_transactions)
                logger.debug(f"Processed {len(new_transactions)} txns.")
                time.sleep(0.5)
                st.rerun()
            else:
                time.sleep(1.0)
                st.rerun()
        metrics_col1, metrics_col2 = st.columns(2)
        charts_col1, charts_col2 = st.columns(2)
        with metrics_col1:
            rate = processor.get_transaction_rate(minutes=1) if processor else 0.0
            st.metric('Current Txn Rate', f"{rate:.1f} txn/min")
        with metrics_col2:
            st.metric("Processed Txns (Live)", st.session_state.get('processed_count', 0))
        try:
            if processor:
                product_fig, category_fig = plot_streaming_stats(processor, top_n=10)
                with charts_col1:
                    st.subheader("Freq. Products (Live)")
                    st.pyplot(product_fig)
                with charts_col2:
                    st.subheader("Freq. Categories (Live)")
                    st.pyplot(category_fig)
                st.subheader("Hourly Txn Count (Live)")
                time_fig = plot_streaming_time_series(processor, hours=12)
                st.pyplot(time_fig)
            else:
                st.info("Streaming processor not available.")
        except Exception as plot_err:
            logger.error(f"Error plotting streaming: {plot_err}", exc_info=True)
            st.error("Plotting failed.")
        if not st.session_state.simulation_running:
            if st.button('Refresh Stats', key='refresh_stream_stats'):
                logger.info("Manual refresh.")
                st.rerun()
        logger.debug("Streaming tab rendered.")
    except Exception as e:
        logger.error(f"Error rendering streaming: {e}", exc_info=True)
        st.error("Streaming tab failed.")

def render_recommendation_system(dm, transactions_df, products_list):
    try:
        st.header('🤝 Product Recommendation System')
        st.markdown("Get recommendations based on purchase history and similarity.")
        st.caption("*Concepts: PageRank, Graph Methods, Collaborative/Content Filtering*")
        recommender_instance = init_recommender()
        if recommender_instance is None:
            st.error("Recommender init failed.")
            return
        if not products_list:
            st.warning("No product data.")
            return
        product_options = {f"{p.get('name','?')} (ID: {p.get('product_id','?')})": p.get('product_id') for p in products_list if p.get('product_id') is not None}
        if not product_options:
            st.warning("No valid product options.")
            return
        tab_ranks, tab_single, tab_basket = st.tabs(["Top Ranked", "Single Product", "Basket Recs"])
        with tab_ranks:
            st.subheader('Top Products by Importance (PageRank)')
            top_products = recommender_instance.get_product_ranks(top_n=15)
            if top_products:
                ranks_data = [
                    {
                        'Rank': i + 1,
                        'Product': p.get('product_info', {}).get('name', f"ID:{p.get('product_id','?')}"),
                        'Category': p.get('product_info', {}).get('category', 'Unknown'),
                        'Score': f"{p.get('rank',0):.6f}"
                    }
                    for i, p in enumerate(top_products)
                ]
                st.dataframe(pd.DataFrame(ranks_data))
            else:
                st.info('No product ranks available.')
        with tab_single:
            st.subheader('Recommendations for a Single Product')
            selected_product_display = st.selectbox('Select product:', options=list(product_options.keys()), key='rec_single_select')
            if selected_product_display:
                product_id = product_options.get(selected_product_display)
                if product_id is not None:
                    rec_col1, rec_col2 = st.columns(2)
                    with rec_col1:
                        st.markdown("**Users also bought...**")
                        collab_recs = recommender_instance.get_collaborative_recommendations(product_id, top_n=5)
                        if collab_recs:
                            for i, r in enumerate(collab_recs):
                                st.write(f"{i+1}. {r.get('product_info',{}).get('name', f'ID:{r.get('product_id','?')}')} (Score: {r.get('score',0):.4f})")
                        else:
                            st.info("None found.")
                    with rec_col2:
                        st.markdown("**Similar products...**")
                        content_recs = recommender_instance.get_content_recommendations(product_id, top_n=5)
                        if content_recs:
                            for i, r in enumerate(content_recs):
                                st.write(f"{i+1}. {r.get('product_info',{}).get('name', f'ID:{r.get('product_id','?')}')} (Sim: {r.get('similarity',0):.4f})")
                        else:
                            st.info("None found.")
                    st.subheader('Hybrid Recommendations (Combined)')
                    hybrid_recs = recommender_instance.get_hybrid_recommendations(product_id, top_n=10)
                    if hybrid_recs:
                        try:
                            fig_hybrid = plot_product_recommendations(hybrid_recs)
                            st.pyplot(fig_hybrid)
                        except Exception as hybrid_e:
                            logger.error(f"Error plotting hybrid recs: {hybrid_e}", exc_info=True)
                            st.warning("Plot failed.")
                    else:
                        st.info('No hybrid recommendations found.')
                else:
                    st.warning("Invalid product.")
        with tab_basket:
            st.subheader('Recommendations for Basket')
            selected_basket_display = st.multiselect('Select products in basket:', options=list(product_options.keys()), key='rec_basket_select')
            basket_ids = [product_options.get(p) for p in selected_basket_display if product_options.get(p) is not None]
            if basket_ids:
                basket_recs = recommender_instance.get_basket_recommendations(basket_ids, top_n=5)
                if basket_recs:
                    st.markdown("**Frequently bought next:**")
                    for i, r in enumerate(basket_recs):
                        st.write(f"{i+1}. {r.get('product_info',{}).get('name', f'ID:{r.get('product_id','?')}')} (Score: {r.get('score',0):.4f})")
                else:
                    st.info('No recommendations found.')
            else:
                st.info('Select products.')
        logger.debug("Recommendation tab rendered.")
    except Exception as e:
        logger.error(f"Error rendering recsys: {e}", exc_info=True)
        st.error("Recsys tab failed.")

def render_ml_tab(dm, transaction_df, inventory_df, products_df):
    try:
        st.header('🤖 Machine Learning Models')
        st.markdown("Apply predictive analytics and anomaly detection.")
        st.caption("*Concepts: Regression, Anomaly Detection, Feature Engineering*")
        if transaction_df.empty or inventory_df.empty or products_df.empty:
            st.warning("Insufficient data for ML.")
            return
        from models.ml_models import InventoryPredictionModel, SpoilageDetectionModel, generate_inventory_report
        model_type = st.radio("Select Task:", ["Inventory Prediction", "Spoilage Detection", "Generate PDF Report"], key="ml_model_type_select", horizontal=True)
        st.markdown("---")
        if model_type == "Inventory Prediction":
            st.subheader("Demand Forecasting & Low Stock")
            st.markdown("Train/use RandomForest for demand prediction.")
            if 'prediction_model' not in st.session_state:
                try:
                    pred_model_path = 'models/inventory_prediction_model.joblib'
                    st.session_state.prediction_model = InventoryPredictionModel(pred_model_path)
                    logger.info("Loaded pred model.")
                    st.info("Loaded pre-trained model.")
                except FileNotFoundError:
                    logger.info("Pred model not found.")
                    st.session_state.prediction_model = InventoryPredictionModel()
                    st.warning("Model not trained yet.")
                except Exception as e:
                    logger.error(f"Err loading pred model: {e}", exc_info=True)
                    st.error(f"Err loading: {e}")
                    st.session_state.prediction_model = InventoryPredictionModel()
            pred_model = st.session_state.prediction_model
            col_pred_train, col_pred_run = st.columns(2)
            with col_pred_train:
                if st.button("Train Prediction Model", key="train_pred"):
                    with st.spinner("Training..."):
                        try:
                            pred_model = InventoryPredictionModel()
                            X, y = pred_model.prepare_data(transaction_df, inventory_df)
                            if X.empty or y.empty:
                                raise ValueError("Prep empty.")
                            metrics = pred_model.train(X, y)
                            st.session_state.prediction_model = pred_model
                            pred_model.save_model()
                            st.success(f"Trained! (RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.2f})")
                            logger.info("Pred model trained.")
                            if 'future_predictions' in st.session_state:
                                del st.session_state.future_predictions
                            if 'low_stock_items' in st.session_state:
                                del st.session_state.low_stock_items
                        except Exception as e:
                            logger.error(f"Err training pred: {e}", exc_info=True)
                            st.error(f"Training err: {e}")
            with col_pred_run:
                if st.button("Predict & Find Low Stock", key="run_pred"):
                    if pred_model.model is None:
                        st.error("Model not trained yet.")
                    else:
                        with st.spinner("Predicting..."):
                            try:
                                X_full, _ = pred_model.prepare_data(transaction_df, inventory_df)
                                if X_full.empty:
                                    raise ValueError("Prep empty.")
                                X_latest = X_full.loc[X_full.groupby('product_id')['transaction_date'].idxmax()]
                                predictions = pred_model.predict_future_needs(X_latest, days_ahead=7)
                                low_stock = pred_model.identify_low_stock_items(inventory_df, predictions, threshold_days=5)
                                st.session_state.future_predictions = predictions
                                st.session_state.low_stock_items = low_stock
                                st.success(f"Predicted. Low stock: {len(low_stock)}")
                                logger.info(f"Pred run. Low stock: {len(low_stock)}")
                            except Exception as e:
                                logger.error(f"Err running pred: {e}", exc_info=True)
                                st.error(f"Prediction err: {e}")
            st.markdown("---")
            if 'future_predictions' in st.session_state:
                st.subheader("Predicted Sales - Next 7 Days")
                try:
                    preds_df = st.session_state.future_predictions
                    prod_names = products_df.set_index('product_id')['name'].to_dict()
                    piv = preds_df.pivot_table(index='product_id', columns='date', values='predicted_sales').fillna(0)
                    piv.columns = [pd.to_datetime(c).strftime('%Y-%m-%d') for c in piv.columns]
                    piv['Product Name'] = piv.index.map(lambda x: prod_names.get(str(x), f"ID:{x}"))
                    piv = piv[['Product Name'] + [c for c in piv.columns if c != 'Product Name']]
                    fmt = piv.copy()
                    fmt.iloc[:, 1:] = fmt.iloc[:, 1:].applymap(lambda x: f"{pd.to_numeric(x, errors='coerce'):.1f}")
                    st.dataframe(fmt, height=300)
                except Exception as e:
                    logger.error(f"Err display preds: {e}", exc_info=True)
                    st.error("Err display preds.")
            if 'low_stock_items' in st.session_state:
                st.subheader("⚠️ Low Stock Alert (<= 5 Days)")
                low_df = st.session_state.low_stock_items
                if not low_df.empty:
                    try:
                        prod_names = products_df.set_index('product_id')['name'].to_dict()
                        disp = low_df.copy()
                        disp['Product Name'] = disp['product_id'].map(lambda x: prod_names.get(str(x), f"ID:{x}"))
                        disp = disp.sort_values('days_until_stockout')[['product_id', 'Product Name', 'current_inventory', 'avg_daily_sales', 'days_until_stockout', 'stockout_date']]
                        disp['stockout_date'] = pd.to_datetime(disp['stockout_date']).dt.strftime('%Y-%m-%d')
                        disp['current_inventory'] = disp['current_inventory'].apply(lambda x: f"{pd.to_numeric(x, errors='coerce'):,.0f}")
                        disp['avg_daily_sales'] = disp['avg_daily_sales'].apply(lambda x: f"{pd.to_numeric(x, errors='coerce'):,.1f}")
                        disp['days_until_stockout'] = disp['days_until_stockout'].apply(lambda x: f"{pd.to_numeric(x, errors='coerce'):,.0f}")
                        disp.columns = ['ID', 'Product', 'Inv.', 'Avg Sales', 'Days Out', 'Date Out']
                        st.dataframe(disp, hide_index=True, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Err display low stock: {e}", exc_info=True)
                        st.error("Err display low stock.")
                else:
                    st.info("No low stock items detected.")
        elif model_type == "Spoilage Detection":
            st.subheader("Spoilage / Obsolescence Detection")
            st.markdown("Train/use Isolation Forest for anomaly detection.")
            if 'spoilage_model' not in st.session_state:
                try:
                    spoil_path = 'models/spoilage_detection_model.joblib'
                    st.session_state.spoilage_model = SpoilageDetectionModel(spoil_path)
                    logger.info("Loaded spoil model.")
                    st.info("Loaded pre-trained model.")
                except FileNotFoundError:
                    logger.info("Spoil model not found.")
                    st.session_state.spoilage_model = SpoilageDetectionModel()
                    st.warning("Model not trained.")
                except Exception as e:
                    logger.error(f"Err loading spoil model: {e}", exc_info=True)
                    st.error(f"Err loading: {e}")
                    st.session_state.spoilage_model = SpoilageDetectionModel()
            spoil_model = st.session_state.spoilage_model
            col_spoil_train, col_spoil_run = st.columns(2)
            with col_spoil_train:
                contam = st.slider("Expected Anomaly %", 0.01, 0.20, 0.05, 0.01, format="%.2f", key='spoil_contam', help="Est. % anomalous items.")
                if st.button("Train Spoilage Model", key="train_spoil"):
                    with st.spinner("Training..."):
                        try:
                            spoil_model = SpoilageDetectionModel()
                            feats = spoil_model.prepare_data(transaction_df, inventory_df, products_df)
                            if feats.empty:
                                raise ValueError("Prep empty.")
                            spoil_model.train(feats, contamination=contam)
                            st.session_state.spoilage_model = spoil_model
                            spoil_model.save_model()
                            st.success(f"Spoilage model trained! (Contam: {contam:.2f})")
                            logger.info("Spoil model trained.")
                            if 'spoilt_items' in st.session_state:
                                del st.session_state.spoilt_items
                        except Exception as e:
                            logger.error(f"Err training spoil: {e}", exc_info=True)
                            st.error(f"Training err: {e}")
            with col_spoil_run:
                thresh = st.number_input("Anomaly Score Threshold", min_value=0.0, value=0.6, step=0.05, format="%.2f", key='spoil_thresh', help="Manual threshold (higher = more anomalous).")
                if st.button("Detect Spoilt Items", key="run_spoil"):
                    if spoil_model.model is None:
                        st.error("Model not trained.")
                    else:
                        with st.spinner("Detecting..."):
                            try:
                                spoilt_items = spoil_model.identify_spoilt_items(transaction_df, inventory_df, products_df, threshold=thresh)
                                st.session_state.spoilt_items = spoilt_items
                                st.success(f"Detected {len(spoilt_items)} anomalies.")
                                logger.info(f"Spoil detection run. Anomalies: {len(spoilt_items)}")
                            except Exception as e:
                                logger.error(f"Err running spoil detection: {e}", exc_info=True)
                                st.error(f"Detection err: {e}")
            st.markdown("---")
            if 'spoilt_items' in st.session_state:
                st.subheader("🚨 Potentially Spoilt / Obsolete Items")
                spoil_df = st.session_state.spoilt_items
                if not spoil_df.empty:
                    try:
                        disp = spoil_df.copy()
                        cols = ['product_id', 'name', 'category', 'current_inventory', 'days_since_last_transaction', 'anomaly_score']
                        cols = [c for c in cols if c in disp.columns]
                        disp = disp[cols]
                        if 'current_inventory' in disp:
                            disp['current_inventory'] = disp['current_inventory'].apply(lambda x: f"{pd.to_numeric(x, errors='coerce'):,.0f}")
                        if 'days_since_last_transaction' in disp:
                            disp['days_since_last_transaction'] = disp['days_since_last_transaction'].apply(lambda x: f"{pd.to_numeric(x, errors='coerce'):,.0f}" if pd.to_numeric(x, errors='coerce') != -1 else 'N/A')
                        if 'anomaly_score' in disp:
                            disp['anomaly_score'] = disp['anomaly_score'].apply(lambda x: f"{pd.to_numeric(x, errors='coerce'):.4f}")
                        r_map = {'product_id': 'ID', 'name': 'Product', 'category': 'Category', 'current_inventory': 'Inv.', 'days_since_last_transaction': 'Days Since Sale', 'anomaly_score': 'Anomaly Score'}
                        disp.columns = [r_map.get(c, c) for c in disp.columns]
                        st.dataframe(disp, hide_index=True, use_container_width=True)
                        st.subheader("Top 20 Anomaly Scores")
                        if 'Product' in disp.columns and 'Anomaly Score' in disp.columns:
                            plot_data = spoil_df.sort_values('anomaly_score', ascending=False).head(20)
                            if not plot_data.empty:
                                fig, ax = plt.subplots(figsize=(10, 7))
                                plot_data['score_num'] = pd.to_numeric(plot_data['anomaly_score'], errors='coerce')
                                norm = plot_data['score_num'].fillna(0)
                                colors = plt.cm.viridis(norm / (norm.max() + 1e-9))
                                plot_names = plot_data.get('name', plot_data['product_id'])
                                ax.bar(plot_names, plot_data['score_num'], color=colors)
                                ax.set_ylabel('Anomaly Score')
                                ax.set_title('Top 20 Anomaly Scores')
                                plt.xticks(rotation=75, ha='right')
                                plt.grid(axis='y', ls='--', alpha=0.7)
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.info("No data to plot.")
                        else:
                            st.warning("Plot failed: Columns missing.")
                    except Exception as e:
                        logger.error(f"Err display spoilt: {e}", exc_info=True)
                        st.error("Err display results.")
                else:
                    st.info("No items flagged.")
        elif model_type == "Generate PDF Report":
            st.subheader("Generate Inventory Summary Report (PDF)")
            st.markdown("Create PDF summary of low stock & spoilt items.")
            low_stock_data = st.session_state.get('low_stock_items', pd.DataFrame())
            spoilt_items_data = st.session_state.get('spoilt_items', pd.DataFrame())
            if low_stock_data.empty and spoilt_items_data.empty:
                st.warning("Run prediction/spoilage first.")
            else:
                st.info(f"Report includes {len(low_stock_data)} low stock & {len(spoilt_items_data)} spoilt items.")
                report_fn = st.text_input("Report Filename:", value=f"Inv_Report_{datetime.now():%Y%m%d_%H%M}.pdf")
                if st.button("Generate & Download PDF", key="gen_pdf"):
                    if not report_fn.lower().endswith(".pdf"):
                        st.error("Filename must end .pdf")
                    else:
                        with st.spinner("Generating PDF..."):
                            try:
                                os.makedirs('reports', exist_ok=True)
                                rp = os.path.join('reports', report_fn)
                                logger.info(f"Generating PDF: {rp}")
                                out_path = generate_inventory_report(low_stock_data, spoilt_items_data, products_df, output_path=rp)
                                st.success(f"Report ready!")
                                try:
                                    with open(out_path, "rb") as f:
                                        st.download_button("Download PDF", f, report_fn, "application/pdf")
                                    logger.info(f"Download btn created for {out_path}")
                                except Exception as ed:
                                    logger.error(f"Err download btn: {ed}", exc_info=True)
                                    st.error("Download failed.")
                            except ImportError as e_imp:
                                logger.error(f"ReportLab missing: {e_imp}")
                                st.error("PDF failed: ReportLab not installed.")
                            except Exception as eg:
                                logger.error(f"Err generating PDF: {eg}", exc_info=True)
                                st.error(f"Report error: {eg}")
        logger.debug(f"ML tab rendered (type={model_type}).")
    except Exception as e:
        logger.error(f"Error rendering ML tab: {e}", exc_info=True)
        st.error("ML tab failed.")

# ----- NEW: Customer Segmentation -----
def render_customer_segmentation(transaction_df):
    """Renders the Customer Segmentation tab."""
    try:
        st.header("👥 Customer Segmentation (K-Means)")
        st.markdown("Group customers based on Recency, Frequency, and Monetary value (RFM) using K-Means clustering.")
        st.caption("*Concepts: Unsupervised Learning, Clustering (K-Means), Feature Engineering (RFM)*")
        if transaction_df is None or transaction_df.empty:
            st.warning("Transaction data needed for segmentation.")
            return
        default_k = 4
        k_clusters = st.slider("Select Number of Segments (k)", min_value=2, max_value=10, value=st.session_state.get('segmentation_k', default_k), key="k_means_k")
        if st.button("Run Segmentation Analysis", key="run_segmentation"):
            segmentation_instance = None
            try:
                with st.spinner(f"Running K-Means with k={k_clusters}..."):
                    logger.info(f"Running segmentation for k={k_clusters}")
                    segmentation_instance = CustomerSegmentation()
                    rfm_features = segmentation_instance.prepare_customer_features(transaction_df)
                    if rfm_features.empty:
                        raise ValueError("RFM feature preparation yielded empty results.")
                    segmentation_instance.segment_customers(k=k_clusters)
                    st.session_state.segmentation_model = segmentation_instance
                    st.session_state.segmented_customers = segmentation_instance.get_segmented_customers()
                    st.session_state.segment_centers = segmentation_instance.get_cluster_centers()
                    st.session_state.segmentation_k = k_clusters
                    st.success(f"Segmentation complete with {k_clusters} clusters.")
                    logger.info(f"Segmentation complete (k={k_clusters}).")
            except Exception as seg_e:
                logger.error(f"Error running customer segmentation: {seg_e}", exc_info=True)
                st.error(f"Segmentation analysis failed: {seg_e}")
                if 'segmented_customers' in st.session_state:
                    del st.session_state.segmented_customers
                if 'segment_centers' in st.session_state:
                    del st.session_state.segment_centers
        st.markdown("---")
        if 'segmented_customers' in st.session_state and not st.session_state.segmented_customers.empty:
            segmented_df = st.session_state.segmented_customers
            centers_df = st.session_state.get('segment_centers')
            current_k = st.session_state.get('segmentation_k', 'N/A')
            st.subheader(f"Segmentation Results (k={current_k})")
            st.markdown("**Segment Sizes**")
            segment_counts = segmented_df['Segment'].value_counts().sort_index()
            st.dataframe(segment_counts.reset_index().rename(columns={'index':'Segment ID', 'Segment':'Customer Count'}))
            if centers_df is not None and not centers_df.empty:
                st.markdown("**Segment Characteristics (Average RFM Values)**")
                st.caption("Lower Recency = More Recent. Higher Frequency/Monetary = More Active/Valuable.")
                formatted_centers = centers_df.copy()
                if 'Recency' in formatted_centers:
                    formatted_centers['Recency'] = formatted_centers['Recency'].map('{:.0f}'.format)
                if 'Frequency' in formatted_centers:
                    formatted_centers['Frequency'] = formatted_centers['Frequency'].map('{:.1f}'.format)
                if 'MonetaryValue' in formatted_centers:
                    formatted_centers['MonetaryValue'] = formatted_centers['MonetaryValue'].map('₹{:,.2f}'.format)
                st.dataframe(formatted_centers)
            st.subheader("Segment Visualization")
            plot_cols = st.multiselect("Select features to plot (2 or 3):",
                                       options=['Recency', 'Frequency', 'MonetaryValue'],
                                       default=['Recency', 'Frequency', 'MonetaryValue'],
                                       key='segment_plot_features')
            if 2 <= len(plot_cols) <= 3:
                if not all(col in segmented_df.columns for col in plot_cols):
                    st.warning(f"One or more selected columns for plotting not found in data: {plot_cols}")
                else:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 7))
                        if len(plot_cols) == 2:
                            sns.scatterplot(data=segmented_df, x=plot_cols[0], y=plot_cols[1],
                                            hue='Segment', palette='viridis', s=50, alpha=0.7, ax=ax, legend='full')
                            if centers_df is not None and all(col in centers_df.columns for col in plot_cols):
                                sns.scatterplot(data=centers_df, x=plot_cols[0], y=plot_cols[1],
                                                marker='X', s=250, color='red', label='Centroids', ax=ax, legend=False)
                            ax.set_title(f'{plot_cols[0]} vs {plot_cols[1]} by Segment')
                        else:
                            ax = fig.add_subplot(111, projection='3d')
                            scatter = ax.scatter(segmented_df[plot_cols[0]], segmented_df[plot_cols[1]], segmented_df[plot_cols[2]],
                                                 c=segmented_df['Segment'], cmap='viridis', s=30, alpha=0.6)
                            if centers_df is not None and all(col in centers_df.columns for col in plot_cols):
                                ax.scatter(centers_df[plot_cols[0]], centers_df[plot_cols[1]], centers_df[plot_cols[2]],
                                           marker='X', s=250, c='red', label='Centroids')
                                ax.legend(['Centroids'], loc='upper right', markerscale=1.5)
                            ax.set_xlabel(plot_cols[0])
                            ax.set_ylabel(plot_cols[1])
                            ax.set_zlabel(plot_cols[2])
                            ax.set_title('Customer Segments in 3D RFM Space')
                            legend_elements = scatter.legend_elements(num=current_k if isinstance(current_k, int) else None)
                            ax.legend(handles=legend_elements[0],
                                      labels=[f'Segment {i}' for i in range(current_k)] if isinstance(current_k, int) else [],
                                      title="Segments", loc='best')
                        ax.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig)
                    except ImportError as imp_err:
                        if 'mpl_toolkits' in str(imp_err) and len(plot_cols) == 3:
                            st.warning("3D plotting requires `mpl_toolkits.mplot3d`. Plotting in 2D instead if possible, or install required package.")
                        else:
                            logger.error(f"Import error during plotting: {imp_err}", exc_info=True)
                            st.warning("Could not generate plot due to missing library.")
                    except Exception as plot_e:
                        logger.error(f"Error plotting segments ({len(plot_cols)}D): {plot_e}", exc_info=True)
                        st.warning("Could not generate segment plot.")
            else:
                st.info("Select exactly 2 or 3 features to visualize.")
            st.subheader("Segmented Customer Data")
            st.dataframe(segmented_df.reset_index())
        else:
            st.info("Click 'Run Segmentation Analysis' to see results.")
        logger.debug("Customer segmentation tab rendered.")
    except Exception as e:
        logger.error(f"Error rendering customer segmentation tab: {e}", exc_info=True)
        st.error("Could not render Customer Segmentation tab.")

def render_about():
    try:
        st.header('ℹ️ About This Application')
        st.markdown("Demonstration of Big Data concepts for Inventory Management...")
        st.caption("*Technologies: Python, Streamlit, Pandas, Scikit-learn, etc.*")
        logger.debug("About rendered.")
    except Exception as e:
        logger.error(f"Error rendering about: {e}", exc_info=True)
        st.error("About tab failed.")

# --- Main Application Logic ---
def main():
    try:
        st.set_page_config(layout="wide", page_title="Inventory Management Demo", initial_sidebar_state="expanded")
        logger.info("Main function started.")
        dm = init_data()
        if dm is None:
            st.error("Data Manager failed.")
            st.stop()
        products_list = load_products_cached()
        transactions_df = get_transaction_df_cached()
        inventory_df = load_inventory_cached()
        products_df = pd.DataFrame(products_list)
        preload_ml_models()
        render_header()
        st.sidebar.title("Navigation")
        navigation_options = [
            "📈 Dashboard",
            "🛒 Market Basket",
            "🔴 Streaming",
            "🤝 Recommendations",
            "🤖 Machine Learning",
            "👥 Customer Segmentation",
            "ℹ️ About"
        ]
        selected_section = st.sidebar.radio("Go to", navigation_options, key="main_nav_sidebar")
        logger.info(f"Rendering section: {selected_section}")
        if selected_section == "📈 Dashboard":
            render_dashboard(dm)
        elif selected_section == "🛒 Market Basket":
            render_market_basket_analysis(dm, products_list)
        elif selected_section == "🔴 Streaming":
            render_streaming_data()
        elif selected_section == "🤝 Recommendations":
            render_recommendation_system(dm, transactions_df, products_list)
        elif selected_section == "🤖 Machine Learning":
            render_ml_tab(dm, transactions_df, inventory_df, products_df)
        elif selected_section == "👥 Customer Segmentation":
            render_customer_segmentation(transactions_df)
        elif selected_section == "ℹ️ About":
            render_about()
        logger.info(f"Finished rendering section: {selected_section}")
    except Exception as e:
        logger.critical(f"Critical error in main function: {e}", exc_info=True)
        st.error(f"An critical application error occurred: {e}. Check logs (`app_debug.log`).")

# --- Entry Point ---
if __name__ == "__main__":
    logger.info(f"Executing: {__file__} | Python: {sys.version.split()[0]} | WD: {os.getcwd()}")
    main()
    logger.info("-------------------- Application main() terminated --------------------")
