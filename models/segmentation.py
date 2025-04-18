import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
logger = logging.getLogger("inventory_app." + __name__)

class CustomerSegmentation:

    def __init__(self):
        self.rfm_df = None
        self.scaled_features = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.customer_segments = None
        self.cluster_centers = None
        self.selected_k = None
    def prepare_customer_features(self, transaction_df):

        logger.info("Preparing RFM features for customer segmentation")
        if transaction_df is None or transaction_df.empty:
            logger.warning("Transaction DataFrame is empty. Cannot calculate RFM features.")
            return pd.DataFrame()

        required_cols = {'customer_id', 'transaction_id', 'transaction_date', 'quantity', 'price_per_unit'}
        if not required_cols.issubset(transaction_df.columns):
            missing = required_cols - set(transaction_df.columns)
            logger.error(f"Transaction DataFrame missing required RFM columns: {missing}")
            raise ValueError(f"Transaction DataFrame missing required RFM columns: {missing}")

        df = transaction_df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price_per_unit'] = pd.to_numeric(df['price_per_unit'], errors='coerce')
        df = df.dropna(subset=['transaction_date', 'customer_id', 'transaction_id', 'quantity', 'price_per_unit'])

        if df.empty:
            logger.warning("No valid transaction data after cleaning for RFM.")
            return pd.DataFrame()
        df['TotalPrice'] = df['quantity'] * df['price_per_unit']
        snapshot_date = df['transaction_date'].max() + pd.Timedelta(days=1)
        logger.info(f"RFM Snapshot Date: {snapshot_date}")
        rfm = df.groupby('customer_id').agg({
            'transaction_date': lambda date: (snapshot_date - date.max()).days,
            'transaction_id': 'nunique',
            'TotalPrice': 'sum'
        })
        rfm.rename(columns={'transaction_date': 'Recency',
                            'transaction_id': 'Frequency',
                            'TotalPrice': 'MonetaryValue'}, inplace=True)
        rfm = rfm[rfm['Frequency'] > 0]

        logger.info(f"RFM features calculated for {len(rfm)} customers.")
        self.rfm_df = rfm
        return self.rfm_df

    def scale_features(self):
        if self.rfm_df is None or self.rfm_df.empty:
            raise ValueError("RFM features not prepared. Run prepare_customer_features first.")

        logger.info("Scaling RFM features")
        numeric_cols = self.rfm_df.select_dtypes(include=np.number).columns
        if numeric_cols.empty:
             raise ValueError("No numeric RFM features found to scale.")

        self.scaled_features = self.scaler.fit_transform(self.rfm_df[numeric_cols])
        logger.info("RFM features scaled.")
        return self.scaled_features

    def find_optimal_k(self, max_k=10, random_state=42):

        if self.scaled_features is None:
            self.scale_features()

        inertia = {}
        logger.info(f"Calculating inertia for k=1 to {max_k}")
        k_range = range(1, max_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                            random_state=random_state)
            kmeans.fit(self.scaled_features)
            inertia[k] = kmeans.inertia_
        logger.info("Inertia calculation complete.")
        return inertia

    def segment_customers(self, k, random_state=42):

        if self.scaled_features is None:
            self.scale_features()

        logger.info(f"Applying K-Means clustering with k={k}")
        self.selected_k = k
        self.kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                                   random_state=random_state)
        self.kmeans_model.fit(self.scaled_features)
        self.customer_segments = self.rfm_df.copy()
        self.customer_segments['Segment'] = self.kmeans_model.labels_
        scaled_centers = self.kmeans_model.cluster_centers_
        try:
            original_centers = self.scaler.inverse_transform(scaled_centers)
            self.cluster_centers = pd.DataFrame(original_centers, columns=self.rfm_df.select_dtypes(include=np.number).columns)
            self.cluster_centers.index.name = 'Segment'
            logger.info("Calculated cluster centers (original scale):")
            logger.info(f"\n{self.cluster_centers.to_string()}")
        except Exception as e:
             logger.error(f"Could not inverse transform cluster centers: {e}")
             self.cluster_centers = pd.DataFrame(scaled_centers, columns=self.rfm_df.select_dtypes(include=np.number).columns)


        logger.info(f"Customer segmentation complete. Assigned {k} segments.")
        return self.customer_segments

    def get_segmented_customers(self):
        if self.customer_segments is None:
            logger.warning("Segmentation not performed yet.")
            return pd.DataFrame()
        return self.customer_segments

    def get_cluster_centers(self):
        return self.cluster_centers