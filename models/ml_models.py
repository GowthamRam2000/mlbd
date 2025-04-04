import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # Still used for splitting logic structure, but not random split for time series
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import joblib
import os
import logging
logger = logging.getLogger("inventory_app." + __name__)
class InventoryPredictionModel:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.target = None
        self.model_path = model_path or os.path.join('models', 'inventory_prediction_model.joblib') # Default path
        if model_path and os.path.exists(model_path):
            try:
                self.load_model(model_path)
            except Exception as e:
                logger.error(f"Init: Error loading prediction model from {model_path}: {e}")
                self.model = None
    def prepare_data(self, transaction_df, inventory_df):
        logger.info("Starting data preparation for prediction model from DataFrames...")
        if transaction_df is None or transaction_df.empty:
             logger.warning("Transaction DataFrame is empty or None in prepare_data.")
             return pd.DataFrame(), pd.Series(dtype='float64')
        if inventory_df is None or inventory_df.empty:
             logger.warning("Inventory DataFrame is empty or None in prepare_data.")
             return pd.DataFrame(), pd.Series(dtype='float64')
        required_txn_cols = {'transaction_date', 'product_id', 'quantity'}
        required_inv_cols = {'product_id', 'quantity'}
        if not required_txn_cols.issubset(transaction_df.columns):
            raise ValueError(f"Transaction DF missing columns: {required_txn_cols - set(transaction_df.columns)}")
        if not required_inv_cols.issubset(inventory_df.columns):
            raise ValueError(f"Inventory DF missing columns: {required_inv_cols - set(inventory_df.columns)}")
        transaction_df = transaction_df.copy()
        inventory_df = inventory_df.copy()
        try:
            transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'], errors='coerce')
            transaction_df = transaction_df.dropna(subset=['transaction_date'])
            if transaction_df.empty: raise ValueError("No valid transaction dates found after conversion.")
        except Exception as e:
            logger.error(f"Error converting transaction_date: {e}")
            raise ValueError("Could not convert transaction_date to datetime.")

        transaction_df['product_id'] = transaction_df['product_id'].astype(str)
        inventory_df['product_id'] = inventory_df['product_id'].astype(str)
        inventory_df['quantity'] = pd.to_numeric(inventory_df['quantity'], errors='coerce').fillna(0).astype(int)
        transaction_df['quantity'] = pd.to_numeric(transaction_df['quantity'], errors='coerce').fillna(0).astype(int)
        transaction_df['day_of_week'] = transaction_df['transaction_date'].dt.dayofweek
        transaction_df['day_of_month'] = transaction_df['transaction_date'].dt.day
        transaction_df['month'] = transaction_df['transaction_date'].dt.month
        daily_sales = transaction_df.groupby(['product_id', pd.Grouper(key='transaction_date', freq='D')])['quantity'].sum().reset_index()
        products = transaction_df['product_id'].unique()
        min_date = transaction_df['transaction_date'].min()
        max_date = transaction_df['transaction_date'].max()
        if pd.isna(min_date) or pd.isna(max_date) or not products.any():
            logger.warning("Cannot determine date range or no products found.")
            return pd.DataFrame(), pd.Series(dtype='float64')
        date_range = pd.date_range(min_date, max_date, freq='D')
        product_dates_index = pd.MultiIndex.from_product([products, date_range], names=['product_id', 'transaction_date'])
        product_dates = pd.DataFrame(index=product_dates_index).reset_index()
        full_daily_sales = pd.merge(product_dates, daily_sales, on=['product_id', 'transaction_date'], how='left').fillna({'quantity': 0})
        full_daily_sales['day_of_week'] = full_daily_sales['transaction_date'].dt.dayofweek
        full_daily_sales['day_of_month'] = full_daily_sales['transaction_date'].dt.day
        full_daily_sales['month'] = full_daily_sales['transaction_date'].dt.month
        lag_cols_to_fill = []
        for lag in [1, 2, 3, 7]:
            lag_col = f'lag_{lag}'
            full_daily_sales[lag_col] = full_daily_sales.groupby('product_id')['quantity'].shift(lag)
            lag_cols_to_fill.append(lag_col)
        full_daily_sales[lag_cols_to_fill] = full_daily_sales[lag_cols_to_fill].fillna(0)
        roll_cols_to_fill = []
        for window in [3, 7, 14]:
            roll_col = f'rolling_mean_{window}'
            rolling_means = full_daily_sales.groupby('product_id')['quantity'] \
                                     .rolling(window=window, min_periods=1).mean() \
                                     .reset_index(level=0, drop=True)
            full_daily_sales[roll_col] = rolling_means
            roll_cols_to_fill.append(roll_col)
        full_daily_sales[roll_cols_to_fill] = full_daily_sales[roll_cols_to_fill].fillna(0)
        cat_cols = []
        if 'category' in transaction_df.columns:
            cat_map = transaction_df.drop_duplicates('product_id').set_index('product_id')['category'].to_dict()
            full_daily_sales['category'] = full_daily_sales['product_id'].map(cat_map).fillna('Unknown')
            cat_dummies = pd.get_dummies(full_daily_sales['category'], prefix='category', dummy_na=False)
            cat_cols = cat_dummies.columns.tolist()
            full_daily_sales = pd.concat([full_daily_sales, cat_dummies], axis=1)
            full_daily_sales = full_daily_sales.drop('category', axis=1)

        inventory_map = inventory_df.drop_duplicates('product_id').set_index('product_id')['quantity'].to_dict()
        full_daily_sales['current_inventory'] = full_daily_sales['product_id'].map(inventory_map).fillna(0)

        full_daily_sales['next_day_sales'] = full_daily_sales.groupby('product_id')['quantity'].shift(-1)
        full_daily_sales = full_daily_sales.dropna(subset=['next_day_sales'])
        if full_daily_sales.empty:
             logger.warning("No samples remaining after calculating next_day_sales target.")
             return pd.DataFrame(), pd.Series(dtype='float64')
        full_daily_sales['next_day_sales'] = pd.to_numeric(full_daily_sales['next_day_sales'])

        base_feats = ['day_of_week', 'day_of_month', 'month', 'current_inventory']
        lag_feats = [c for c in full_daily_sales if isinstance(c, str) and c.startswith('lag_')]
        roll_feats = [c for c in full_daily_sales if isinstance(c, str) and c.startswith('roll')]
        self.features = base_feats + lag_feats + roll_feats + cat_cols
        self.target = 'next_day_sales'
        self.features = [f for f in self.features if f in full_daily_sales.columns]
        if not self.features: raise ValueError("No valid features generated.")
        X = full_daily_sales[self.features + ['product_id', 'transaction_date']]
        y = full_daily_sales[self.target]
        if len(X) != len(y): raise ValueError(f"X/y length mismatch after prep: X({len(X)})!=y({len(y)})")
        logger.info(f"Data preparation complete. X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Features prepared ({len(self.features)}): {self.features}")
        return X, y
    def train(self, X, y, test_size=0.2):
        logger.info("Starting model training with CHRONOLOGICAL SPLIT...")
        if self.features is None or self.target is None: raise ValueError("Features/target not set.")
        if 'transaction_date' not in X.columns: raise ValueError("Input X must include 'transaction_date'.")
        if X.empty or y.empty: raise ValueError("Input X or y is empty.")
        X_sorted = X.sort_values(by=['transaction_date'])
        y_sorted = y.loc[X_sorted.index]
        n_samples = len(X_sorted)
        split_index = int(n_samples * (1 - test_size))
        if not (0 < split_index < n_samples): raise ValueError("test_size yields empty train/test set.")
        X_train_df_full = X_sorted.iloc[:split_index]
        X_test_df_full = X_sorted.iloc[split_index:]
        y_train = y_sorted.iloc[:split_index]
        y_test = y_sorted.iloc[split_index:]
        logger.info(f"Chronological split: Train samples={len(X_train_df_full)}, Test samples={len(X_test_df_full)}")
        logger.info(f"Train date range: {X_train_df_full['transaction_date'].min()} to {X_train_df_full['transaction_date'].max()}")
        logger.info(f"Test date range: {X_test_df_full['transaction_date'].min()} to {X_test_df_full['transaction_date'].max()}")
        product_ids_test = X_test_df_full['product_id'].copy()
        dates_test = X_test_df_full['transaction_date'].copy()
        X_train = X_train_df_full[self.features]
        X_test = X_test_df_full[self.features]
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info("Features scaled (fitted on train set).")
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        logger.info("Fitting RandomForestRegressor...")
        self.model.fit(X_train_scaled, y_train)
        logger.info("Model fitting complete.")

        logger.info("Evaluating model on test set...")
        y_pred = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = -np.inf
        try:
             if np.var(y_test) > 1e-9:
                 r2 = r2_score(y_test, y_pred)
             elif np.mean((y_test - y_pred)**2) < 1e-9:
                 r2 = 1.0
             else:
                 r2 = 0.0
                 logger.warning("R2 score calculation: Test target variable might be constant.")
        except Exception as r2_err: logger.error(f"Could not calculate R2 score: {r2_err}")

        logger.info(f"Evaluation Results (Chronological Split): RMSE={rmse:.4f}, R2={r2:.4f}")

        feature_importance_df = pd.DataFrame()
        try:
            importances = self.model.feature_importances_
            if len(importances) == len(self.features):
                feature_importance_df = pd.DataFrame({'feature': self.features, 'importance': importances}).sort_values('importance', ascending=False)
            else: logger.error(f"Feature importance length mismatch.")
        except Exception as e: logger.error(f"Error creating feature importance DF: {e}", exc_info=True)

        test_predictions_df = pd.DataFrame()
        try:
            y_test_values = y_test.values if isinstance(y_test, pd.Series) else y_test
            pred_data = {'actual': y_test_values, 'predicted': y_pred,
                         'product_id': product_ids_test.values if isinstance(product_ids_test, pd.Series) else product_ids_test,
                         'date': dates_test.values if isinstance(dates_test, pd.Series) else dates_test}
            test_predictions_df = pd.DataFrame(pred_data)
        except Exception as e: logger.error(f"Error creating test predictions DF: {e}", exc_info=True)

        metrics = {'rmse': rmse, 'r2': r2, 'feature_importance': feature_importance_df, 'test_predictions': test_predictions_df}
        logger.info("Training process completed.")
        return metrics
    def predict_future_needs(self, X_latest_state, days_ahead=7):
        logger.info(f"Starting future prediction for {days_ahead} days...")
        if self.model is None or self.scaler is None or self.features is None:
            raise ValueError("Model/scaler/features not available. Train or load first.")
        if X_latest_state.empty:
            logger.warning("Input X_latest_state is empty.")
            return pd.DataFrame(columns=['product_id', 'date', 'predicted_sales'])
        if 'product_id' not in X_latest_state.columns: raise ValueError("'product_id' missing.")
        if not set(self.features).issubset(set(X_latest_state.columns)):
            missing = set(self.features) - set(X_latest_state.columns); raise ValueError(f"Missing features: {missing}")

        last_known_date = datetime.now().date() # Prediction starts from tomorrow
        logger.info(f"Predicting starting from day after: {last_known_date}")

        all_preds = []
        current_feats_df = X_latest_state.set_index('product_id')[self.features].copy()

        for day in range(1, days_ahead + 1):
            pred_date = pd.to_datetime(last_known_date + timedelta(days=day))

            current_feats_df['day_of_week'] = pred_date.dayofweek
            current_feats_df['day_of_month'] = pred_date.day
            current_feats_df['month'] = pred_date.month

            # Check for NaNs before scaling
            if current_feats_df[self.features].isnull().values.any():
                 logger.warning(f"NaN values found in features before scaling on prediction day {day}. Filling with 0.")
                 current_feats_df = current_feats_df.fillna(0)

            feats_to_scale = current_feats_df[self.features]
            feats_scaled = self.scaler.transform(feats_to_scale) # Use transform

            pred_sales = self.model.predict(feats_scaled)
            pred_sales = np.maximum(0, pred_sales) # Ensure non-negative

            day_preds = pd.DataFrame({'product_id': current_feats_df.index, 'date': pred_date, 'predicted_sales': pred_sales})
            all_preds.append(day_preds)

            # --- Update features for the *next* day's prediction ---
            pred_sales_map = day_preds.set_index('product_id')['predicted_sales'].to_dict()
            lag_cols_sorted = sorted([f for f in self.features if f.startswith('lag_')], key=lambda x: int(x.split('_')[-1]))
            if lag_cols_sorted:
                 for i in range(len(lag_cols_sorted) - 1, 0, -1): current_feats_df[lag_cols_sorted[i]] = current_feats_df[lag_cols_sorted[i-1]]
                 current_feats_df[lag_cols_sorted[0]] = current_feats_df.index.map(pred_sales_map).fillna(0)
            for window in [3, 7, 14]:
                roll_col = f'rolling_mean_{window}'
                if roll_col in self.features:
                    new_sales = current_feats_df.index.map(pred_sales_map).fillna(0)
                    current_mean = current_feats_df[roll_col].fillna(0)
                    # Avoid division by zero for window=1 case (though unlikely here)
                    if window > 0:
                       current_feats_df[roll_col] = ((current_mean * (window - 1)) + new_sales) / window
                    else:
                       current_feats_df[roll_col] = new_sales # Should not happen with window > 0

        if not all_preds: return pd.DataFrame(columns=['product_id', 'date', 'predicted_sales'])
        all_preds_df = pd.concat(all_preds, ignore_index=True)
        logger.info(f"Future prediction complete: Generated {len(all_preds_df)} predictions.")
        return all_preds_df

    def identify_low_stock_items(self, inventory_df, predictions_df, threshold_days=3):
        """Identifies low stock items using inventory and prediction DataFrames."""
        logger.info(f"Identifying low stock items (threshold: {threshold_days} days)...")
        if inventory_df is None or inventory_df.empty: logger.warning("Inventory DF empty."); return pd.DataFrame(columns=['product_id', 'current_inventory', 'avg_daily_sales', 'days_until_stockout', 'stockout_date'])
        if predictions_df is None or predictions_df.empty: logger.warning("Predictions DF empty."); return pd.DataFrame(columns=['product_id', 'current_inventory', 'avg_daily_sales', 'days_until_stockout', 'stockout_date'])
        if not {'product_id', 'quantity'}.issubset(inventory_df.columns): raise ValueError("Inventory DF missing 'product_id' or 'quantity'.")
        if not {'product_id', 'date', 'predicted_sales'}.issubset(predictions_df.columns): raise ValueError("Predictions DF missing 'product_id', 'date', or 'predicted_sales'.")

        inventory_df = inventory_df.copy(); predictions_df = predictions_df.copy()
        inventory_df['product_id'] = inventory_df['product_id'].astype(str)
        inventory_df['quantity'] = pd.to_numeric(inventory_df['quantity'], errors='coerce').fillna(0).astype(int)
        predictions_df['product_id'] = predictions_df['product_id'].astype(str)
        predictions_df['predicted_sales'] = pd.to_numeric(predictions_df['predicted_sales'], errors='coerce').fillna(0)
        predictions_df['date'] = pd.to_datetime(predictions_df['date'], errors='coerce')
        predictions_df = predictions_df.dropna(subset=['date'])
        if predictions_df.empty: logger.warning("No valid dates in predictions DF."); return pd.DataFrame(columns=['product_id', 'current_inventory', 'avg_daily_sales', 'days_until_stockout', 'stockout_date'])

        try: pivot_preds = predictions_df.pivot_table(index='product_id', columns=predictions_df['date'].dt.date, values='predicted_sales', aggfunc='sum').fillna(0)
        except Exception as e: logger.error(f"Error pivoting predictions: {e}", exc_info=True); return pd.DataFrame()
        pivot_preds = pivot_preds.sort_index(axis=1); cumulative_sales = pivot_preds.cumsum(axis=1)
        current_inventory = inventory_df.drop_duplicates('product_id').set_index('product_id')['quantity']
        inventory_levels = current_inventory.to_frame(name='quantity').join(cumulative_sales, how='inner')
        if inventory_levels.empty: logger.warning("No common products in inventory/predictions."); return pd.DataFrame(columns=['product_id', 'current_inventory', 'avg_daily_sales','days_until_stockout', 'stockout_date'])

        stockout_info = []; predicted_dates = pivot_preds.columns
        for product_id, row in inventory_levels.iterrows():
            current_stock = row['quantity']; days_to_stockout = -1; stockout_date = None
            for i, date in enumerate(predicted_dates):
                if current_stock < row[date]: # Compare stock to cumulative sales needed by 'date'
                    days_to_stockout = i + 1; stockout_date = pd.to_datetime(date); break
            if days_to_stockout != -1:
                avg_sales = row[predicted_dates].mean() # Avg predicted sales over forecast window
                stockout_info.append({'product_id': product_id, 'current_inventory': current_stock, 'avg_daily_sales': avg_sales, 'days_until_stockout': days_to_stockout, 'stockout_date': stockout_date})

        if not stockout_info: logger.info("No items predicted to stock out."); return pd.DataFrame(columns=['product_id', 'current_inventory', 'avg_daily_sales', 'days_until_stockout', 'stockout_date'])

        low_stock_df = pd.DataFrame(stockout_info)
        low_stock_filtered_df = low_stock_df[low_stock_df['days_until_stockout'] <= threshold_days].copy().sort_values('days_until_stockout')
        logger.info(f"Found {len(low_stock_filtered_df)} items predicted low stock within {threshold_days} days.")
        return low_stock_filtered_df

    def save_model(self, path=None):
        """Save the trained model, scaler, and features."""
        if self.model is None or self.scaler is None or self.features is None: raise ValueError("Model/scaler/features not available.")
        save_path = path or self.model_path; os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_data = {'model': self.model, 'scaler': self.scaler, 'features': self.features, 'target': self.target}
        try: joblib.dump(model_data, save_path); logger.info(f"InventoryPredictionModel saved to {save_path}"); return save_path
        except Exception as e: logger.error(f"Error saving prediction model: {e}", exc_info=True); raise

    def load_model(self, path):
        """Load a trained model, scaler, and features."""
        if not os.path.exists(path): raise FileNotFoundError(f"Prediction model file not found: {path}")
        try:
            model_data = joblib.load(path); required = ['model', 'scaler', 'features', 'target']
            if not all(k in model_data for k in required): raise ValueError(f"Model file {path} missing: {set(required) - set(model_data.keys())}")
            self.model = model_data['model']; self.scaler = model_data['scaler']; self.features = model_data['features']; self.target = model_data['target']; self.model_path = path
            logger.info(f"InventoryPredictionModel loaded from {path}"); logger.info(f"Loaded features ({len(self.features)}): {self.features}")
            return self
        except Exception as e: logger.error(f"Error loading prediction model: {e}", exc_info=True); raise

# ==============================================================================
# Spoilage Detection Model (Accepts DataFrames)
# ==============================================================================
class SpoilageDetectionModel:
    """Anomaly detection model accepting DataFrames for spoilage/obsolescence detection."""
    def __init__(self, model_path=None):
        """Initialize the spoilage detection model."""
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.model_path = model_path or os.path.join('models', 'spoilage_detection_model.joblib') # Default path
        if model_path and os.path.exists(model_path):
            try: self.load_model(model_path)
            except Exception as e: logger.error(f"Init: Error loading spoilage model from {model_path}: {e}"); self.model = None

    def prepare_data(self, transaction_df, inventory_df, product_info_df=None):
        """Prepares features for spoilage detection from DataFrames."""
        logger.info("Starting data preparation for SpoilageDetectionModel from DataFrames...")
        if transaction_df is None or transaction_df.empty: raise ValueError("Transaction DF empty/None.")
        if inventory_df is None or inventory_df.empty: raise ValueError("Inventory DF empty/None.")

        required_txn_cols = {'transaction_date', 'product_id', 'quantity'}; required_inv_cols = {'product_id', 'quantity'}
        if not required_txn_cols.issubset(transaction_df.columns): raise ValueError(f"Spoilage Txn DF missing: {required_txn_cols - set(transaction_df.columns)}")
        if not required_inv_cols.issubset(inventory_df.columns): raise ValueError(f"Spoilage Inv DF missing: {required_inv_cols - set(inventory_df.columns)}")
        if product_info_df is not None and not product_info_df.empty and 'product_id' not in product_info_df.columns: raise ValueError("Product Info DF missing 'product_id'.")

        transaction_df = transaction_df.copy(); inventory_df = inventory_df.copy()
        if product_info_df is not None: product_info_df = product_info_df.copy()

        # --- Type Conversions and Cleaning ---
        try: transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'], errors='coerce'); transaction_df = transaction_df.dropna(subset=['transaction_date'])
        except: raise ValueError("Could not convert transaction_date.")
        transaction_df['product_id'] = transaction_df['product_id'].astype(str)
        inventory_df['product_id'] = inventory_df['product_id'].astype(str)
        if product_info_df is not None: product_info_df['product_id'] = product_info_df['product_id'].astype(str)
        inventory_df['quantity'] = pd.to_numeric(inventory_df['quantity'], errors='coerce').fillna(0).astype(int)
        transaction_df['quantity'] = pd.to_numeric(transaction_df['quantity'], errors='coerce').fillna(0).astype(int)
        if 'price_per_unit' in transaction_df.columns: transaction_df['price_per_unit'] = pd.to_numeric(transaction_df['price_per_unit'], errors='coerce')

        # --- Feature Engineering ---
        today = pd.to_datetime(datetime.now().date())
        last_trans_date = transaction_df.groupby('product_id')['transaction_date'].max()
        days_since_last_transaction = (today - last_trans_date).dt.days
        sales_sum = transaction_df.groupby('product_id')['quantity'].sum()
        sales_days = transaction_df.groupby('product_id')['transaction_date'].apply(lambda x: x.dt.date.nunique())
        sales_velocity = (sales_sum / sales_days.clip(lower=1)).fillna(0)
        daily_qty = transaction_df.groupby(['product_id', pd.Grouper(key='transaction_date', freq='D')])['quantity'].sum()
        sales_consistency = daily_qty.groupby('product_id').std().fillna(0)
        current_inv = inventory_df.drop_duplicates('product_id').set_index('product_id')['quantity']
        inventory_turnover = (sales_sum / current_inv.clip(lower=1)).fillna(0)
        first_trans_date = transaction_df.groupby('product_id')['transaction_date'].min()
        inventory_age = (today - first_trans_date).dt.days
        price_changes = None
        if 'price_per_unit' in transaction_df.columns and transaction_df['price_per_unit'].notna().any():
            daily_avg_price = transaction_df.dropna(subset=['price_per_unit']).groupby(['product_id', pd.Grouper(key='transaction_date', freq='D')])['price_per_unit'].mean()
            price_changes = daily_avg_price.groupby('product_id').std().fillna(0)

        all_product_ids = inventory_df['product_id'].unique()
        features_df = pd.DataFrame(index=all_product_ids); features_df.index.name = 'product_id'
        features_df['days_since_last_transaction'] = features_df.index.map(days_since_last_transaction)
        features_df['sales_velocity'] = features_df.index.map(sales_velocity); features_df['sales_consistency'] = features_df.index.map(sales_consistency)
        features_df['inventory_turnover'] = features_df.index.map(inventory_turnover); features_df['inventory_age'] = features_df.index.map(inventory_age)
        features_df['current_inventory'] = features_df.index.map(current_inv)
        if price_changes is not None: features_df['price_changes'] = features_df.index.map(price_changes)

        cat_cols = []; category_source = None
        if product_info_df is not None and not product_info_df.empty and 'category' in product_info_df.columns:
             category_source = product_info_df.drop_duplicates('product_id').set_index('product_id')['category']; logger.info("Using category from Product Info for spoilage.")
        elif 'category' in transaction_df.columns:
             logger.warning("Using category from Transactions for spoilage (Product Info missing category)."); category_source = transaction_df.drop_duplicates('product_id').set_index('product_id')['category']
        if category_source is not None:
            features_df['category'] = features_df.index.map(category_source).fillna('Unknown')
            cat_dummies = pd.get_dummies(features_df['category'], prefix='category', dummy_na=False); cat_cols = cat_dummies.columns.tolist()
            features_df = pd.concat([features_df.drop('category', axis=1), cat_dummies], axis=1)

        features_df = features_df.fillna(0)
        for col in features_df.columns: # Ensure numeric
            if not pd.api.types.is_numeric_dtype(features_df[col]): features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

        self.features = features_df.columns.tolist()
        logger.info(f"Spoilage features prepared ({len(self.features)}). Shape: {features_df.shape}")
        logger.info(f"Features prepared: {self.features}")
        return features_df

    def train(self, features_df, contamination='auto'):
        """Trains the IsolationForest model using prepared features."""
        logger.info("Starting spoilage model training...")
        if features_df.empty: raise ValueError("Input features_df is empty.")
        if features_df.isnull().values.any(): logger.warning("Features contain NaN before scaling."); features_df = features_df.fillna(0)
        if self.features is None: self.features = features_df.columns.tolist(); logger.warning("Features inferred from input.")

        training_features = features_df[self.features] # Select only defined features
        logger.info("Scaling features for IsolationForest...");
        self.scaler = StandardScaler() # Re-initialize scaler
        X_scaled = self.scaler.fit_transform(training_features); logger.info("Features scaled.")
        self.model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
        logger.info(f"Fitting IsolationForest (contamination={contamination})..."); self.model.fit(X_scaled); logger.info("Fit complete.")
        return self

    def predict_spoilage(self, features_df):
        """Predicts anomaly scores and flags using the trained model."""
        logger.info("Predicting potential spoilage...")
        if self.model is None or self.scaler is None or self.features is None: raise ValueError("Model/scaler/features not available.")
        if features_df.empty: logger.warning("Input features_df empty."); return pd.DataFrame(columns=['product_id', 'anomaly_score', 'is_anomaly'])
        if not set(self.features).issubset(set(features_df.columns)): raise ValueError(f"Input missing features: {set(self.features)-set(features_df.columns)}")

        predict_features = features_df[self.features].fillna(0) # Fill NaNs just in case
        if predict_features.isnull().values.any(): logger.warning("NaNs found even after fillna(0).")

        logger.info("Scaling features for prediction..."); X_scaled = self.scaler.transform(predict_features); logger.info("Features scaled.")
        logger.info("Calculating anomaly scores..."); anomaly_scores = -self.model.decision_function(X_scaled); logger.info("Scores calculated.")
        logger.info("Making binary anomaly predictions..."); predictions = self.model.predict(X_scaled); logger.info("Binary predictions made.")
        results = pd.DataFrame({'product_id': features_df.index, 'anomaly_score': anomaly_scores, 'is_anomaly': predictions == -1}).sort_values('anomaly_score', ascending=False)
        logger.info(f"Spoilage prediction complete. Found {results['is_anomaly'].sum()} potential anomalies (before threshold).")
        return results

    def identify_spoilt_items(self, transaction_df, inventory_df, products_df, threshold=None):
        """End-to-end: Prepares data, predicts anomalies, and enriches results."""
        logger.info("Identifying potentially spoilt items (end-to-end)...")
        if self.model is None or self.scaler is None or self.features is None:
             if self.model_path and os.path.exists(self.model_path): self.load_model(self.model_path)
             else: raise ValueError("Spoilage model must be trained or loaded first.")

        features = self.prepare_data(transaction_df, inventory_df, products_df)
        if features.empty: logger.warning("Feature prep empty."); return pd.DataFrame()
        predictions = self.predict_spoilage(features)
        if threshold is not None: logger.info(f"Applying custom threshold: {threshold}"); predictions['is_anomaly'] = predictions['anomaly_score'] > threshold
        spoilt_items = predictions[predictions['is_anomaly']].copy()
        logger.info(f"Filtered to {len(spoilt_items)} items based on prediction/threshold.")

        if not spoilt_items.empty:
            # Enrich results safely
            if products_df is not None and not products_df.empty:
                products_df_copy = products_df.copy(); products_df_copy['product_id'] = products_df_copy['product_id'].astype(str)
                product_map = products_df_copy.drop_duplicates('product_id').set_index('product_id')
                spoilt_items['name'] = spoilt_items['product_id'].map(product_map.get('name'))
                spoilt_items['category'] = spoilt_items['product_id'].map(product_map.get('category'))
            else: spoilt_items['name'] = None; spoilt_items['category'] = None

            if inventory_df is not None and not inventory_df.empty:
                 inventory_df_copy = inventory_df.copy(); inventory_df_copy['product_id'] = inventory_df_copy['product_id'].astype(str)
                 inv_map = inventory_df_copy.drop_duplicates('product_id').set_index('product_id')['quantity'].fillna(0).astype(int)
                 spoilt_items['current_inventory'] = spoilt_items['product_id'].map(inv_map).fillna(0)
            else: spoilt_items['current_inventory'] = 0

            if 'days_since_last_transaction' in features.columns: spoilt_items['days_since_last_transaction'] = spoilt_items['product_id'].map(features['days_since_last_transaction']).fillna(-1)
            else: spoilt_items['days_since_last_transaction'] = -1

            spoilt_items['name'] = spoilt_items['name'].fillna(spoilt_items['product_id'].apply(lambda x: f"Product {x}"))
            spoilt_items['category'] = spoilt_items['category'].fillna('Unknown')
            spoilt_items['days_since_last_transaction'] = spoilt_items['days_since_last_transaction'].fillna(-1)

            final_cols = ['product_id', 'anomaly_score', 'is_anomaly', 'name', 'category', 'current_inventory', 'days_since_last_transaction']
            spoilt_items = spoilt_items.reindex(columns=final_cols) # Ensure all columns exist in order

        logger.info(f"Returning details for {len(spoilt_items)} potentially spoilt items.")
        return spoilt_items

    def save_model(self, path=None):
        """Save the trained spoilage model, scaler, and features."""
        if self.model is None or self.scaler is None or self.features is None: raise ValueError("Model/scaler/features not available.")
        save_path = path or self.model_path; os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model_data = {'model': self.model, 'scaler': self.scaler, 'features': self.features}
        try: joblib.dump(model_data, save_path); logger.info(f"SpoilageDetectionModel saved to {save_path}"); return save_path
        except Exception as e: logger.error(f"Error saving spoilage model: {e}", exc_info=True); raise

    def load_model(self, path):
        """Load a trained spoilage model, scaler, and features."""
        if not os.path.exists(path): raise FileNotFoundError(f"Spoilage model file not found: {path}")
        try:
            model_data = joblib.load(path); required = ['model', 'scaler', 'features']
            if not all(k in model_data for k in required): raise ValueError(f"Spoilage model file {path} missing: {set(required) - set(model_data.keys())}")
            self.model = model_data['model']; self.scaler = model_data['scaler']; self.features = model_data['features']; self.model_path = path
            logger.info(f"SpoilageDetectionModel loaded from {path}"); logger.info(f"Loaded features ({len(self.features)}): {self.features}")
            return self
        except Exception as e: logger.error(f"Error loading spoilage model: {e}", exc_info=True); raise


# ==============================================================================
# PDF Report Generation Function (Accepts DataFrames)
# ==============================================================================
def generate_inventory_report(low_stock_items, spoilt_items, products_df=None, output_path=None):
    """Generate a PDF report summarizing low stock and potentially spoilt items."""
    try:
        from reportlab.lib.pagesizes import letter, landscape; from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle; from reportlab.lib.units import inch
        logger.info("Generating inventory PDF report...")
    except ImportError: logger.error("ReportLab not found."); raise ImportError("ReportLab not installed.")

    if output_path is None:
        output_dir = 'reports'; os.makedirs(output_dir, exist_ok=True); timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'inventory_report_{timestamp}.pdf')

    doc = SimpleDocTemplate(output_path, pagesize=landscape(letter)); styles = getSampleStyleSheet(); story = []
    title_style=ParagraphStyle('ReportTitle', parent=styles['h1'], fontSize=16, spaceAfter=12, alignment=1)
    subtitle_style=ParagraphStyle('SectionTitle', parent=styles['h2'], fontSize=14, spaceAfter=10)
    normal_style=styles['Normal']; th_style=ParagraphStyle('TableHeader', parent=normal_style, fontName='Helvetica-Bold', alignment=1)
    tc_style=ParagraphStyle('TableCell', parent=normal_style, fontSize=8); tc_right_style=ParagraphStyle('TableCellRight', parent=tc_style, alignment=2)
    story.append(Paragraph("Inventory Management Report", title_style)); story.append(Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", normal_style)); story.append(Spacer(1, 0.2*inch))

    product_names = {}; product_cats = {}
    if products_df is not None and not products_df.empty:
        try:
             products_df = products_df.copy(); products_df['product_id'] = products_df['product_id'].astype(str)
             prod_info = products_df.drop_duplicates('product_id').set_index('product_id')
             if 'name' in prod_info.columns: product_names = prod_info['name'].to_dict()
             if 'category' in prod_info.columns: product_cats = prod_info['category'].to_dict()
             logger.info(f"Created lookup for {len(product_names)} products.")
        except Exception as e: logger.warning(f"Could not create product lookup: {e}")

    # --- Low Stock Items Section ---
    story.append(Paragraph("Low Stock Items (Predicted Stockout Alert)", subtitle_style))
    if low_stock_items is not None and not low_stock_items.empty:
        req_cols = {'product_id', 'current_inventory', 'avg_daily_sales', 'days_until_stockout', 'stockout_date'};
        if not req_cols.issubset(low_stock_items.columns):
            missing = req_cols - set(low_stock_items.columns); story.append(Paragraph(f"Error: Low stock data missing: {missing}.", normal_style)); logger.error(f"Low stock DF missing: {missing}")
        else:
            data = [[Paragraph(h, th_style) for h in ['ID', 'Product Name', 'Inv.', 'Avg Sales', 'Days Out', 'Date Out']]]
            low_stock_table_df = low_stock_items.copy(); low_stock_table_df['product_id'] = low_stock_table_df['product_id'].astype(str)
            for _, r in low_stock_table_df.iterrows():
                pid_str = r['product_id']; name = r.get('product_name', product_names.get(pid_str, f"P {pid_str}"))
                inv = f"{pd.to_numeric(r['current_inventory'], errors='coerce'):,.0f}" if pd.notna(r['current_inventory']) else 'N/A'
                avg_sales = f"{pd.to_numeric(r['avg_daily_sales'], errors='coerce'):,.1f}" if pd.notna(r['avg_daily_sales']) else 'N/A'
                days_out = f"{pd.to_numeric(r['days_until_stockout'], errors='coerce'):,.0f}" if pd.notna(r['days_until_stockout']) else 'N/A'
                date_out = r['stockout_date'].strftime('%Y-%m-%d') if pd.notna(r['stockout_date']) else 'N/A'
                data.append([Paragraph(pid_str, tc_style), Paragraph(str(name), tc_style), Paragraph(inv, tc_right_style), Paragraph(avg_sales, tc_right_style), Paragraph(days_out, tc_right_style), Paragraph(date_out, tc_style)])
            tbl = Table(data, repeatRows=1, colWidths=[0.8*inch, 3.5*inch, 0.8*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            tbl.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.darkblue), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 1, colors.black), ('PADDING', (0,0), (-1,-1), 4), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('BACKGROUND', (0,1), (-1,-1), colors.aliceblue), *[(('BACKGROUND', (0,i), (-1,i), colors.lightpink) if pd.to_numeric(low_stock_table_df.iloc[i-1]['days_until_stockout'], errors='coerce') <= 3 else {}) for i in range(1, len(data))]]))
            story.append(tbl)
    else: story.append(Paragraph("No low stock items identified/provided.", normal_style))
    story.append(Spacer(1, 0.2*inch))

    # --- Potentially Spoilt Items Section ---
    story.append(Paragraph("Potentially Spoilt/Obsolete Items (Anomaly Detection)", subtitle_style))
    if spoilt_items is not None and not spoilt_items.empty:
        req_cols = {'product_id', 'anomaly_score', 'current_inventory', 'days_since_last_transaction'};
        if not req_cols.issubset(spoilt_items.columns):
            missing = req_cols - set(spoilt_items.columns); story.append(Paragraph(f"Error: Spoilt items data missing: {missing}.", normal_style)); logger.error(f"Spoilt items DF missing: {missing}")
        else:
            data = [[Paragraph(h, th_style) for h in ['ID', 'Product Name', 'Category', 'Inv.', 'Days Since Sale', 'Anomaly Score']]]
            spoilt_table_df = spoilt_items.copy(); spoilt_table_df['product_id'] = spoilt_table_df['product_id'].astype(str)
            for _, r in spoilt_table_df.iterrows():
                 pid_str = r['product_id']; name = r.get('name', product_names.get(pid_str, f"P {pid_str}")); category = r.get('category', product_cats.get(pid_str, 'Unknown'))
                 inv = f"{pd.to_numeric(r['current_inventory'], errors='coerce'):,.0f}" if pd.notna(r['current_inventory']) else 'N/A'
                 days_since = f"{pd.to_numeric(r['days_since_last_transaction'], errors='coerce'):,.0f}" if pd.notna(r['days_since_last_transaction']) and r['days_since_last_transaction'] != -1 else 'N/A'
                 score = f"{pd.to_numeric(r['anomaly_score'], errors='coerce'):,.4f}" if pd.notna(r['anomaly_score']) else 'N/A'
                 data.append([Paragraph(pid_str, tc_style), Paragraph(str(name), tc_style), Paragraph(str(category), tc_style), Paragraph(inv, tc_right_style), Paragraph(days_since, tc_right_style), Paragraph(score, tc_right_style)])
            tbl = Table(data, repeatRows=1, colWidths=[0.8*inch, 3.5*inch, 1.5*inch, 0.8*inch, 1.2*inch, 1.2*inch])
            tbl.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.darkred), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('GRID', (0,0), (-1,-1), 1, colors.black), ('PADDING', (0,0), (-1,-1), 4), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('BACKGROUND', (0,1), (-1,-1), colors.seashell), *[(('BACKGROUND', (0,i), (-1,i), colors.salmon) if pd.to_numeric(spoilt_table_df.iloc[i-1]['anomaly_score'], errors='coerce') > 0.6 else {}) for i in range(1, len(data))]]))
            story.append(tbl)
    else: story.append(Paragraph("No potentially spoilt items identified/provided.", normal_style))

    try: doc.build(story); logger.info(f"Inventory report generated: {output_path}"); return output_path
    except Exception as e: logger.error(f"Error building PDF '{output_path}': {e}", exc_info=True); raise