import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib  # To save objects for the API later


class CreditRiskPreprocessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_path = raw_data_path
        self.processed_path = processed_data_path
        self.df = None
        self.customer_df = None
        self.woe_bins = {}  # To store WoE mappings

    def load_data(self):
        """Loads and performs initial cleaning."""
        print(f"Loading data from {self.raw_path}...")
        self.df = pd.read_csv(self.raw_path, parse_dates=['TransactionStartTime'])

        # Extract date features immediately for aggregation
        self.df['TransactionHour'] = self.df['TransactionStartTime'].dt.hour
        self.df['TransactionMonth'] = self.df['TransactionStartTime'].dt.month
        return self.df

    def create_rfm_features(self):
        """
        Task 4: Create Proxy Target using RFM Analysis.
        Logic:
        - Recency: Days since last transaction
        - Frequency: Total count of transactions
        - Monetary: Total positive spend (Debits)
        """
        print("Creating RFM Features...")

        # 1. Snapshot date (use the max date in dataset + 1 day)
        snapshot_date = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        # 2. Filter for Monetary: Only look at Debits (Amount > 0) for spending power
        debits = self.df[self.df['Amount'] > 0]

        # 3. Aggregate
        # Recency (days since last transaction)
        recency = self.df.groupby('CustomerId')['TransactionStartTime'] \
            .agg(lambda x: (snapshot_date - x.max()).days)

        # Frequency (only positive/debit transactions)
        frequency = self.df[self.df['Amount'] > 0].groupby('CustomerId').size()

        # Monetary (total positive spend)
        monetary = self.df[self.df['Amount'] > 0].groupby('CustomerId')['Amount'].sum()

        # Merge into one dataframe
        rfm = pd.DataFrame({
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary
        }).fillna(0)



        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        self.customer_df = rfm
        return self.customer_df

    def engineer_target_variable(self):
        """
        Task 4 (Continued): Clustering to define 'High Risk'.
        High Risk = High Recency (Inactive), Low Frequency, Low Monetary.
        """
        print("Engineering Target Variable (Clustering)...")

        # Log transform to handle skew (The "Whales" issue from EDA)
        rfm_log = np.log1p(self.customer_df[['Recency', 'Frequency', 'Monetary']])

        # Scale
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)

        # KMeans Clustering (k=3: Good, Average, Bad)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(rfm_scaled)
        joblib.dump(scaler, "../models/rfm_scaler.pkl")
        joblib.dump(kmeans, "../models/rfm_kmeans.pkl")
        self.customer_df['Cluster'] = clusters

        # Identify the "Risk" cluster
        # We look for the cluster with the LOWEST mean Monetary value
        cluster_summary = self.customer_df.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        })

        # High risk = high Recency + low Frequency + low Monetary
        high_risk_cluster = cluster_summary.apply(
            lambda row: row['Recency'] - row['Frequency'] - row['Monetary'], axis=1
        ).idxmax()

        print(f"Cluster Summary (Monetary Means): \n{cluster_summary}")
        print(f"Identified High Risk Cluster: {high_risk_cluster}")

        # Create Target: 1 = High Risk, 0 = Low Risk
        self.customer_df['is_high_risk'] = (self.customer_df['Cluster'] == high_risk_cluster).astype(int)

        # Drop the cluster column now that we have the label
        self.customer_df.drop(columns=['Cluster'], inplace=True)

        # Validate imbalance
        risk_counts = self.customer_df['is_high_risk'].value_counts()
        print(f"Target Distribution:\n{risk_counts}")

    def create_aggregate_features(self):
        """
        Task 3: Feature Engineering (Aggregations).
        Create rich features for the model to learn from.
        """
        print("Creating Aggregate Features...")

        # 1. Product Category Counts (Pivot Table)
        # How many times did they buy 'airtime' vs 'utility'?
        cat_counts = self.df.pivot_table(
            index='CustomerId',
            columns='ProductCategory',
            values='TransactionId',
            aggfunc='count',
            fill_value=0
        )
        cat_counts.columns = [f"Cat_{c}" for c in cat_counts.columns]

        # 2. Channel Usage
        channel_counts = self.df.pivot_table(
            index='CustomerId',
            columns='ChannelId',
            values='TransactionId',
            aggfunc='count',
            fill_value=0
        )
        channel_counts.columns = [f"Ch_{c}" for c in channel_counts.columns]

        # 3. Time of Day stats
        time_stats = self.df.groupby('CustomerId')['TransactionHour'].agg(['mean', 'std']).fillna(0)
        time_stats.columns = ['Avg_Tx_Hour', 'Std_Tx_Hour']

        # Merge everything into customer_df
        self.customer_df = self.customer_df.join(cat_counts).join(channel_counts).join(time_stats)

        # 4. Channel Diversity (Entropy)
        def entropy(row):
            counts = row[row > 0]
            total = counts.sum()
            if total == 0:
                return 0
            p = counts / total
            return -np.sum(p * np.log(p))

        channel_cols = [c for c in self.customer_df.columns if c.startswith("Ch_")]
        self.customer_df["Channel_Diversity"] = self.customer_df[channel_cols].apply(entropy, axis=1)

        # 5. Product Category Diversity (Entropy)
        cat_cols = [c for c in self.customer_df.columns if c.startswith("Cat_")]
        self.customer_df["Category_Diversity"] = self.customer_df[cat_cols].apply(entropy, axis=1)

        # 6. Engagement Score (simple, interpretable)
        # Normalize R, F, M
        r_norm = (self.customer_df["Recency"] - self.customer_df["Recency"].min()) / (
                    self.customer_df["Recency"].max() - self.customer_df["Recency"].min())
        f_norm = (self.customer_df["Frequency"] - self.customer_df["Frequency"].min()) / (
                    self.customer_df["Frequency"].max() - self.customer_df["Frequency"].min())
        m_norm = (self.customer_df["Monetary"] - self.customer_df["Monetary"].min()) / (
                    self.customer_df["Monetary"].max() - self.customer_df["Monetary"].min())

        self.customer_df["Engagement_Score"] = f_norm + m_norm - r_norm

        # Fill any remaining NaNs
        self.customer_df.fillna(0, inplace=True)


    def calculate_woe_iv(self, feature, target):
        """
        Task 3 (Advanced): Weight of Evidence (WoE) Calculation.
        (Simplified version for demonstration).
        """
        # Binning for continuous variables would happen here
        # For now, we will stick to the aggregate features
        pass

    def run_pipeline(self):
        self.load_data()
        self.create_rfm_features()  # RFM -> Basis for Target
        self.engineer_target_variable()  # Clustering -> Target Label
        self.create_aggregate_features()  # Feature Engineering

        print(f"Saving processed data to {self.processed_path}...")
        self.customer_df.to_csv(self.processed_path)
        print("Pipeline Completed Successfully.")


if __name__ == "__main__":
    # Adjust paths as needed
    processor = CreditRiskPreprocessor(
        raw_data_path="../data/raw/data.csv",
        processed_data_path="../data/processed/customer_risk_data_final.csv"
    )
    processor.run_pipeline()