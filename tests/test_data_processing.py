import pytest
import pandas as pd
import numpy as np
from src.data_processing import CreditRiskPreprocessor


# Create a sample dataframe that mimics the real Xente data
@pytest.fixture
def sample_data():
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'TransactionStartTime': pd.to_datetime([
            '2023-12-01 10:00:00',  # C1: Recent
            '2023-12-05 10:00:00',  # C1: More Recent
            '2023-01-01 10:00:00',  # C2: Old (Ghost)
            '2023-12-05 12:00:00'  # C3: Recent
        ]),
        'Amount': [1000, 500, 100, 5000],  # All debits for simplicity
        'ProductCategory': ['airtime', 'utility_bill', 'airtime', 'financial_services'],
        'ChannelId': ['Channel_1', 'Channel_1', 'Channel_2', 'Channel_3'],
        'TransactionHour': [10, 10, 10, 12],
        'TransactionMonth': [12, 12, 1, 12]
    }
    return pd.DataFrame(data)


def test_rfm_calculation(sample_data):
    """
    Test 1: Verify that RFM metrics are calculated correctly.
    Requirement: [cite: 69] 'Test that your feature engineering function returns expected columns'
    """
    # Initialize processor (paths don't matter since we inject data)
    processor = CreditRiskPreprocessor("dummy_raw.csv", "dummy_processed.csv")
    processor.df = sample_data

    # Run the specific function
    rfm_df = processor.create_rfm_features()

    # Assertions
    # 1. Check if columns exist
    assert 'Recency' in rfm_df.columns
    assert 'Frequency' in rfm_df.columns
    assert 'Monetary' in rfm_df.columns

    # 2. Check Logic for Customer C1 (2 transactions, total 1500)
    c1_data = rfm_df.loc['C1']
    assert c1_data['Frequency'] == 2
    assert c1_data['Monetary'] == 1500.0

    # 3. Check Logic for Customer C2 (1 transaction, total 100)
    c2_data = rfm_df.loc['C2']
    assert c2_data['Frequency'] == 1
    assert c2_data['Monetary'] == 100.0


def test_aggregate_features_columns(sample_data):
    """
    Test 2: Verify that One-Hot/Pivot features are created.
    """
    processor = CreditRiskPreprocessor("dummy_raw.csv", "dummy_processed.csv")
    processor.df = sample_data

    # We need to create RFM first because 'create_aggregate_features' expects 'customer_df' to exist
    processor.create_rfm_features()

    # Run aggregation
    processor.create_aggregate_features()

    # Check if categorical columns were created (One-Hot Logic)
    # C1 bought 'airtime' and 'utility_bill', so Cat_airtime should be > 0
    assert 'Cat_airtime' in processor.customer_df.columns
    assert 'Cat_utility_bill' in processor.customer_df.columns

    # C1 bought airtime once
    assert processor.customer_df.loc['C1', 'Cat_airtime'] == 1