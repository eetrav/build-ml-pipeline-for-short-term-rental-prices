import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data: pd.DataFrame):
    """
    Function to test that expected columns are present in the dataframe.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data: pd.DataFrame):
    """
    Function to test that neighborhood_names are valid.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in/around NYC.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Compare distribution of input dataset to reference dataset.

    Apply a threshold on the KL divergence to detect if the distribution of the
    new data is significantly different than that of the reference dataset.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
        ref_data (pd.DataFrame): Reference dataframe for rental pricing.
        kl_threshold (float): Threshold to check for dataset divergence.
    """

    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold

def test_row_count(data: pd.DataFrame):
    """
    Function to test that input dataframe has a reasonable number of rows.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
    """

    assert 15000 < data.shape[0] < 1000000

def test_price_range(data: pd.DataFrame, min_price: float, max_price: float):
    """
    Function to check that the nightly price falls within a certain range.

    Args:
        data (pd.DataFrame): Input dataframe for rental pricing.
        min_price (float): Lower bound for nightly pricing (in USD).
        max_price (float): Upper bound for nightly pricing (in USD).
    """
    
    assert data['price'].between(min_price, max_price).all()

