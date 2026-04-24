import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def open_dataset(
        file_path: str = None,
        target: str = None,
        features: list[str] = None,
):
    # Open main datafile
    df = pd.read_csv(file_path, index_col='Date', parse_dates=['Date'])

    # Add shifted workday feature indicating if the next day is a workday
    if 'nextday_workday' in features:
        df['nextday_workday'] = df['workday'].shift(-24, fill_value=0)

    # Get relevant columns
    relevant_cols = [target] + features if target not in features else features

    # Remove all irrelevant columns
    df = df.drop(df.columns.difference(relevant_cols), axis=1)

    return df


def split_dataset(
        df: pd.DataFrame,
        train_range: tuple[str, str],
        val_range: tuple[str, str],
        test_range: tuple[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train, validation, and test sets based on year-month ranges.

    Args:
        df (pd.DataFrame): The input dataframe with a datetime index.
        train_range (tuple[str, str]): Start and end year-month for the training set (e.g., ('2015-01', '2016-12')).
        val_range (tuple[str, str]): Start and end year-month for the validation set (e.g., ('2017-01', '2017-06')).
        test_range (tuple[str, str]): Start and end year-month for the test set (e.g., ('2017-07', '2017-12')).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test datasets.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The dataframe index must be a DatetimeIndex.")

        # Filter data for the training set
    train_start, train_end = train_range
    train_df = df.loc[train_start:train_end]

    # Filter data for the validation set
    val_start, val_end = val_range
    val_df = df.loc[val_start:val_end]

    # Filter data for the test set
    test_start, test_end = test_range
    test_df = df.loc[test_start:test_end]

    return train_df, val_df, test_df


def scale_data(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        columns_to_scale: list[str] = None,
        target_column: str = None,
        scaler_type: str = 'min-max'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, object]:
    """
    Scales specific columns and/or the target column in the training, validation, and test sets.

    Args:
        train_df (pd.DataFrame): Training dataset (must have a datetime index).
        val_df (pd.DataFrame): Validation dataset (optional).
        test_df (pd.DataFrame): Test dataset (optional).
        columns_to_scale (list[str]): List of column names to scale (optional).
        target_column (str): Name of the target column to scale (optional).
        scaler_type (str): Type of scaler ('min-max' or 'standard').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object, object]:
            Scaled train, validation, test datasets, feature scaler (or None), and target scaler (or None).
    """
    assert isinstance(train_df, pd.DataFrame), 'Train dataset must be a pandas DataFrame.'
    if columns_to_scale is None and target_column is None:
        raise ValueError("Either 'columns_to_scale' or 'target_column' must be specified for scaling.")

    # Initialize scalers
    feature_scaler = None
    target_scaler = None

    if columns_to_scale:
        if scaler_type == 'min-max':
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == 'standard':
            feature_scaler = StandardScaler()
        else:
            raise ValueError("Invalid scaler_type. Choose 'min-max' or 'standard'.")
        feature_scaler.fit(train_df[columns_to_scale].values)  # Fit feature scaler on training data

    if target_column:
        if scaler_type == 'min-max':
            target_scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == 'standard':
            target_scaler = StandardScaler()
        else:
            raise ValueError("Invalid scaler_type. Choose 'min-max' or 'standard'.")
        target_scaler.fit(train_df[target_column].values.reshape(-1, 1))  # Fit target scaler on training data

    # Scale training data
    train_df_scaled = train_df.copy()
    if columns_to_scale and feature_scaler:
        train_df_scaled[columns_to_scale] = feature_scaler.transform(train_df[columns_to_scale].values)
    if target_column and target_scaler:
        train_df_scaled[target_column] = target_scaler.transform(
            train_df[target_column].values.reshape(-1, 1)).flatten()

        # Scale validation data (if provided)
    if val_df is not None:
        val_df_scaled = val_df.copy()
        if columns_to_scale and feature_scaler:
            val_df_scaled[columns_to_scale] = feature_scaler.transform(val_df[columns_to_scale].values)
        if target_column and target_scaler:
            val_df_scaled[target_column] = target_scaler.transform(
                val_df[target_column].values.reshape(-1, 1)).flatten()
    else:
        val_df_scaled = None

        # Scale test data (if provided)
    if test_df is not None:
        test_df_scaled = test_df.copy()
        if columns_to_scale and feature_scaler:
            test_df_scaled[columns_to_scale] = feature_scaler.transform(test_df[columns_to_scale].values)
        if target_column and target_scaler:
            test_df_scaled[target_column] = target_scaler.transform(
                test_df[target_column].values.reshape(-1, 1)).flatten()
    else:
        test_df_scaled = None

    return train_df_scaled, val_df_scaled, test_df_scaled, feature_scaler, target_scaler


def sliding_window(
        features,
        labels,
        window_size: int,
        label_distance: int,
        multihorizon: bool = False,
        return_tensor: bool = True
):
    """
    Converts time-series data into input feature sequences (`x`) and corresponding labels (`y`)
    using a sliding window approach.

    Args:
        features (np.ndarray): Time-series feature data of shape (num_samples, num_features).
        labels (np.ndarray): 1D array of labels associated with the feature sequences (shape: (num_samples,)).
        window_size (int): Specifying the length of each input feature sequence.
        label_distance (int): Number of steps into the future for the label.
        multihorizon (bool): Whether to generate multiple future labels (default: False).
        return_tensor (bool): Whether to return the data as tensor (default: True).

    Returns:
        tuple[np.ndarray, np.ndarray] or tuple[torch.tensor, torch.tensor]:
            - x: Feature sequences of shape (num_windows, ws, num_features).
            - y: Corresponding labels of shape (num_windows,) or (num_windows, label_distance) for multihorizon.
    """
    # Sanity check: Ensure features and labels have the same length
    assert len(features) == len(labels), "Feature and label arrays must have the same length."

    x = []  # List to store feature windows
    y = []  # List to store corresponding labels
    L = len(features)  # Total number of samples in the time series

    # Sliding window generation logic
    if not multihorizon:
        # Single horizon: One label per feature window
        for i in range(L - window_size - label_distance + 1):
            window = features[i:i + window_size]  # Extract feature window
            label = labels[i + window_size + label_distance - 1]  # Extract single label (future prediction)
            x.append(window)
            y.append(label)
    else:
        # Multi-horizon: Multiple labels per feature window
        for i in range(L - window_size - label_distance + 1):
            window = features[i:i + window_size]  # Extract feature window
            label = labels[
                    i + window_size:i + window_size + label_distance]  # Extract multiple labels (future predictions)
            x.append(window)
            y.append(label)

    # Convert lists to numpy arrays
    x = np.array(x, dtype=np.float32)  # Shape: (num_windows, ws, num_features)
    y = np.array(y, dtype=np.float32)  # Shape: (num_windows,) or (num_windows, label_distance)

    if return_tensor:
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

    return x, y
