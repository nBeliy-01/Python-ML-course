import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    
    Example:
        df = load_data('train.csv')
    """
    return pd.read_csv(file_path, index_col=0)

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets.
    
    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Target column name.
        test_size (float, optional): Proportion of data for validation. Defaults to 0.2.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and validation datasets.
    
    Example:
        train_df, val_df = split_data(df, 'Exited')
    """
    return train_test_split(df, stratify=df[target_col], test_size=test_size, random_state=42)

def identify_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify numeric and categorical columns in a dataset.
    
    Args:
        df (pd.DataFrame): Input dataset.
    
    Returns:
        Tuple[list, list]: Numeric and categorical column names.
    
    Example:
        numeric_cols, categorical_cols = identify_column_types(df)
    """
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols

def process_geography_column(df: pd.DataFrame, column_name: str, target_column: str) -> Dict[str, float]:
    """
    Compute Weight of Evidence (WOE) transformation for a categorical column.
    
    Args:
        df (pd.DataFrame): Input dataset.
        column_name (str): Name of the categorical column.
        target_column (str): Name of the target column.
    
    Returns:
        Dict[str, float]: Mapping of categories to WOE values.
    
    Example:
        woe_map = process_geography_column(df, 'Geography', 'Exited')
    """
    total_counts = df[column_name].value_counts()
    positive_counts = df[df[target_column] == 1][column_name].value_counts()
    negative_counts = df[df[target_column] == 0][column_name].value_counts()
    woe_map = {}
    for category in total_counts.index:
        if category in positive_counts and category in negative_counts:
            woe_map[category] = np.log((positive_counts[category] / total_counts[category]) /
                                       (negative_counts[category] / total_counts[category]))
        else:
            woe_map[category] = 0.0
    return woe_map

def preprocess_data(file_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Preprocess dataset: load, split, transform, and return processed data.
    
    Args:
        file_path (str): Path to the CSV file.
        target_col (str): Target column name.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Processed training and validation data and targets.
    
    Example:
        train_inputs, val_inputs, train_targets, val_targets = preprocess_data('train.csv', 'Exited')
    """
    df = load_data(file_path)
    train_df, val_df = split_data(df, target_col)
    
    input_cols = list(train_df.columns[:-1])
    train_inputs, train_targets = train_df[input_cols], train_df[target_col]
    val_inputs, val_targets = val_df[input_cols], val_df[target_col]
    
    numeric_cols, categorical_cols = identify_column_types(train_inputs)
    
    woe_geography = process_geography_column(train_df, 'Geography', target_col)
    train_inputs['Geography'] = train_inputs['Geography'].replace(woe_geography)
    val_inputs['Geography'] = val_inputs['Geography'].replace(woe_geography)
    
    return train_inputs, val_inputs, train_targets, val_targets
