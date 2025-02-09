import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple, List, Dict

# WOE для колонки Geography
def process_geography_column(X, column_name='Geography', target_column='Exited'):
    """
    Create WOE for Geography column.
    """
    total_gender_counts = X[column_name].value_counts()
    negative_gender_counts = X[X[target_column] == 0][column_name].value_counts()
    positive_gender_counts = X[X[target_column] == 1][column_name].value_counts()

    woe_France = None
    woe_Germany = None
    woe_Spain = None

    woe_France = np.log((positive_gender_counts['France'] / total_gender_counts['France']) /
                                  (negative_gender_counts['France'] / total_gender_counts['France']))
    
    try:
        woe_Germany = np.log((positive_gender_counts['Germany'] / total_gender_counts['Germany']) /
                                        (negative_gender_counts['Germany'] / total_gender_counts['Germany']))
    except:
        pass
        
    woe_Spain = np.log((positive_gender_counts['Spain'] / total_gender_counts['Spain']) /
                                    (negative_gender_counts['Spain'] / total_gender_counts['Spain']))
    woe_map = {"Spain": woe_Spain, "Germany": woe_Germany, "France": woe_France}
    
    return woe_map

def split_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    """
    Split the dataset into training and validation sets.
    
    Args:
        raw_df (pd.DataFrame): The input DataFrame containing the dataset.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
            - Training DataFrame
            - Validation DataFrame
            - List of input column names
            - Target column name
    
    Example:
        >>> train_df, val_df, input_cols, target_col = split_data(df)
    """

    raw_df = raw_df.drop(columns=['CustomerId', 'Surname'])
    
    train_df, val_df = train_test_split(raw_df, stratify=raw_df['Exited'], test_size=0.2, random_state=42)
    input_cols = list(train_df.columns[:-1].values)
    target_col = train_df.columns[-1]

    return train_df, val_df, input_cols, target_col

def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        Tuple[List[str], List[str]]: List of numeric column names and list of categorical column names.
    
    Example:
        >>> numeric_cols, categorical_cols = identify_column_types(df)
    """
    numeric_cols = list(df.select_dtypes(['float', 'int']).columns.values)
    categorical_cols = list(df.select_dtypes(['object']).columns.values)
    return numeric_cols, categorical_cols

def scale_numeric_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler.
    
    Args:
        train_inputs (pd.DataFrame): Training input features.
        val_inputs (pd.DataFrame): Validation input features.
        numeric_cols (List[str]): List of numeric column names.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
            - Scaled training input features
            - Scaled validation input features
            - Fitted StandardScaler instance
    """
    scaler = StandardScaler()
    train_inputs[numeric_cols] = scaler.fit_transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler

def encode_categorical_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, categorical_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder]:
    """
    One-hot encode categorical features.
    
    Args:
        train_inputs (pd.DataFrame): Training input features.
        val_inputs (pd.DataFrame): Validation input features.
        categorical_cols (List[str]): List of categorical column names.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, OneHotEncoder]:
            - Encoded training data
            - Encoded validation data
            - Fitted OneHotEncoder instance
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_train = encoder.fit_transform(train_inputs[categorical_cols])
    encoded_val = encoder.transform(val_inputs[categorical_cols])
    return encoded_train, encoded_val, encoder

def preprocess_data(raw_df: pd.DataFrame):
    """
    Preprocess the input DataFrame by splitting it into train and validation sets,
    extracting input and target columns, and applying feature transformations.
    
    Args:
        raw_df (pd.DataFrame): The input dataset.
    
    Returns:
        Dict[np.ndarray, pd.Series, np.ndarray, pd.Series, List[str], StandardScaler, OneHotEncoder]:
            - Processed training features
            - Training target values
            - Processed validation features
            - Validation target values
            - List of input columns
            - StandardScaler instance
            - OneHotEncoder instance
    
    Example:
        >>> info = preprocess_data(df)
    """
    train_df, val_df, input_cols, target_col = split_data(raw_df)

    woe_geography = process_geography_column(train_df, 'Geography', 'Exited')
    train_df['Geography'] = train_df['Geography'].replace(woe_geography)
    val_df['Geography'] = val_df['Geography'].replace(woe_geography)

    # # # Тест №1 - розділити Age до 40 та від 40
    train_df['more_than_40_years'] = train_df['Age'].apply(lambda x: 1 if x >= 40 else 0)
    val_df['more_than_40_years'] = val_df['Age'].apply(lambda x: 1 if x >= 40 else 0)
    
    # # # Тест №2 - розділити Balance до від 100 до 120 k
    train_df['range_balance_100_120'] = train_df['Balance'].apply(lambda x: 1 if x >= 100000 and x <= 120000 else 0)
    val_df['range_balance_100_120'] = val_df['Balance'].apply(lambda x: 1 if x >= 100000 and x <= 120000 else 0)
    
    # # # Тест №3 - розділити Balance 0 та не нуль
    train_df['is_zero_balance'] = train_df['Balance'].apply(lambda x: 1 if x == 0 else 0)
    val_df['is_zero_balance'] = val_df['Balance'].apply(lambda x: 1 if x == 0 else 0)
    
    # # # Тест №4 - підсвітити клієнтів, в яких кількість продуктів або 1, 3, 4 або лише 3, 4
    train_df['many_products'] = train_df['NumOfProducts'].apply(lambda x: 1 if np.isin(x, [3, 4]) else 0)
    val_df['many_products'] = val_df['NumOfProducts'].apply(lambda x: 1 if np.isin(x, [3, 4]) else 0)

    processed_cols = ['more_than_40_years', 'range_balance_100_120', 'is_zero_balance', 'many_products']
    
    train_inputs, train_targets = train_df[input_cols + processed_cols], train_df[target_col]
    val_inputs, val_targets = val_df[input_cols + processed_cols], val_df[target_col]
    
    numeric_cols, categorical_cols = identify_column_types(train_inputs)
    train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
    encoded_train, encoded_val, encoder = encode_categorical_features(train_inputs, val_inputs, categorical_cols)
    
    train_inputs = np.hstack([train_inputs[numeric_cols].values, encoded_train])
    val_inputs = np.hstack([val_inputs[numeric_cols].values, encoded_val])

    result = {
        "X_train": train_inputs,
        "y_train": train_targets,
        "X_val": val_inputs,
        "y_val": val_targets,
        "input_cols": input_cols + processed_cols,
        "scaler": scaler,
        "encoder": encoder,
        "woe_geography": woe_geography
    }
    
    return result

def preprocess_new_data(raw_test_df: pd.DataFrame, woe_geography: Dict, scaler: StandardScaler, encoder: OneHotEncoder):
    """
    Preprocess the input DataFrame by processing new data, and applying feature transformations.
    """
    
    raw_test_df = raw_test_df.drop(columns=["Surname", "CustomerId"])

    raw_test_df['Geography'] = raw_test_df['Geography'].replace(woe_geography)
    raw_test_df['more_than_40_years'] = raw_test_df['Age'].apply(lambda x: 1 if x >= 40 else 0)
    raw_test_df['range_balance_100_120'] = raw_test_df['Balance'].apply(lambda x: 1 if x >= 100000 and x <= 120000 else 0)
    raw_test_df['is_zero_balance'] = raw_test_df['Balance'].apply(lambda x: 1 if x == 0 else 0)
    raw_test_df['many_products'] = raw_test_df['NumOfProducts'].apply(lambda x: 1 if np.isin(x, [3, 4]) else 0)
    
    numeric_cols, categorical_cols = identify_column_types(raw_test_df)

    scaled_input_df = scaler.transform(raw_test_df[numeric_cols])
    encoded_input_df = encoder.transform(raw_test_df[categorical_cols])
    
    df = np.hstack([scaled_input_df, encoded_input_df])

    return df
    

def plot_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str):
    """
    Plot the ROC curve for the given predictions.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted probabilities.
        dataset_name (str): Name of the dataset.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str):
    """
    Plot the confusion matrix for the given predictions.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        dataset_name (str): Name of the dataset.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure()
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.show()
