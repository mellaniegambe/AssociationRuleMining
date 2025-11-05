import streamlit as st
import pandas as pd

# Ensure required packages are installed before importing seaborn/matplotlib
import subprocess
import sys

for pkg in ["seaborn", "matplotlib"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import seaborn as sns
import matplotlib.pyplot as plt

st.title("Market Basket Analysis Dashboard")
st.write("Analyze customer purchasing patterns and discover product associations")

# File Upload Section
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xls", "xlsx"]
)

# Helper Functions

def preprocess_data(df):
    """
    Preprocess data to make it suitable for apriori algorithm.
    Handles different types of input data formats.
    
    # Convert to boolean/binary if not already
    df = df.astype(bool).astype(int)
    Args:
        df (pandas.DataFrame): Input DataFrame
    Returns:
        pandas.DataFrame: Preprocessed DataFrame with binary values
    """
    # Make a copy to avoid modifying original data
    df_copy = df.copy()
    
    # Check if the data is already in the correct format (binary/boolean)
    if set(df_copy.values.ravel()).issubset({0, 1, True, False, 'True', 'False', '0', '1'}):
        # Convert to boolean then to int (0,1)
        for col in df_copy.columns:
            df_copy[col] = df_copy[col].map({'True': True, 'False': False, 
                                           '1': True, '0': False, 
                                           1: True, 0: False}).astype(int)
        return df_copy
    
    # Check if we have numerical data (e.g., quantity-based)
    if df_copy.select_dtypes(include=['int64', 'float64']).columns.any():
        # Convert to binary (1 if quantity > 0, else 0)
        df_copy = (df_copy > 0).astype(int)
        return df_copy
    
    # For categorical/text data, convert to one-hot encoded format
    if df_copy.select_dtypes(include=['object']).columns.any():
        try:
            # First, try to handle comma-separated values
            if df_copy.iloc[0].str.contains(',').any():
                # Split comma-separated values and create one-hot encoding
                all_items = set()
                for col in df_copy.columns:
                    items = df_copy[col].str.split(',').explode().str.strip()
                    all_items.update(items.unique())
                
                # Remove any empty or null values
                all_items = {item for item in all_items if item and pd.notna(item)}
                
                # Create binary columns for each unique item
                binary_df = pd.DataFrame(index=df_copy.index)
                for item in sorted(all_items):
                    binary_df[item] = df_copy.apply(
                        lambda row: any(item in str(val).split(',') for val in row), 
                        axis=1
                    ).astype(int)
                return binary_df
            
            # If not comma-separated, treat each unique value as a separate column
            else:
                binary_df = pd.get_dummies(df_copy).astype(int)
                return binary_df
                
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return None

    return None

def validate_data(df):
    """Validate that data contains only binary values."""
    return ((df == 0) | (df == 1)).all().all()

def load_data(file):
    """
    Load and preprocess the data file.
    
    Args:
        file: Uploaded file object
    Returns:
        pandas.DataFrame: Preprocessed DataFrame suitable for apriori algorithm
    """
    try:
        # Check file extension
        file_extension = file.name.split('.')[-1].lower()
        
        # Read the file based on its extension
        if file_extension == 'csv':
            # Try different encodings
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='latin1')
                
            # Remove any unnamed columns
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # If first column looks like an index, use it as index
            if df.columns[0].lower() in ['index', 'id', 'transaction_id', 'transaction']:
                df.set_index(df.columns[0], inplace=True)
            
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Preprocess the data
        processed_df = preprocess_data(df)
        
        if processed_df is None:
            st.error("Could not process the data into the required format.")
            return None
            
        # Validate final format
        if not validate_data(processed_df):
            st.error("Processed data is not in the correct format (binary values only).")
            return None
            
        return processed_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_statistics(df):
    """
    Display basic statistics and behavior of the preprocessed data.
    Shows item frequency, transaction statistics, and heatmaps.
    """
    st.subheader("Statistical Overview of Data")
    st.markdown("**Item Frequency (Number of Transactions Each Item Appears In):**")
    item_freq = df.sum().sort_values(ascending=False)
    st.bar_chart(item_freq)

    st.markdown("**Transaction Size Distribution (Number of Items per Transaction):**")
    transaction_size = df.sum(axis=1)
    st.line_chart(transaction_size.value_counts().sort_index())

    st.markdown("**Basic Statistics:**")
    stats_data = {
        "Number of Transactions": df.shape[0],
        "Number of Unique Items": df.shape[1],
        "Average Items per Transaction": transaction_size.mean(),
        "Max Items in a Transaction": transaction_size.max(),
        "Min Items in a Transaction": transaction_size.min(),
        "Most Frequent Item": item_freq.idxmax(),
        "Least Frequent Item": item_freq.idxmin()
    }
    st.table(pd.DataFrame(stats_data, index=["Value"]).T)

    # Optional: Show correlation heatmap
    st.markdown("**Item Correlation Heatmap:**")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, ax=ax, cmap="YlGnBu", center=0)
    st.pyplot(fig)

# Use the uploaded file
if uploaded_file is not None:
    processed_df = load_data(uploaded_file)
    if processed_df is not None:
        st.subheader("Preprocessed Data Sample")
        st.write(processed_df.head())
        st.success("Data uploaded and preprocessed successfully.")

        # Display statistical behavior of the data
        display_statistics(processed_df)
    else:
        st.error("Failed to preprocess the uploaded data.")
else:
    st.info("Please upload a CSV or Excel dataset to get started.")
