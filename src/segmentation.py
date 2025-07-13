import pandas as pd
import numpy as np
import os

def load_and_clean_data(filepath='data/sample_chatlogs.pkl'):
    """
    Loads and cleans the chatlog data.
    This version includes cleaning for invalid fan_id values.
    """
    if not os.path.exists(filepath):
        print(f"Error: Data file not found at '{filepath}'")
        return None

    # Ensure the 'data' directory exists before trying to load from it
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Info: 'data' directory created. Please place 'sample_chatlogs.pkl' inside it.")
        return None

    df = pd.read_pickle(filepath)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    print("Data loaded successfully.")

    # --- NEW: Clean invalid fan_id values ---
    # This is the fix for the '!' issue.
    # We will remove any rows that have a missing or invalid fan_id.
    initial_rows = len(df)
    df.dropna(subset=['fan_id'], inplace=True)
    # Ensure fan_id is a string to use .str accessor, and remove leading/trailing spaces
    df = df[df['fan_id'].astype(str).str.strip() != '!']
    print(f"Cleaned invalid fan_ids. Removed {initial_rows - len(df)} rows.")
    # --- End of new cleaning step ---

    # Standardize data types first
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Filter out system messages and drop empty rows in a single operation
    is_not_system_message = ~df['fan_message'].str.contains('System:', na=False)
    df_cleaned = df.loc[is_not_system_message].dropna(subset=['fan_message', 'chatter_message'], how='all').copy()
    
    return df_cleaned

def segment_conversations(df):
    """Segments messages into unique conversations based on fan-model pairs and time."""
    print("Segmenting conversations...")
    df = df.sort_values(['fan_id', 'model_name', 'datetime']).copy()

    # Create a unique ID for each fan-model relationship
    df['fan_model_id'] = df['fan_id'].astype(str) + '_' + df['model_name']

    # A new conversation starts after a 4-hour gap
    df['time_gap'] = df.groupby('fan_model_id')['datetime'].diff()
    df['new_conversation_start'] = (df['time_gap'] > pd.Timedelta(hours=4)) | (df['time_gap'].isna())

    # Assign a globally unique ID to each conversation
    conversation_group = df['new_conversation_start'].cumsum()
    df['conversation_id'] = df['fan_model_id'] + '_conv_' + conversation_group.astype(str)
    
    print(f"Segmented into {df['conversation_id'].nunique()} unique conversations.")
    return df

def calculate_conversation_features(df):
    """Calculates engagement, revenue, and status features for each conversation."""
    print("Calculating conversation features...")
    
    # Use a compatible syntax for pandas aggregation
    conv_features = df.groupby('conversation_id').agg({
        'fan_id': 'first',
        'model_name': 'first',
        'revenue': 'sum',
        'purchased': 'sum',
        'fan_message': 'count',
        'datetime': ['min', 'max']
    })

    # Flatten the multi-level columns that the aggregation creates
    conv_features.columns = ['_'.join(col).strip() for col in conv_features.columns.values]

    # Rename the columns to our desired format
    conv_features = conv_features.rename(columns={
        'fan_id_first': 'fan_id',
        'model_name_first': 'model_name',
        'revenue_sum': 'total_revenue',
        'purchased_sum': 'num_purchases',
        'fan_message_count': 'message_count',
        'datetime_min': 'conversation_start',
        'datetime_max': 'conversation_end'
    }).reset_index()

    # --- Calculate derived features ---
    conv_features['duration_hours'] = (conv_features['conversation_end'] - conv_features['conversation_start']).dt.total_seconds() / 3600
    conv_features['purchase_rate'] = (conv_features['num_purchases'] / conv_features['message_count']).fillna(0)
    
    latest_time_in_dataset = df['datetime'].max()
    conv_features['days_since_last_message'] = (latest_time_in_dataset - conv_features['conversation_end']).dt.days
    conv_features['is_active'] = conv_features['conversation_end'] >= (latest_time_in_dataset - pd.Timedelta(days=2))

    # --- Calculate purchase-timing features ---
    convos_with_purchases = conv_features[conv_features['num_purchases'] > 0]['conversation_id']
    purchase_df = df[df['conversation_id'].isin(convos_with_purchases)]
    
    if not purchase_df.empty:
        first_purchase_times = purchase_df[purchase_df['purchased'] > 0].groupby('conversation_id')['datetime'].min().rename('first_purchase_time')
        conv_features = conv_features.merge(first_purchase_times, on='conversation_id', how='left')
        
        conv_features['time_to_first_purchase_hours'] = (conv_features['first_purchase_time'] - conv_features['conversation_start']).dt.total_seconds() / 3600

        def get_messages_before_purchase(group):
            first_purchase_time = group['first_purchase_time'].iloc[0]
            if pd.isna(first_purchase_time): 
                return None
            return len(group[group['datetime'] < first_purchase_time])

        temp_df = df.merge(conv_features[['conversation_id', 'first_purchase_time']], on='conversation_id', how='left')
        
        msgs_before = temp_df.groupby('conversation_id').apply(get_messages_before_purchase, include_groups=False).rename('messages_before_first_purchase')
        conv_features = conv_features.merge(msgs_before, on='conversation_id', how='left')
    else:
        conv_features['first_purchase_time'] = pd.NaT
        conv_features['time_to_first_purchase_hours'] = np.nan
        conv_features['messages_before_first_purchase'] = np.nan

    print("All features calculated.")
    return conv_features

if __name__ == "__main__":
    INPUT_FILEPATH = 'data/sample_chatlogs.pkl'
    OUTPUT_DIR = 'outputs'
    OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, 'conversation_features.csv')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chat_logs_df = load_and_clean_data(filepath=INPUT_FILEPATH)
    if chat_logs_df is not None and not chat_logs_df.empty:
        conversations_df = segment_conversations(chat_logs_df)
        conversation_features = calculate_conversation_features(conversations_df)
        
        conversation_features.to_csv(OUTPUT_FILEPATH, index=False)
        
        print(f"\n✅ Phase 1 complete. Output saved to: {OUTPUT_FILEPATH}")
        print("\nSample of final output:")
        print(conversation_features.head())
    else:
        print("\nCould not proceed due to missing or empty data after cleaning.")

