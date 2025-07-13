import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta
from tqdm.auto import tqdm # Import tqdm for the progress bar

def process_fan_group(group, timestamp_col='datetime'):
    """
    This function is applied to each group of messages belonging to a single fan-model pair.
    It's much more efficient than a for loop over unique IDs.
    """
    # Sort messages once for the group
    group = group.sort_values(timestamp_col)
    
    # Find purchase and message timestamps
    purchases = group[group['price'] > 0]
    first_message_time = group[timestamp_col].iloc[0]
    last_message_time = group[timestamp_col].iloc[-1]
    first_sale_time = purchases[timestamp_col].min() if not purchases.empty else None
    last_sale_time = purchases[timestamp_col].max() if not purchases.empty else None
    
    stages = []
    
    # --- Stage 1: Acquisition ---
    stage1_end_time = first_sale_time if pd.notna(first_sale_time) else last_message_time
    stage1_mask = (group[timestamp_col] >= first_message_time) & (group[timestamp_col] <= stage1_end_time)
    stage1_messages = group.loc[stage1_mask]
    if not stage1_messages.empty:
        stages.append({
            'fan_model_id': group['fan_model_id'].iloc[0],
            'stage': 1,
            'start_time': first_message_time,
            'end_time': stage1_end_time,
            'text': ' '.join(stage1_messages['message'])
        })

    # If there are sales, proceed to stages 2 and 3
    if pd.notna(first_sale_time):
        # --- Stage 2: Monetization ---
        if pd.notna(last_sale_time) and first_sale_time != last_sale_time:
            stage2_start_time = first_sale_time + timedelta(microseconds=1)
            stage2_mask = (group[timestamp_col] >= stage2_start_time) & (group[timestamp_col] <= last_sale_time)
            stage2_messages = group.loc[stage2_mask]
            if not stage2_messages.empty:
                stages.append({
                    'fan_model_id': group['fan_model_id'].iloc[0],
                    'stage': 2,
                    'start_time': stage2_start_time,
                    'end_time': last_sale_time,
                    'text': ' '.join(stage2_messages['message'])
                })

        # --- Stage 3: Nurture/Churn ---
        if last_message_time > last_sale_time:
            stage3_start_time = last_sale_time + timedelta(microseconds=1)
            stage3_mask = group[timestamp_col] >= stage3_start_time
            stage3_messages = group.loc[stage3_mask]
            if not stage3_messages.empty:
                stages.append({
                    'fan_model_id': group['fan_model_id'].iloc[0],
                    'stage': 3,
                    'start_time': stage3_start_time,
                    'end_time': last_message_time,
                    'text': ' '.join(stage3_messages['message'])
                })
                
    return pd.DataFrame(stages)

if __name__ == '__main__':
    # Define paths to match your project structure
    logs_file_path = 'data/sample_chatlogs.pkl'
    features_file_path = os.path.join('outputs', 'conversation_features.csv')
    output_dir = 'outputs'

    # Load the data
    try:
        with open(logs_file_path, 'rb') as f:
            logs_df = pickle.load(f)
        
        conversation_features_df = pd.read_csv(features_file_path)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}.")
        print("Please ensure your Phase 1 script has run and created the necessary files.")
        exit()

    # --- Data Preparation ---
    print("Preparing data for segmentation...")
    TIMESTAMP_COLUMN = 'datetime'
    logs_df[TIMESTAMP_COLUMN] = pd.to_datetime(logs_df[TIMESTAMP_COLUMN])
    
    # Combine messages into a single column
    if 'fan_message' in logs_df.columns and 'chatter_message' in logs_df.columns:
        logs_df['message'] = logs_df['fan_message'].fillna('') + ' ' + logs_df['chatter_message'].fillna('')
        logs_df['message'] = logs_df['message'].str.strip()
    else:
        logs_df['message'] = logs_df['message'].astype(str).fillna('')

    # Create the fan_model_id for grouping
    logs_df['fan_model_id'] = logs_df['fan_id'].astype(str) + '_' + logs_df['model_name'].astype(str)
    
    # --- Vectorized Processing ---
    print("Starting optimized stage segmentation...")
    # This is the core optimization. It processes the data group by group.
    # We only need to process groups that are present in the conversation_features file.
    relevant_fan_model_ids = conversation_features_df['fan_id'].astype(str) + '_' + conversation_features_df['model_name'].astype(str)
    logs_to_process = logs_df[logs_df['fan_model_id'].isin(relevant_fan_model_ids)]

    # Initialize tqdm for pandas, which adds the .progress_apply() method
    # You may need to install tqdm: pip install tqdm
    tqdm.pandas(desc="Processing Fan Groups")

    # Use .progress_apply() instead of .apply() to show the progress bar
    # The result of groupby().apply() is already a DataFrame, so we don't need to concat later.
    final_staged_df = logs_to_process.groupby('fan_model_id', group_keys=False).progress_apply(
        process_fan_group, timestamp_col=TIMESTAMP_COLUMN
    )
    
    print("Segmentation complete.")
    
    # --- Save Output ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'stage_conversations.csv')
    
    if not final_staged_df.empty:
        # The DataFrame might have a multi-level index after apply, so we reset it.
        final_staged_df.reset_index(drop=True, inplace=True)
        final_staged_df.to_csv(output_path, index=False)
        print(f"✅ Staged conversation data saved to '{output_path}'")
    else:
        print("⚠️ No staged conversations were generated. Output file not created.")
