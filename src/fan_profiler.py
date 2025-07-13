import pandas as pd
import os
import time
import json
from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set in .env file.")

client = Groq(api_key=API_KEY)

# --- File Configuration ---
CACHE_FILE = 'output/llm_profile_cache.json'
OUTPUT_FILE = 'output/fan_profiles.csv'

# --- Token Estimation (FIXED) ---
def estimate_tokens(text: str):
    """More accurate token estimate: 1 token ≈ 3.3 characters for English text."""
    return int(len(text) / 3.3)

# --- Caching Functions ---
def load_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

# --- Cache Validation Functions ---
def validate_profile_data(profile_data):
    """Check if profile data contains all required fields with meaningful content."""
    required_fields = [
        "demographics", "personal_details", "life_events",
        "personality_and_needs", "purchase_motivators", "communication_patterns"
    ]
    
    if not isinstance(profile_data, dict):
        return False
    
    for field in required_fields:
        if field not in profile_data:
            return False
        
        value = profile_data[field]
        if not isinstance(value, str):
            return False
        
        # Check if the field has meaningful content (not just "Not mentioned" or empty)
        if not value or value.strip() == "" or value.strip().lower() == "not mentioned":
            continue  # This field is empty, but that's okay
        
        # Check if field has reasonable length (at least 10 characters for meaningful content)
        if len(value.strip()) < 10:
            return False
    
    return True

def clean_cache(cache):
    """Remove invalid profiles from cache that don't meet current requirements."""
    cleaned_cache = {}
    removed_count = 0
    
    for fan_model_id, profile_data in cache.items():
        if validate_profile_data(profile_data):
            cleaned_cache[fan_model_id] = profile_data
        else:
            removed_count += 1
            print(f"Removing invalid profile from cache: {fan_model_id}")
    
    if removed_count > 0:
        print(f"Cleaned {removed_count} invalid profiles from cache")
        save_cache(cleaned_cache)
    
    return cleaned_cache

# --- Prompt Formatting (FIXED for proper JSON formatting) ---
def format_prompt(conversation_text):
    system_prompt = """You are an expert data analyst specializing in understanding human conversation. Your task is to analyze a conversation history and extract a detailed profile of the fan based on their interactions.

IMPORTANT: You must return ONLY a valid JSON object with the following exact structure. Fill each field with detailed, descriptive information based on the conversation.

CRITICAL JSON FORMATTING RULES:
- Use double quotes for all strings
- Replace all apostrophes with spaces (do not -> do not, can't -> cannot)
- Replace all internal quotes with single quotes or remove them
- Use plain language without special characters
- Be descriptive but use clean, simple sentences

{
  "demographics": "Detailed age indicators and relationship status information from the conversation",
  "personal_details": "Job information, hobbies, interests, and location hints mentioned", 
  "life_events": "Major life events like divorce, job changes, health issues, family situations",
  "personality_and_needs": "Personality traits, emotional needs, behavioral patterns, and psychological insights",
  "purchase_motivators": "What drives their spending decisions, financial motivations, and purchasing patterns",
  "communication_patterns": "How they communicate, when they message, tone, frequency, and interaction style"
}

ANALYSIS GUIDELINES:
- Extract specific details and examples from the conversation
- Synthesize information into coherent profiles
- Use "Not mentioned" only if truly no relevant information exists
- Focus on actionable insights that reveal the fan as a person
- Include specific examples when possible (ages mentioned, job titles, locations, etc.)
- Capture emotional states and underlying motivations
- Note communication timing, frequency, and style patterns

Return only the JSON object with no other text."""

    user_prompt = f"""Analyze this conversation and extract the fan profile information:

{conversation_text}

Return only the JSON object:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# --- Profile Merging Logic (Updated for new descriptive fields) ---
def merge_profiles(profiles):
    """Merges descriptive profiles from multiple conversation chunks."""
    
    # These are the fields we expect from the new prompt
    fields = [
        "demographics", "personal_details", "life_events",
        "personality_and_needs", "purchase_motivators", "communication_patterns"
    ]
    
    final_profile = {field: "Not mentioned" for field in fields}

    for field in fields:
        all_details = set()
        for profile in profiles:
            # Get the detail from a chunk's profile, default to "Not mentioned"
            detail = profile.get(field, "Not mentioned")
            if detail and detail != "Not mentioned":
                # Clean up sentences and add to a set to ensure uniqueness
                # Split by common sentence delimiters and strip whitespace
                sentences = [s.strip() for s in detail.replace(';', '.').split('.') if s.strip()]
                all_details.update(sentences)
        
        if all_details:
            # Join with periods and ensure proper sentence structure
            combined_text = ". ".join(sorted(list(all_details)))
            # Clean up any double periods or other formatting issues
            combined_text = combined_text.replace("...", ".").replace("..", ".")
            if not combined_text.endswith("."):
                combined_text += "."
            final_profile[field] = combined_text

    return final_profile

# --- Enhanced JSON Parsing with Fallback ---
def safe_json_parse(response_text):
    """Safely parse JSON with fallback cleaning."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to clean common JSON issues
        cleaned = response_text.strip()
        
        # Remove any markdown formatting
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        # Clean up common JSON issues without being too aggressive
        # Replace smart quotes with regular quotes
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace(''', "'").replace(''', "'")
        
        # Fix unescaped quotes in strings (simple approach)
        import re
        # This regex looks for patterns like: "text"more text"
        # and replaces the middle quote with a single quote
        cleaned = re.sub(r'": "([^"]*)"([^"]*)"([^"]*)"', r'": "\1\'\2\'\3"', cleaned)
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON after cleaning: {response_text[:300]}...")
            return None

# --- LLM Call (FIXED with better error handling) ---
@retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(3))
def get_llm_profile(conversation_text: str):
    """
    Analyzes conversation text, intelligently slicing it to fit within the model's
    token limit and merging the results.
    """
    # FIXED: Reduced limits to stay well under Groq's 6000 TPM limit
    MAX_TOKENS_ALLOWED = 5000  # More conservative limit
    MAX_OUTPUT_TOKENS = 800    # Increased for detailed profiles
    
    dummy_messages = format_prompt("")
    prompt_overhead_text = "".join(m['content'] for m in dummy_messages)
    PROMPT_OVERHEAD_TOKENS = estimate_tokens(prompt_overhead_text)

    SAFE_SLICE_TOKENS = MAX_TOKENS_ALLOWED - PROMPT_OVERHEAD_TOKENS - MAX_OUTPUT_TOKENS
    
    print(f"Debug: Prompt overhead: {PROMPT_OVERHEAD_TOKENS} tokens")
    print(f"Debug: Safe slice limit: {SAFE_SLICE_TOKENS} tokens")
    
    slices = []
    
    if estimate_tokens(conversation_text) <= SAFE_SLICE_TOKENS:
        slices = [conversation_text]
    else:
        # FIXED: More conservative chunking
        chars_per_chunk = int(SAFE_SLICE_TOKENS * 3.0)  # More conservative ratio
        current_pos = 0
        while current_pos < len(conversation_text):
            end_pos = current_pos + chars_per_chunk
            chunk = conversation_text[current_pos:end_pos]
            slices.append(chunk)
            current_pos = end_pos

    print(f"Debug: Split into {len(slices)} chunks")
    
    all_partial_profiles = []

    for idx, chunk in enumerate(slices):
        if not chunk.strip():
            continue

        messages = format_prompt(chunk)
        
        # FIXED: Add token validation before API call
        total_input_tokens = sum(estimate_tokens(m['content']) for m in messages)
        print(f"Debug: Chunk {idx+1} estimated tokens: {total_input_tokens}")
        
        if total_input_tokens + MAX_OUTPUT_TOKENS > MAX_TOKENS_ALLOWED:
            print(f"⚠️ Chunk {idx+1} still too large ({total_input_tokens} tokens), skipping")
            continue
        
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=messages,
                temperature=0.1,  # Reduced for more consistent JSON
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
            
            response_text = response.choices[0].message.content
            result = safe_json_parse(response_text)
            
            if result is None:
                print(f"⚠️ Failed to parse JSON in chunk {idx+1}, skipping")
                continue
                
        except Exception as e:
            print(f"⚠️ API Error on chunk {idx+1}: {e}")
            continue

        # --- Filter results to ensure they match the expected structure ---
        expected_fields = {
            "demographics", "personal_details", "life_events",
            "personality_and_needs", "purchase_motivators", "communication_patterns"
        }
        filtered_result = {k: v for k, v in result.items() if k in expected_fields}
        for field in expected_fields:
            if field not in filtered_result:
                filtered_result[field] = "Not mentioned"
        
        all_partial_profiles.append(filtered_result)

    if all_partial_profiles:
        return merge_profiles(all_partial_profiles)
    else:
        print(f"Could not generate a profile for a conversation.")
        return None

# --- Batch Processor (FIXED: Added rate limiting) ---
def process_conversation_batch(conversations_df, cache, batch_size=50):
    processed_count = 0
    total_conversations = len(conversations_df)

    with tqdm(total=total_conversations, desc="Processing conversations", unit="conv") as pbar:
        for i in range(0, len(conversations_df), batch_size):
            batch = conversations_df.iloc[i:i+batch_size]
            for index, row in batch.iterrows():
                fan_model_id = row['fan_model_id']
                if fan_model_id in cache:
                    pbar.update(1)
                    continue
                try:
                    profile_data = get_llm_profile(row['full_conversation'])
                    if profile_data and validate_profile_data(profile_data):
                        profile_data['fan_model_id'] = fan_model_id
                        cache[fan_model_id] = profile_data
                        processed_count += 1
                        if processed_count % 5 == 0:
                            save_cache(cache)
                            pbar.set_postfix({"Processed": processed_count, "Cache saved": "✓"})
                    else:
                        pbar.set_postfix({"Last status": "Invalid profile generated"})
                except Exception as e:
                    pbar.set_postfix({"Last error": str(e)[:50]})
                pbar.update(1)
                # FIXED: Longer rate limit to respect Groq's TPM limits
                time.sleep(4)  # Increased from 3 to 4 seconds
            save_cache(cache)
            pbar.set_postfix({"Batch completed": i//batch_size + 1, "Total processed": processed_count})

    return processed_count

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # NOTE: Ensure you have 'HOMEWORK_LOGS.pkl' or your specified file
        full_logs_df = pd.read_pickle('sample_chatlogs.pkl') 
    except FileNotFoundError:
        print("Error: 'sample_chatlogs.pkl' not found. Please ensure the file is in the correct directory.")
        exit()

    try:
        # Create a unique ID for each fan-model relationship, as per the README
        full_logs_df['fan_model_id'] = full_logs_df['fan_id'].astype(str) + "_" + full_logs_df['model_name']
        full_logs_df['fan_message'] = full_logs_df['fan_message'].fillna('')
        full_logs_df['chatter_message'] = full_logs_df['chatter_message'].fillna('')
        # Combine messages into a clear conversational format
        full_logs_df['full_message'] = "Model: " + full_logs_df['chatter_message'] + "\n" + "Fan: " + full_logs_df['fan_message']
    except KeyError as e:
        print(f"Error: A required column ({e}) was not found in the input file.")
        exit()

    # Group all messages by the unique fan-model ID
    conversations_to_profile = full_logs_df.groupby('fan_model_id')['full_message'].apply(
        lambda msgs: "\n---\n".join(msgs)
    ).reset_index(name='full_conversation')

    cache = load_cache()
    
    # Clean cache of invalid profiles
    cache = clean_cache(cache)
    
    # Find conversations that need processing (not in cache OR invalid profiles)
    conversations_to_process_df = conversations_to_profile[
        ~conversations_to_profile['fan_model_id'].isin(cache.keys())
    ]

    total_to_process = len(conversations_to_process_df)
    print(f"Found {len(conversations_to_profile)} total unique fan-model relationships.")
    print(f"Loaded {len(cache)} valid profiles from cache.")
    print(f"Starting to process {total_to_process} new conversations.")
    
    # Show a sample of what's in cache vs what needs processing
    if len(cache) > 0:
        print(f"\nSample cached profile fields:")
        sample_profile = next(iter(cache.values()))
        for field in sample_profile:
            if field != 'fan_model_id':
                value = sample_profile[field]
                preview = (value[:80] + "...") if len(value) > 80 else value
                print(f"  {field}: {preview}")
    
    if total_to_process > 0:
        print(f"\nExample conversation to process (first 200 chars):")
        print(f"  {conversations_to_process_df.iloc[0]['full_conversation'][:200]}...")
    print()

    if not conversations_to_process_df.empty:
        processed_count = process_conversation_batch(conversations_to_process_df, cache)
        print(f"\nProcessed {processed_count} new profiles.")
        save_cache(cache)
        print("Final cache saved.")
    else:
        print("No new conversations to process.")

    if cache:
        final_profiles_df = pd.DataFrame(list(cache.values()))
        # Reorder columns to have the ID first, followed by the profile fields
        cols = ['fan_model_id'] + [col for col in final_profiles_df.columns if col != 'fan_model_id']
        final_profiles_df = final_profiles_df[cols]
        
        final_profiles_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ All processing complete. Final profiles saved to {OUTPUT_FILE}")
        print(f"Total profiles: {len(final_profiles_df)}")
        print("\n--- Sample of Final Profiles ---")
        print(final_profiles_df.head())
    else:
        print("\nNo profiles were generated or found in cache.")