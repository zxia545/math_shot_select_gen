import os
import argparse
import time
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils # Assuming utils.py is in the same directory or python path

# --- Prompt Template ---
# Defines how we ask the LLM to compare the pairs.
# Crucial for getting reliable results.
SYSTEM_PROMPT = """You are an expert evaluator. Your task is to carefully compare two responses ('Output Response') to a given query ('Input Query').
Determine which response is superior based on overall quality, considering factors like clarity, accuracy, completeness, helpfulness, and relevance to the query.
"""

USER_PROMPT_TEMPLATE = """Please compare the following two items based on their 'Input Query' and 'Output Response'.

**Input Query:**
{input_query}

**Item 1 Output Response:**
{output_1}


**Item 2 Output Response:**
{output_2}

**Evaluation Task:**
Analyze both 'Output Response' texts. Which item's response ('Item 1' or 'Item 2') provides a better overall answer to the 'Input Query'?

**Your Answer Format:**
You MUST respond ONLY with the exact text "Item 1" or "Item 2". Do not include any other words, explanations, or punctuation.
"""


# --- Helper Function for Parsing LLM Choice ---
def parse_llm_choice(response_text, pair_id="N/A"):
    """
    Parses the LLM response to determine the chosen item (1 or 2).
    Returns 1 or 2, defaulting to 1 on ambiguity or error.
    """
    if not response_text:
        print(f"[Warning] Empty LLM response for pair {pair_id}. Defaulting to Item 1.")
        return 1

    cleaned_response = response_text.strip()

    # Try exact matches first (most reliable if model follows instructions)
    if cleaned_response == "Item 1":
        return 1
    if cleaned_response == "Item 2":
        return 2

    # If not exact, try simple containment (less reliable)
    lower_response = cleaned_response.lower()
    is_item1 = "item 1" in lower_response
    is_item2 = "item 2" in lower_response

    if is_item1 and not is_item2:
        print(f"[Warning] LLM response '{response_text}' for pair {pair_id} contained 'Item 1' but wasn't exact. Interpreting as Item 1.")
        return 1
    if is_item2 and not is_item1:
        print(f"[Warning] LLM response '{response_text}' for pair {pair_id} contained 'Item 2' but wasn't exact. Interpreting as Item 2.")
        return 2

    # Ambiguous or incorrectly formatted response
    print(f"[Warning] Ambiguous/incorrect LLM response format: '{response_text}' for pair {pair_id}. Defaulting to Item 1.")
    return 1 # Default choice


# --- Function to Process a Single Pair ---
def process_pair(item1, item2, api_base, model_name, max_tokens=10, temperature=0.0, max_retries=3):
    """
    Sends a pair comparison request to the LLM and returns the chosen item.
    """
    pair_id = f"Idx({item1.get('idx', 'N/A')},{item2.get('idx', 'N/A')})"

    # Use input from item 1 (assuming inputs should ideally be the same for a pair)
    input_query = item1.get("input", "[Input Missing]")
    output_1 = item1.get("output", "[Output Missing]")
    output_2 = item2.get("output", "[Output Missing]")

    if input_query == "[Input Missing]" or output_1 == "[Output Missing]" or output_2 == "[Output Missing]":
         print(f"[Warning] Missing input/output for pair {pair_id}. Comparison may be unreliable. Defaulting to Item 1.")
         # Cannot reliably ask LLM, so default immediately
         return item1

    # Check if inputs actually match (optional but good sanity check)
    # if item1.get("input") != item2.get("input"):
    #     print(f"[Warning] Inputs for paired items {pair_id} do not match. Using Item 1's input.")

    user_prompt = USER_PROMPT_TEMPLATE.format(
        input_query=input_query,
        output_1=output_1,
        output_2=output_2
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    chosen_item_index = 1 # Default to Item 1 in case of complete failure

    for attempt in range(max_retries):
        try:
            response = utils.chat_completion(
                api_base=api_base,
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens, # Expecting very short response ("Item 1" or "Item 2")
                temperature=temperature # Low/zero temp for deterministic choice
            )
            chosen_item_index = parse_llm_choice(response, pair_id=pair_id)
            # print(f"Pair {pair_id}: LLM chose Item {chosen_item_index}") # Debug print
            break # Success! Exit retry loop.
        except Exception as e:
            print(f"[Error] API call failed (Attempt {attempt+1}/{max_retries}) for {pair_id}: {e}")
            if attempt == max_retries - 1:
                print(f"[Error] Max retries reached for {pair_id}. Defaulting to Item 1.")
                chosen_item_index = 1 # Default on final failure
            else:
                wait_time = 2 ** attempt # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    return item1 if chosen_item_index == 1 else item2

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Use a hosted vLLM model to perform pairwise comparison and selection on JSONL files.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the vLLM model (e.g., /path/to/model/files).')
    parser.add_argument('--served_model_name', type=str, required=True, help='Name the model will be served as (used in API calls and output filenames).')
    parser.add_argument('--port', type=int, default=8000, help='Port to host the vLLM server on.')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs for vLLM tensor parallelism.')
    parser.add_argument('--input_dir', type=str, default='./output_splits', help='Directory containing the split_N.jsonl files.')
    parser.add_argument('--output_dir', type=str, default='./output_model_select_half', help='Directory to save the selected items.')
    parser.add_argument('--threads', type=int, default=10, help='Number of concurrent API calls for pair comparisons.')
    parser.add_argument('--max_tokens', type=int, default=10, help='Max tokens for the LLM choice response (should be small).')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for the LLM choice response (0.0 for deterministic).')

    args = parser.parse_args()

    # --- Validate Utils ---
    required_funcs = ['start_vllm_server', 'stop_vllm_server', 'chat_completion', 'read_jsonl', 'write_jsonl']
    if not all(hasattr(utils, func) for func in required_funcs):
        print("Error: utils.py is missing required functions.")
        print(f"Needs: {', '.join(required_funcs)}")
        sys.exit(1)


    # --- Start Server ---
    vllm_process = None
    api_base = f"http://localhost:{args.port}/v1" # Define API base based on port
    try:
        print(f"Attempting to start vLLM server for model '{args.served_model_name}'...")
        vllm_process = utils.start_vllm_server(args.model_path, args.served_model_name, args.port, args.gpu)
        print(f"Server started (or already running). API Base: {api_base}")
    except Exception as e:
        print(f"Error starting vLLM server: {e}")
        print("Please ensure vLLM is installed and configured correctly.")
        sys.exit(1) # Exit if server fails to start


    try:
        # --- Prepare Directories ---
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")

        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' not found.")
            sys.exit(1)

        # --- Process Files ---
        print(f"Looking for files matching 'split_*.jsonl' in: {args.input_dir}")
        try:
            input_files = sorted([f for f in os.listdir(args.input_dir) if f.startswith('split_') and f.endswith('.jsonl')])
        except FileNotFoundError:
             print(f"Error: Input directory '{args.input_dir}' does not exist.")
             sys.exit(1)


        if not input_files:
             print(f"No 'split_*.jsonl' files found in '{args.input_dir}'.")
        else:
            print(f"Found {len(input_files)} file(s) to process.")

        total_processed_files = 0
        for filename in input_files:
            input_filepath = os.path.join(args.input_dir, filename)
            print(f"\n--- Processing file: {filename} ---")

            # Extract split number N (optional, for context)
            n_split_from_name = 0
            match = re.search(r'split_(\d+)\.jsonl', filename)
            if match:
                n_split_from_name = int(match.group(1))

            # Read data using utils function (yields items)
            data = list(utils.read_jsonl(input_filepath))
            if not data:
                print(f"File '{filename}' is empty or failed to read. Skipping.")
                continue

            n_actual = len(data)
            print(f"Read {n_actual} items.")

            if n_split_from_name > 0 and n_actual != n_split_from_name:
                 print(f"[Warning] Filename suggests {n_split_from_name} items, but found {n_actual}. Using actual count.")

            if n_actual % 2 != 0:
                print(f"[Error] File '{filename}' contains an odd number of items ({n_actual}). Cannot pair perfectly. Skipping.")
                continue
            if n_actual == 0:
                 print(f"File '{filename}' resulted in zero items after reading. Skipping.")
                 continue

            selected_items = []
            futures = {}
            # Use ThreadPoolExecutor for concurrent API calls
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                print(f"Submitting {n_actual // 2} pairs for comparison using {args.threads} threads...")
                # Iterate through pairs
                for i in range(0, n_actual, 2):
                    item1 = data[i]
                    item2 = data[i+1]
                    # Submit task to executor
                    future = executor.submit(process_pair, item1, item2, api_base, args.served_model_name, args.max_tokens, args.temperature)
                    futures[future] = (item1.get('idx', i), item2.get('idx', i+1)) # Store pair info for logging errors

                print("Waiting for comparisons to complete...")
                # Collect results as they complete
                for future in as_completed(futures):
                    pair_info = futures[future]
                    try:
                        chosen_item = future.result()
                        if chosen_item: # Ensure result is not None or unexpected
                             selected_items.append(chosen_item)
                    except Exception as e:
                        # This catches errors from within the future task itself if not handled internally
                        print(f"[Error] Task execution failed for pair ({pair_info[0]} vs {pair_info[1]}): {e}")

            # Sort selected items by original index? Optional.
            # selected_items.sort(key=lambda x: x.get('idx', float('inf'))) # Use inf for missing idx

            # Construct output filename using the SERVED model name
            # Use n_actual (number of items read) for consistency in naming if filename was misleading
            output_filename = f"{args.served_model_name}_split_{n_actual}_selected.jsonl"
            output_filepath = os.path.join(args.output_dir, output_filename)

            if selected_items:
                print(f"Attempting to save {len(selected_items)} selected items to {output_filepath}")
                # Save the selected items using utils function
                utils.write_jsonl(output_filepath, selected_items)
                print(f"Successfully saved {output_filepath}")
                total_processed_files += 1
            else:
                print(f"No items were successfully selected for {filename}. No output file generated.")


    except KeyboardInterrupt:
         print("\nProcess interrupted by user.")
    except Exception as e:
         print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- Stop Server ---
        if vllm_process:
            print("\nAttempting to stop vLLM server...")
            utils.stop_vllm_server(vllm_process)
            print("Server stop command issued.")
        else:
             print("\nServer was not started by this script or failed to start.")

    print(f"\n--- Script Finished. Processed {total_processed_files} file(s). ---")


if __name__ == "__main__":
    main()