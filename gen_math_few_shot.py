# gen_math_few_shot.py
# (Renamed from gen_math.py for clarity)

import os
import argparse
import time
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm # Added tqdm for progress visualization
import utils # Assuming utils.py is in the same directory or python path

# --- Constants ---
DEFAULT_TEST_FILE = 'math_test_500.jsonl'

# --- Prompt Templates ---
# (Same as in the previous generate_and_eval script)
GENERATION_SYSTEM_PROMPT = "You are an Expert Mathematician. Your task is to provide a response that is thorough and accurate. Solve the following math problem with accurate, complete, and clear explanations. Break down your reasoning into a logical chain of steps, and provide the final answer only after completing the reasoning."
GENERATION_SHOT_FORMAT = "User: {input}\nAssistant: {output}"
GENERATION_FINAL_QUERY_FORMAT = "User: {input}\nAssistant:"

# --- Helper Functions ---

def construct_few_shot_prompt(shots, test_input):
    """Constructs the full prompt string with few-shot examples."""
    prompt_parts = []
    for shot in shots:
        try:
            prompt_parts.append(GENERATION_SHOT_FORMAT.format(
                input=shot.get('input', ''),
                output=shot.get('output', '')
            ))
        except KeyError:
            print(f"[Warning] Skipping shot due to missing 'input' or 'output': {shot.get('idx', 'N/A')}")
            continue
        except TypeError as e:
            print(f"[Error] Error formatting shot (check input/output types): {shot.get('idx', 'N/A')} - {e}")
            continue

    prompt_parts.append(GENERATION_FINAL_QUERY_FORMAT.format(input=test_input))
    # Separate shots and final query by double newline for clarity
    return "\n\n".join(prompt_parts)

# --- Core Processing Function ---

def process_test_item_with_shots(test_item, shots, api_base, model_name, max_tokens, temperature, max_retries=3):
    """
    Processes a single test item using few-shot examples to generate an answer.
    """
    test_idx = test_item.get('idx', 'N/A')
    problem_input = test_item.get("input") # Input from the test file item

    if not problem_input:
        print(f"[Warning] Skipping item with idx {test_idx} due to missing 'input' field in test data.")
        result = test_item.copy()
        result["llm_answer"] = "[Error] Missing 'input' field in test data"
        return result

    # Construct the few-shot prompt
    full_user_prompt = construct_few_shot_prompt(shots, problem_input)

    # Structure for chat completion API
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": full_user_prompt}
    ]

    llm_answer = "[Error] Generation Failed"
    for attempt in range(max_retries):
        try:
            response = utils.chat_completion(
                api_base=api_base,
                model_name=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            llm_answer = response
            break # Success
        except Exception as e:
            print(f"[Error] API call failed (Attempt {attempt+1}/{max_retries}) for test idx {test_idx}: {e}")
            if attempt == max_retries - 1:
                llm_answer = f"[LLM Error] Generation failed after {max_retries} attempts: {e}"
            else:
                 wait_time = 2 ** attempt # Exponential backoff
                 print(f"Retrying in {wait_time} seconds...")
                 time.sleep(wait_time)

    # Combine original test item data with the generated answer
    result_item = test_item.copy()
    result_item["llm_answer"] = llm_answer
    return result_item

# --- Main Execution Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers for a fixed test set using different few-shot examples.")

    # LLM Model Args
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model (if hosting locally with vLLM).")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model being served (used in API calls and output filenames).")
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for the LLM API (e.g., http://localhost:8000/v1). If provided, ignores --model_path/--port/--gpu.")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the model on (if using --model_path).")
    parser.add_argument("--gpu", type=int, default=1, help="Number of GPUs to use (if using --model_path).")

    # Generation Parameters
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation (Lower for math is often better).")

    # Data Args
    parser.add_argument('--test_file', type=str, default=DEFAULT_TEST_FILE, help=f'Path to the main test JSONL file (default: {DEFAULT_TEST_FILE}).')
    parser.add_argument('--shots_dir', type=str, required=True, help='Directory containing the few-shot JSONL files.')
    parser.add_argument('--shot_selection_prefix', type=str, default=None, help='Optional prefix to filter shot files in --shots_dir (e.g., for model-selected shots).')
    parser.add_argument('--output_dir', type=str, default='./few_shot_gen_results', help='Directory to save the generated results.')

    # Concurrency Args
    parser.add_argument("--threads", type=int, default=10, help="Number of concurrent API calls.")

    args = parser.parse_args()

    # --- Validate Utils ---
    # (Add check for utils functions if needed)

    # --- Determine API Base ---
    effective_api_base = args.api_base
    if not effective_api_base and args.model_path:
         effective_api_base = f"http://localhost:{args.port}/v1"
    elif not effective_api_base and not args.model_path:
         print("Error: Must provide either --api_base or --model_path for the generator model.")
         sys.exit(1)
    # Auto-add /v1 if missing
    if not effective_api_base.endswith('/v1'):
         effective_api_base = effective_api_base.rstrip('/') + '/v1'


    # --- Start Server (if local path provided and no api_base override) ---
    vllm_process = None
    should_start_server = args.model_path
    try:
        if should_start_server:
            print(f"Starting vLLM server for model '{args.model_name}' on port {args.port}...")
            vllm_process = utils.start_vllm_server(args.model_path, args.model_name, args.port, args.gpu)
            print(f"Using local API base: {effective_api_base}")
        else:
            print(f"Using provided API base: {effective_api_base}")


        # --- Load Fixed Test Data ---
        print(f"\nLoading test data from: {args.test_file}")
        try:
            test_data = list(utils.read_jsonl(args.test_file))
            if not test_data:
                 print(f"Error: No data found in test file {args.test_file}. Exiting.")
                 sys.exit(1)
            print(f"Loaded {len(test_data)} test items.")
        except FileNotFoundError:
             print(f"Error: Test file not found at {args.test_file}")
             sys.exit(1)
        except Exception as e:
             print(f"Error reading test file {args.test_file}: {e}")
             sys.exit(1)

        # --- Find Shot Files ---
        if not os.path.isdir(args.shots_dir):
            print(f"Error: Shots directory '{args.shots_dir}' not found.")
            sys.exit(1)

        shot_files = []
        try:
            all_files = sorted(os.listdir(args.shots_dir))
        except OSError as e:
            print(f"Error listing files in shots directory '{args.shots_dir}': {e}")
            sys.exit(1)

        for f in all_files:
            if f.endswith('.jsonl'):
                # Apply prefix filter if provided
                if args.shot_selection_prefix:
                    if f.startswith(args.shot_selection_prefix):
                        shot_files.append(os.path.join(args.shots_dir, f))
                else:
                    # If no prefix, include all jsonl files
                     shot_files.append(os.path.join(args.shots_dir, f))

        if not shot_files:
            print(f"Error: No shot files found in '{args.shots_dir}'" +
                  (f" matching prefix '{args.shot_selection_prefix}'." if args.shot_selection_prefix else "."))
            sys.exit(1)

        print(f"\nFound {len(shot_files)} shot file(s) to process:")
        # Limit printing if too many files
        display_limit = 10
        for i, sf in enumerate(shot_files):
            if i < display_limit:
                 print(f" - {os.path.basename(sf)}")
            elif i == display_limit:
                 print(f" - ... (and {len(shot_files) - display_limit} more)")


        # --- Create Output Directory ---
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nOutput directory: {args.output_dir}")

        # --- Main Loop: Process Each Shot File ---
        total_processed_shots = 0
        for shot_file_path in shot_files:
            shot_filename = os.path.basename(shot_file_path)
            print(f"\n{'='*20} Processing Shot File: {shot_filename} {'='*20}")

            # Load shots for this iteration
            shots = list(utils.read_jsonl(shot_file_path))
            if not shots:
                print(f"Warning: Shot file '{shot_filename}' is empty or failed to read. Skipping.")
                continue
            print(f"Loaded {len(shots)} shots from {shot_filename}.")

            # --- Generation Phase ---
            generated_results_for_this_shot = []
            with ThreadPoolExecutor(max_workers=args.threads) as executor:
                # Create futures list
                futures = [
                    executor.submit(
                        process_test_item_with_shots,
                        item, shots, effective_api_base, args.model_name,
                        args.max_tokens, args.temperature
                    ) for item in test_data
                ]
                # Process completed futures with tqdm progress bar
                print(f"Submitting {len(test_data)} generation tasks...")
                for future in tqdm.tqdm(as_completed(futures), total=len(test_data), desc=f"Generating ({shot_filename})"):
                    try:
                        result = future.result()
                        generated_results_for_this_shot.append(result)
                    except Exception as e:
                        # This might catch errors if future.result() raises something unexpected
                        print(f"\nError retrieving result from generation task: {e}")

            print(f"Generation complete for {shot_filename}. Generated {len(generated_results_for_this_shot)} results.")

            # --- Save Results ---
            if generated_results_for_this_shot:
                output_basename = os.path.splitext(shot_filename)[0]
                # Include generator model name in output filename
                output_filename = f"{args.model_name}_{output_basename}_gen.jsonl"
                output_filepath = os.path.join(args.output_dir, output_filename)

                print(f"Saving generation results to: {output_filepath}")
                utils.write_jsonl(output_filepath, generated_results_for_this_shot)
                print(f"Successfully saved.")
                total_processed_shots += 1
            else:
                print(f"No results generated for {shot_filename}. No output file created.")

    except KeyboardInterrupt:
         print("\nProcess interrupted by user.")
    except Exception as e:
         print(f"\nAn unexpected error occurred in the main script: {e}")
         import traceback
         traceback.print_exc() # Print full traceback for unexpected errors
    finally:
        # --- Stop Server ---
        if vllm_process:
            print("\nStopping vLLM server...")
            utils.stop_vllm_server(vllm_process)
            print("Server stop command issued.")
        else:
            print("\nServer was not started by this script or was already running externally.")

    print(f"\n--- Few-Shot Generation Script Finished. Processed {total_processed_shots} shot file(s). ---")