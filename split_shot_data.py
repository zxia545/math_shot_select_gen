import json
import random
import os
import math # Not strictly necessary with // but good to have if needed

# --- Configuration ---
INPUT_FILE = './data/shot_data_ori_60.jsonl'
SPLIT_SIZES = [2, 6, 10, 18, 22, 30, 38, 42, 46, 50, 54, 58]
SPLIT_OUTPUT_DIR = './output_splits'
RANDOM_HALF_OUTPUT_DIR = './output_random_half'

# --- Helper Functions ---

def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of loaded JSON objects."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line #{i+1} in {file_path}: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    return data

def write_jsonl(file_path, data_list):
    """Writes a list of dictionaries to a JSONL file."""
    try:
        # Ensure parent directory exists
        parent_dir = os.path.dirname(file_path)
        if parent_dir: # Check if it's not empty (i.e., not just a filename)
            os.makedirs(parent_dir, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Successfully wrote {len(data_list)} lines to {file_path}")
        return True
    except IOError as e:
        print(f"Error writing to {file_path}: {e}")
        return False
    except Exception as e:
         print(f"An unexpected error occurred during writing to {file_path}: {e}")
         return False

# --- Main Script Logic ---

def main():
    print(f"Starting script...")
    print(f"Input file: {INPUT_FILE}")
    print(f"Split sizes: {SPLIT_SIZES}")
    print(f"Split output directory: {SPLIT_OUTPUT_DIR}")
    print(f"Random half output directory: {RANDOM_HALF_OUTPUT_DIR}")

    # 1. Read all data from the original file
    all_data = read_jsonl(INPUT_FILE)
    if all_data is None:
        print("Exiting due to file read error.")
        return

    total_lines = len(all_data)
    print(f"\nRead {total_lines} lines from {INPUT_FILE}.")

    if total_lines == 0:
         print("Input file is empty. No output files will be generated.")
         return

    # 2. Create output directories
    try:
        os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)
        os.makedirs(RANDOM_HALF_OUTPUT_DIR, exist_ok=True)
        print(f"Ensured output directories exist.")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        return

    # 3. Process each desired split size
    for size in SPLIT_SIZES:
        print("-" * 20)
        print(f"Processing target split size: {size}")

        # Check if requested size is feasible
        if size <= 0:
             print(f"Warning: Invalid size {size}. Skipping.")
             continue
        if size > total_lines:
            print(f"Warning: Requested size {size} is larger than total lines ({total_lines}). Using all {total_lines} lines instead for split_{size}.")
            current_size = total_lines # Adjust size to what's available
        else:
            current_size = size

        # 4. Randomly sample for the 'split' file
        # random.sample chooses unique elements without replacement
        split_data = random.sample(all_data, current_size)
        split_filename = os.path.join(SPLIT_OUTPUT_DIR, f"split_{current_size}.jsonl")
        if not write_jsonl(split_filename, split_data):
             print(f"Skipping random half generation for size {current_size} due to write error.")
             continue # Skip to next size if writing failed

        # 5. Randomly sample half from the 'split' data for the 'random' file
        half_size = current_size // 2 # Integer division gives floor

        if half_size <= 0:
             print(f"Info: Half size calculated as {half_size} for split size {current_size}. No 'random_{half_size}.jsonl' file generated for this size.")
             continue

        # Sample from the *already sampled* split_data
        random_half_data = random.sample(split_data, half_size)
        # Naming uses the calculated half_size
        random_half_filename = os.path.join(RANDOM_HALF_OUTPUT_DIR, f"random_{half_size}.jsonl")
        write_jsonl(random_half_filename, random_half_data)

    print("-" * 20)
    print("\nScript finished.")

if __name__ == "__main__":
    main()