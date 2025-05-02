# utils.py
import os
import time
import json
import requests
from typing import Dict, Any, List
import subprocess
from openai import OpenAI
import json
import os


def get_training_data(input_file, output_file):
    processed_data = []
    llm_ignore_data = []
    istype2 = True
    for record in read_jsonl(input_file):
        idx = record.get("idx")
        question = record.get("q", "")
        if "a_std" in record:
            istype2 = False
            answer = record.get("a_std", "")
        elif "a" in record:
            istype2 = True
            answer = record.get("a", "")
        confusion = record.get("t", "")
        response = record.get("response", "")

        if "[LLM Error]" in response:
            llm_ignore_data.append(record)
            continue
        else:
            this_sys = (
                        "You are a mathematician and educator. Solve the following math problem with accurate, complete, and clear explanations. "
                        "Break down your reasoning into a logical chain of steps, and provide the final answer only after completing the reasoning."
                    )
            processed_data.append(
                {
                    'idx': idx,
                    'system': this_sys,
                    'input': question,
                    'output': answer
                }
            )
            
            if istype2:
                final_system = (
                    "You are a mathematician and educator dedicated to resolving confusions about math solutions. "
                    "Provide clear, step-by-step explanations to logically address each confusion."
                )
                final_input = (
                    f"Math Problem: {question}\n"
                    f"Solution: {answer}\n\n\n"
                    f"Confusions about the solution: {confusion}\n\n"
                    "Please address the confusions.\n\n"
                )
            else:
                final_system = (
                    "You are a mathematician and educator dedicated to resolving confusions about correct math solutions. "
                    "Provide clear, step-by-step explanations to logically address each confusion."
                )
                final_input = (
                    f"Math Problem: {question}\n"
                    f"Correct Solution: {answer}\n\n\n"
                    f"Confusions about the solution: {confusion}\n\n"
                    "Please address the confusions.\n\n"
                )
            final_output = response
                 
            processed_data.append(
                {
                    'idx': idx,
                    'system': final_system,
                    'input': final_input,
                    'output': final_output
                }
            )
        
    output_file_name = output_file.split("/")[-1]
    path_to_output = "/".join(output_file.split("/")[:-1])

    write_jsonl(output_file, processed_data, append=True)
    print(f"[INFO] Processed data has been saved to {output_file}")
    
    llm_error_file_path = f"{path_to_output}/llm_error_{output_file_name}"
    if len(llm_ignore_data) > 0:
        write_jsonl(llm_error_file_path, llm_ignore_data, append=True)
        print(f"[INFO] LLM Error data has been saved to {llm_error_file_path}")

def filter_and_fix_file(file_path):
    """
    Reads a JSONL file, removes invalid lines, and overwrites the original file with only valid lines.
    """
    valid_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.strip():  # Check if the line is not empty
                try:
                    json.loads(line)  # Attempt to load the line as JSON
                    valid_lines.append(line)  # Store valid lines
                except json.JSONDecodeError:
                    print(f"Invalid JSON line removed: {line.strip()}")  # Log invalid line
    
    # Overwrite the original file with valid lines
    with open(file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(valid_lines)

def read_jsonl(file_path):
    filter_and_fix_file(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(file_path, data_list, append=False):
    """
    Writes a list of dictionaries to a JSONL file.
    If append is True, appends the data to the file instead of overwriting it.
    """
    mode = 'a' if append else 'w'
    
    # check if file exists
    if not os.path.exists(file_path):
        # check the parent directory and create if it doesn't exist
        parent_dir = os.path.dirname(file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        
        # create the file if it doesn't exist
        with open(file_path, 'w', encoding='utf-8') as f:
            pass
        
        # update mode to write
        mode = 'w'


    with open(file_path, mode, encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def chat_completion(api_base: str, model_name: str, messages: list, max_tokens=256, temperature=0.7):
    """
    Generic helper that uses the new openai client interface to get a chat completion.
    """
    
    if '/v1' not in api_base:
        api_base = api_base + '/v1'
    
    client = OpenAI(base_url=api_base, api_key="xxx")  # point to the local vLLM server
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message.content



def start_vllm_server(model_path: str, model_name: str, port: int, gpu: int = 1):
    """
    Launches a vLLM OpenAI API server via subprocess.
    model_path: The path or name of the model you want to host
    port: Which port to host on
    gpu: The tensor-parallel-size (number of GPUs)
    """
    # Command to activate conda environment and start the server
    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f"--served-model-name={model_name}",
        f'--tensor-parallel-size={gpu}',
        f"--gpu-memory-utilization=0.85",
        f'--port={port}',
        '--trust-remote-code'
    ]

    process = subprocess.Popen(command, shell=False)
    
    wait_for_server(f"http://localhost:{port}", 600)
    
    print(f"[INFO] Started vLLM server for model '{model_path}' on port {port} (GPU={gpu}).")

    return process


def start_vllm_server_with_gpus(model_path: str, model_name: str, port: int, gpus: List[int]):
    """
    Launches a vLLM OpenAI API server via subprocess with specific GPUs assigned.

    Parameters:
    model_path: str - The path or name of the model you want to host.
    model_name: str - The name of the model to be served.
    port: int - The port to host the server on.
    gpus: List[int] - List of GPU indices to be assigned for this server.

    Returns:
    process: subprocess.Popen - The process running the vLLM server.
    """
    gpu_list = ",".join(map(str, gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    command = [
        'python', '-m', 'vllm.entrypoints.openai.api_server',
        f'--model={model_path}',
        f'--served-model-name={model_name}',
        f'--tensor-parallel-size={len(gpus)}',
        '--gpu-memory-utilization=0.85',
        f'--port={port}',
        '--trust-remote-code'
    ]

    process = subprocess.Popen(command, shell=False, env=os.environ.copy())
    
    wait_for_server(f"http://localhost:{port}", 600)

    print(f"[INFO] Started vLLM server for model '{model_name}' on port {port} with GPUs {gpu_list}.")

    return process

def allocate_gpus(total_gpus: int, processes: int) -> List[List[int]]:
    """
    Allocate GPUs for multiple processes.

    Parameters:
    total_gpus: int - Total number of GPUs available.
    processes: int - Number of processes to allocate GPUs for.

    Returns:
    List[List[int]] - A list where each sublist contains the GPUs assigned to a process.
    """
    if total_gpus < processes:
        raise ValueError("Not enough GPUs available for the number of processes.")

    gpus_per_process = total_gpus // processes
    extra_gpus = total_gpus % processes

    allocation = []
    start = 0

    for i in range(processes):
        end = start + gpus_per_process + (1 if i < extra_gpus else 0)
        allocation.append(list(range(start, end)))
        start = end

    return allocation



def wait_for_server(url: str, timeout: int = 600):
    """
    Polls the server's /models endpoint until it responds with HTTP 200 or times out.
    """
    start_time = time.time()
    while True:
        try:
            r = requests.get(url + "/v1/models", timeout=3)
            if r.status_code == 200:
                print("[INFO] vLLM server is up and running.")
                return
        except Exception:
            pass
        if time.time() - start_time > timeout:
            raise RuntimeError(f"[ERROR] Server did not start at {url} within {timeout} seconds.")
        time.sleep(2)
        
def stop_vllm_server(process):
    process.terminate()
    process.wait()
    print("[INFO] Stopped vLLM server.")



def create_output_directory(model_name: str):
    """
    Creates the output directory named after the LLM model, if it doesn't exist.
    Returns the path to that directory.
    """
    output_dir = os.path.join("outputs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
