import subprocess
import time
import os
import shutil
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, set_seed

# Define your custom cache directory.
CACHE_DIR = "./model_cache"

class CodeGenerator:
    def __init__(self, model_name: str = "stabilityai/stable-code-instruct-3b", max_length: int = 256, seed: int = 42):
        """
        Initialize the CodeGenerator with the specified model.
        Uses GPU if available and caches model files in a custom directory.
        """
        # Load model and tokenizer with caching.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

        # Automatically select GPU if available, otherwise use CPU.
        device = 0 if torch.cuda.is_available() else -1

        # Create the text generation pipeline.
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )
        set_seed(seed)
        self.max_length = max_length

    def generate_code(self, prompt: str) -> str:
        """
        Generate Python code from the given prompt.
        The code is expected to be output between the markers: <<<CODE>>> and <<<END CODE>>>.
        """
        generated = self.generator(
            prompt,
            max_length=self.max_length,
            num_return_sequences=1,
            truncation=True
        )
        text = generated[0]['generated_text']
        print("Full generated text:\n", text)

        # Markers for code extraction.
        start_marker = "<<<CODE>>>"
        end_marker = "<<<END CODE>>>"

        # Attempt to extract the code between markers.
        if start_marker in text and end_marker in text:
            code = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
        else:
            # If markers are missing, log a warning and remove the prompt.
            print("Warning: Markers not found in generated text. Attempting to extract code by removing prompt.")
            code = text.replace(prompt, "").strip()
        return code

def write_code_to_file(code: str, filename: str = "main.py"):
    """
    Write the generated code to a file.
    """
    with open(filename, "w") as file:
        file.write(code)
    print(f"Code written to {filename}")

def test_generated_code(filename: str = "main.py") -> (bool, str):
    """
    Execute the generated code in a subprocess and return a success flag along with output or error.
    """
    try:
        result = subprocess.run(["python", filename], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, result.stderr
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return False, "Execution timed out."
    except Exception as e:
        return False, str(e)

def iterative_code_generation(initial_prompt: str, max_attempts: int = 3):
    """
    Iteratively generate code, test it, and use any error messages to refine the prompt.
    """
    generator = CodeGenerator()
    
    # Base prompt instructing the model.
    base_prompt = (
        "You are a Python code generation assistant. "
        "Do not include any explanations, comments, or additional text outside of the code markers. "
        "Output only valid Python code between the markers below.\n\n"
        f"{initial_prompt}\n\n"
    )

    # Start with a clean prompt that expects code between markers.
    prompt = base_prompt + "<<<CODE>>>\n<<<END CODE>>>"
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt} ---")
        print("Current Prompt:\n", prompt)
        
        # Generate code.
        code = generator.generate_code(prompt)
        print("\nGenerated Code:\n", code)
        
        # Write the generated code to a file.
        write_code_to_file(code)
        
        # Test the code.
        success, output = test_generated_code()
        if success:
            print("\nTest passed. Output:")
            print(output)
            return code
        else:
            print("\nTest failed with error:")
            print(output)
            # Shorten the error message if too long.
            error_lines = output.strip().splitlines()
            short_error = "\n".join(error_lines[:5])  # Use first 5 lines of error
            # Update the prompt with error details.
            prompt = (
                base_prompt +
                "<<<CODE>>>\n"
                f"# The previously generated code resulted in the following error:\n"
                f"# {short_error}\n"
                "<<<END CODE>>>"
            )
            # Pause briefly to avoid hitting any rate limits.
            time.sleep(1)

    print("Max attempts reached. Could not generate working code.")
    return None

def clear_cache(cache_dir=CACHE_DIR):
    """
    Clear the model cache by deleting the cache directory.
    """
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared cache at {cache_dir}.")
    else:
        print(f"No cache directory found at {cache_dir}.")

def main():
    # Define the initial prompt for generating a code node that performs addition.
    initial_prompt = (
        "Build me a codenode that performs the addition of two numbers. "
        "The code should include a function named 'add(a, b)' that returns the sum of a and b, "
        "and a main block that demonstrates its usage by printing the result of add(2, 3)."
    )
    
    final_code = iterative_code_generation(initial_prompt, max_attempts=3)
    if final_code:
        print("\nFinal working code:\n")
        print(final_code)
    else:
        print("Failed to generate working code after multiple attempts.")

if __name__ == "__main__":
    main()