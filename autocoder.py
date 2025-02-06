import subprocess
import time
import os
import shutil
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, set_seed

# Define your custom cache directory.
CACHE_DIR = "./model_cache"

class CodeGenerator:
    def __init__(self, model_name: str = "stabilityai/stable-code-instruct-3b", max_length: int = 256, seed: int = 42):
        """
        Initialize the CodeGenerator with the specified model.
        This version uses the GPU (if available) and downloads model files to a custom cache directory.
        
        :param model_name: The Hugging Face model identifier.
        :param max_length: The maximum number of tokens to generate.
        :param seed: Seed for reproducibility.
        """
        # Load the model and tokenizer explicitly with the custom cache directory.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        
        # Initialize the text-generation pipeline using the loaded model and tokenizer.
        # device=0 ensures the use of the GPU.
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0  # Use GPU (ensure your GPU is available and configured correctly)
        )
        set_seed(seed)
        self.max_length = max_length

    def generate_code(self, prompt: str) -> str:
        """
        Generate code based on the input prompt.
        :param prompt: A text prompt describing the code to generate.
        :return: A string containing the generated code.
        """
        generated = self.generator(
            prompt,
            max_length=self.max_length,
            num_return_sequences=1,
            truncation=True  # Ensure input is truncated appropriately.
        )
        text = generated[0]['generated_text']

        # Look for our marker and only return the code that follows.
        marker = "# CODE START"
        if marker in text:
            code = text.split(marker, 1)[1].strip()
        else:
            # Fallback: try to remove the original prompt if it's echoed.
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
    Execute the code in a subprocess and capture any errors.
    :return: A tuple of (success flag, output or error message)
    """
    try:
        # Execute the file with a timeout of 10 seconds.
        result = subprocess.run(["python", filename], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            # There was an error executing the code.
            return False, result.stderr
        return True, result.stdout
    except Exception as e:
        return False, str(e)

def iterative_code_generation(initial_prompt: str, max_attempts: int = 3):
    """
    Generate code iteratively: generate code, test it, and if errors occur, inform the model.
    :param initial_prompt: The initial prompt for code generation.
    :param max_attempts: Maximum number of attempts to generate error‑free code.
    :return: The final working code or None if unsuccessful.
    """
    generator = CodeGenerator()
    
    # Modify the prompt: add a clear marker and explicit instructions.
    prompt = (
        initial_prompt +
        "\n# CODE START\n" +
        "# IMPORTANT: Only output the Python code after the '# CODE START' marker. Do not repeat the prompt or add extra commentary."
    )

    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt} ---")
        print("Prompt:\n", prompt)
        
        # Generate code from the current prompt.
        code = generator.generate_code(prompt)
        print("\nGenerated Code:\n", code)
        
        # Write code to file.
        write_code_to_file(code)
        
        # Test the generated code.
        success, output = test_generated_code()
        if success:
            print("\nTest passed. Output:")
            print(output)
            return code
        else:
            print("\nTest failed with error:")
            print(output)
            # Append the error details to the prompt and ask the model to fix the code.
            prompt = (
                initial_prompt +
                "\n# CODE START\n" +
                "# The previous generated code resulted in the following error when executed:\n"
                f"# {output.strip()}\n"
                "# Please provide a corrected version of the code. Only output the Python code after the marker."
            )
            # Pause briefly before retrying.
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
    # Define the initial prompt to generate a codenode that performs addition.
    initial_prompt = (
        "Build me a codenode that does the addition of two numbers. Write the complete code for this functionality in a file named main.py. "
        "The code should include:\n"
        "  - A function `add(a, b)` that returns the sum of a and b.\n"
        "  - A main block that demonstrates its usage by printing the result of add(2, 3).\n"
    )
    
    final_code = iterative_code_generation(initial_prompt, max_attempts=3)
    if final_code:
        print("\nFinal working code:\n")
        print(final_code)
    else:
        print("Failed to generate working code after multiple attempts.")

if __name__ == "__main__":
    main()