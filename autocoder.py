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
        Uses the GPU (if available) and caches the model files to a custom directory.
        """
        # Load model and tokenizer explicitly with cache_dir.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        
        # Create a text-generation pipeline using the loaded model and tokenizer.
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0  # Force GPU usage (ensure GPU is available)
        )
        set_seed(seed)
        self.max_length = max_length

    def generate_code(self, prompt: str) -> str:
        """
        Generate Python code from the given prompt.
        The prompt instructs the model to output only valid Python code between the markers:
        <<<CODE>>> and <<<END CODE>>>.
        """
        generated = self.generator(
            prompt,
            max_length=self.max_length,
            num_return_sequences=1,
            truncation=True
        )
        text = generated[0]['generated_text']
        
        # Define robust markers.
        start_marker = "<<<CODE>>>"
        end_marker = "<<<END CODE>>>"
        
        # Extract code between markers.
        if start_marker in text and end_marker in text:
            code = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
        else:
            # Fallback: try to remove the prompt text if markers are missing.
            code = text.replace(prompt, "").strip()
        return code

def write_code_to_file(code: str, filename: str = "main.py"):
    """
    Write the generated code to the specified file.
    """
    with open(filename, "w") as file:
        file.write(code)
    print(f"Code written to {filename}")

def test_generated_code(filename: str = "main.py") -> (bool, str):
    """
    Execute the generated code in a subprocess and return the success flag and output or error.
    """
    try:
        result = subprocess.run(["python", filename], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, result.stderr
        return True, result.stdout
    except Exception as e:
        return False, str(e)

def iterative_code_generation(initial_prompt: str, max_attempts: int = 3):
    """
    Iteratively generate code using the model, test it, and if errors occur,
    supply the error details to prompt the model for a correction.
    """
    generator = CodeGenerator()
    
    # Construct a revised prompt template.
    # Note: We instruct the model to output ONLY valid Python code between the markers,
    # and nothing else.
    prompt = (
        "You are a Python code generation assistant. "
        "Do not include any explanations, comments, or additional text outside of the code markers. "
        "Output only valid Python code between the markers below.\n\n"
        f"{initial_prompt}\n\n"
        "<<<CODE>>>\n"
        "<<<END CODE>>>"
    )

    for attempt in range(1, max_attempts + 1):
        print(f"\n--- Attempt {attempt} ---")
        print("Prompt:\n", prompt)
        
        # Generate code from the current prompt.
        code = generator.generate_code(prompt)
        print("\nGenerated Code:\n", code)
        
        # Write the generated code to main.py.
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
            # Update the prompt to include the error details.
            prompt = (
                "You are a Python code generation assistant. "
                "Do not include any explanations, comments, or additional text outside of the code markers. "
                "Output only valid Python code between the markers below.\n\n"
                f"{initial_prompt}\n\n"
                "<<<CODE>>>\n"
                f"# The previously generated code resulted in this error when executed:\n"
                f"# {output.strip()}\n"
                "<<<END CODE>>>"
            )
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
        "Build me a codenode that does the addition of two numbers. "
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