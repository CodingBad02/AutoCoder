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
        # Explicitly load the model and tokenizer with a custom cache directory.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        
        # Initialize the text-generation pipeline with GPU support (device=0).
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0
        )
        set_seed(seed)
        self.max_length = max_length

    def generate_code(self, prompt: str) -> str:
        """
        Generate code based on the input prompt.
        The model is instructed to output code between the markers:
            <<<CODE>>> and <<<END CODE>>>
        :param prompt: A text prompt describing the code to generate.
        :return: A string containing the extracted generated Python code.
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
        
        # Attempt to extract code between the markers.
        if start_marker in text and end_marker in text:
            code = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
        else:
            # Fallback: remove the prompt and any extraneous markdown formatting.
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
    
    # Construct the prompt using robust code markers.
    prompt = (
        initial_prompt +
        "\n<<<CODE>>>\n" +
        "# IMPORTANT: Output only the Python code between the markers <<<CODE>>> and <<<END CODE>>>. " +
        "Do not include any markdown formatting, comments, or additional text outside these markers.\n" +
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
            # Append error details to the prompt for a corrected version.
            prompt = (
                initial_prompt +
                "\n<<<CODE>>>\n" +
                "# The previously generated code resulted in the following error when executed:\n" +
                f"# {output.strip()}\n" +
                "# Please provide a corrected version of the code between the markers <<<CODE>>> and <<<END CODE>>>. " +
                "Do not include any markdown formatting or extra text outside these markers.\n" +
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
        "Build me a codenode that does the addition of two numbers. " +
        "Write the complete code for this functionality in a file named main.py. " +
        "The code should include:\n" +
        "  - A function 'add(a, b)' that returns the sum of a and b.\n" +
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