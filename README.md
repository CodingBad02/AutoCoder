# CodeNode AutoCoder

This project implements an iterative code generation system that uses an open-source code generation model to create and test a Python code snippet based on natural language instructions. It is designed to integrate into a Node and Workflow Builder system where user queries can be transformed into working code.

## Features

- **Iterative Code Generation:**  
  The system generates code from a natural language prompt and tests it. If the code fails, the error is appended to the prompt, and the model is asked to fix it until a working version is produced (up to a maximum number of attempts).

- **GPU Support:**  
  The model is configured to run on a GPU for faster inference (if available).

- **Custom Cache Directory:**  
  Models are downloaded to a custom cache directory (`./model_cache`) to keep your workspace organized. The cache directory is added to `.gitignore` to prevent it from being committed to version control.

## Prerequisites

- Python 3.8 or higher
- PyTorch with GPU support (if you wish to run the model on GPU)
- [Transformers](https://huggingface.co/transformers/) library