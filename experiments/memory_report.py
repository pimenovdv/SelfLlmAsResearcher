import sys
from src.experiment_utils import load_model_and_tokenizer, get_model_memory_footprint, clear_memory

def main():
    model_name = "gpt2"
    model, _ = load_model_and_tokenizer(model_name)
    mem_bytes = get_model_memory_footprint(model)
    print(f"Memory footprint for {model_name}: {mem_bytes} bytes")
    clear_memory()

if __name__ == "__main__":
    main()
