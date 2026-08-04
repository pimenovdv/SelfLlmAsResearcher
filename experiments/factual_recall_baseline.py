import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from src.metrics import brier_score, top_k_accuracy, mean_reciprocal_rank

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_model_factual_recall(model_name, prompts_targets):
    logging.info(f"Testing model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt, target_word in prompts_targets:
        inputs = tokenizer(prompt, return_tensors="pt")
        target_ids = tokenizer.encode(target_word)
        target_id = target_ids[0]

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[0, -1, :]
        predicted_token_id = torch.argmax(next_token_logits).item()
        predicted_word = tokenizer.decode(predicted_token_id)

        target_logit = next_token_logits[target_id].item()

        # Calculate Brier Score and other metrics
        brier = brier_score(outputs.logits, target_id)
        top_k = top_k_accuracy(outputs.logits, target_id, k=5)
        mrr = mean_reciprocal_rank(outputs.logits, target_id)

        logging.info(f"Prompt: '{prompt}'")
        logging.info(f"Predicted word: '{predicted_word}' (id: {predicted_token_id})")
        logging.info(f"Target word: '{target_word}' (id: {target_id})")
        logging.info(f"Target logit: {target_logit:.4f}")
        logging.info(f"Brier Score: {brier:.4f}")
        logging.info(f"Top-5 Accuracy: {top_k:.4f}")
        logging.info(f"MRR: {mrr:.4f}\n")


if __name__ == "__main__":
    prompts_targets = [
        ("The capital of France is", " Paris"),
        ("The capital of Italy is", " Rome"),
        ("The capital of Germany is", " Berlin"),
        ("The capital of Spain is", " Madrid"),
        ("The capital of Russia is", " Moscow"),
    ]

    models_to_test = ["gpt2", "EleutherAI/gpt-neo-125m"]

    for model_name in models_to_test:
        test_model_factual_recall(model_name, prompts_targets)
