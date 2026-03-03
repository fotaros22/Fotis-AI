from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import MODEL_NAME

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def chat(input_text):
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Fotis-AI Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        reply = chat(user_input)
        print(f"Fotis-AI: {reply}")
