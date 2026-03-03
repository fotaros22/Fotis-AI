import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def fotis_ai_chat(prompt):
    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📬 Fotis‑AI Web Chatbot")
    user_input = gr.Textbox(
        label="🗨️ Your message",
        placeholder="Type a message here and press Enter"
    )
    output_text = gr.Textbox(label="🤖 Fotis‑AI says")
    user_input.submit(fotis_ai_chat, user_input, output_text)

if __name__ == "__main__":
    demo.launch()
