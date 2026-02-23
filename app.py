import re
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_REPO = "Wengelawiit/finance-tinyllama-lora"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(device)

model = PeftModel.from_pretrained(base, ADAPTER_REPO).to(device)
model.eval()

try:
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
except Exception:
    pass

LABELS = ["negative", "neutral", "positive"]

label_token_ids = {
    lab: tokenizer(" " + lab, add_special_tokens=False)["input_ids"]
    for lab in LABELS
}

finance_words = [
    "stock","shares","profit","profits","loss","losses","revenue","earnings","dividend","market",
    "bank","loan","interest","inflation","bond","equity","merger","acquisition",
    "ipo","valuation","cash","cashflow","forecast","guidance","quarter","q1","q2","q3","q4",
    "ceo","cfo","board","layoffs","bankruptcy","debt","default","margin","miss","downgrade"
]

def looks_finance(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in finance_words)

def is_greeting(text: str) -> bool:
    t = (text or "").lower().strip()
    return t in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

@torch.inference_mode()
def score_label_with_cache(prompt_ids, lab_ids) -> float:
    """
    Score P(label | prompt) using cached past_key_values.
    Returns average log-prob per label token (length-normalized).
    """
    # Run prompt once to get cache
    prompt = torch.tensor([prompt_ids], device=device)
    out = model(input_ids=prompt, use_cache=True)
    past = out.past_key_values

    logp_sum = 0.0
    prev_token = prompt[:, -1:]  

    for tok_id in lab_ids:
        step = model(input_ids=prev_token, past_key_values=past, use_cache=True)
        logits = step.logits[:, -1, :] 
        logp_sum += torch.log_softmax(logits, dim=-1)[0, tok_id].item()

        past = step.past_key_values
        prev_token = torch.tensor([[tok_id]], device=device)

    return logp_sum / max(len(lab_ids), 1)

@torch.inference_mode()
def predict_label(msg: str) -> str:
    prompt = (
        "Instruction: Classify the sentiment of this financial text.\n"
        f"Text: {msg.strip()}\n"
        "Answer:"
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    scores = {lab: score_label_with_cache(prompt_ids, label_token_ids[lab]) for lab in LABELS}
    return max(scores, key=scores.get)

def chat(msg, history):
    msg = (msg or "").strip()

    if is_greeting(msg):
        return "Hi! Paste a finance sentence and I’ll label it as positive, neutral, or negative."

    if not looks_finance(msg):
        return "I’m trained for finance sentiment. Please paste a finance-related sentence (profits, shares, revenue, loans, etc.)."

    return predict_label(msg)

ui = gr.ChatInterface(chat, title="Finance Assistant (TinyLlama + LoRA)")
ui.launch()
