# FINANCE-TINY  
### Fine-Tuning TinyLlama for Financial Sentiment Classification using LoRA

---

## Project Overview

FINANCE-TINY is a domain-specific chatbot built by fine-tuning **TinyLlama-1.1B-Chat-v1.0** using **Low-Rank Adaptation (LoRA)** for financial sentiment classification.

The model classifies financial sentences into three categories:

- **Positive**
- **Neutral**
- **Negative**

Instead of full fine-tuning, this project uses parameter-efficient LoRA adapters to specialize the model under limited GPU constraints.

---

##  Problem Statement

General-purpose language models do not automatically understand financial context.  
Words such as *“leveraged”*, *“volatile”*, or *“short”* carry domain-specific meanings that differ from everyday sentiment.

This project addresses the question:

> How can a small instruction-tuned language model be efficiently adapted for accurate financial sentiment classification without performing full fine-tuning?

---

## Dataset

- **Financial PhraseBank** (Hugging Face)
- Version: `sentences_allagree`
- 3-class sentiment dataset
- 90/10 train-test split
- Instruction-style formatting applied for generative classification

Example format:
Instruction: Classify the sentiment of this financial text.
Text: The company reported record quarterly revenue.
Answer: positive


---

##  Model & Fine-Tuning Setup

### Base Model
- TinyLlama-1.1B-Chat-v1.0
- ~1.1B parameters
- Instruction-tuned

### Fine-Tuning Method
- Low-Rank Adaptation (LoRA)
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`

### Training Framework
- Hugging Face `Trainer`
- AdamW optimizer
- Parameter-efficient training on Google Colab GPU

---

##  Experiments & Results

Two LoRA experiments were conducted to evaluate hyperparameter effects.

###  Performance Comparison

| Model | Learning Rate | Setup | Accuracy | Macro F1 | ROUGE-L |
|--------|--------------|--------|----------|----------|----------|
| Base TinyLlama | – | No fine-tuning | 0.269 | 0.201 | 0.400 |
| LoRA – Experiment 1 | 2e-4 | Short run | 0.559 | 0.239 | 0.555 |
| LoRA – Experiment 2 | 5e-5 | 2 epochs | **0.811** | **0.733** | **0.945** |

Final model improved accuracy from **26.9% → 81.1%**, demonstrating substantial gains from parameter-efficient domain adaptation.

---

##  Confusion Matrix (Final Model)

Rows = True Labels  
Columns = Predicted Labels  
Order: `[negative, neutral, positive]`

[[44 2 0]
[ 2 125 0]
[ 7 32 15]]


### Interpretation

- Strong performance on **negative** and **neutral** classes  
- Lower recall on **positive** class  
- Most misclassifications occur between positive and neutral  
- Indicates conservative prediction behavior and possible class imbalance influence  

---

## Deployment

The fine-tuned model was deployed using:

- **Gradio** for user interaction
- **Hugging Face Spaces** for hosting

The interface:
- Accepts financial sentences
- Returns predicted sentiment label
- Handles greetings and non-finance inputs gracefully

---

##  Repository Structure
finance-tiny-llm-lora/
│
├── Domain_Specific_Assistant_via_LLMs_Fine_Tuning.ipynb
├── requirements.txt
├── FINANCE-TINY_Report.pdf
└── README.md


---

##  Installation

Clone the repository:
git clone https://github.com/Solomon-the-octave/finance-tiny-llm-lora.git

cd finance-tiny-llm-lora


---

##  Key Takeaways

- LoRA enables efficient domain adaptation without full fine-tuning.
- Lightweight LLMs can achieve strong performance with proper hyperparameter tuning.
- Structured evaluation (accuracy, Macro F1, confusion matrix) is critical.
- Deployment highlights real-world inference challenges beyond notebook metrics.

---

##  Author

**Wengelawit Ayalew Solomon**  
Domain: Finance  
Summative – Chatbot Project  

---

