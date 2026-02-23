FINANCE-TINY: Financial Sentiment Classification using LoRA
Overview

This project builds a domain-specific financial sentiment assistant by fine-tuning TinyLlama-1.1B-Chat using Low-Rank Adaptation (LoRA).

The model classifies financial sentences into:

Positive

Neutral

Negative

Instead of full fine-tuning, parameter-efficient LoRA adapters were used to specialize the model under limited GPU constraints.

Dataset

Financial PhraseBank (sentences_allagree version)

3-class sentiment classification

90/10 train-test split

Model

Base model: TinyLlama-1.1B-Chat-v1.0

Fine-tuning method: LoRA

LoRA configuration:

Rank (r): 16

Alpha: 32

Dropout: 0.05

Target modules: q_proj, k_proj, v_proj, o_proj

Experiments & Results
Model	Accuracy	Macro F1	ROUGE-L
Base	0.269	0.201	0.400
LoRA Exp1	0.559	0.239	0.555
LoRA Exp2	0.811	0.733	0.945

Final model achieved 81.1% accuracy, demonstrating significant improvement over the baseline.

Confusion Matrix (Final Model)

Rows = true labels
Columns = predicted labels
Order: [negative, neutral, positive]

[[44   2   0]
 [ 2 125   0]
 [ 7  32  15]]
Deployment

The fine-tuned model was deployed using Gradio and hosted on Hugging Face Spaces.

Files

Domain_Specific_Assistant_via_LLMs_Fine_Tuning.ipynb – Training & evaluation notebook

app.py – Gradio deployment script

FINANCE-TINY_Report.pdf – Final report
