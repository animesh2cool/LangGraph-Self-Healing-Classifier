# LangGraph Self-Healing Classifier

This project is a self-correcting sentiment classifier using **LangGraph**, **LoRA fine-tuned DistilBERT**, and a CLI. The classifier predicts Positive or Negative sentiment and falls back to user clarification if the model confidence is too low.

---

# Fine-Tuning

Used LoRA (Low-Rank Adaptation) to fine-tune `distilbert-base-uncased` for binary sentiment classification.

---
# Requirements

Install dependencies:

pip install -r requirements.txt

---

# Launching the LangGraph DAG

The DAG (Directed Acyclic Graph) defines a simple flow:
1. Run inference on input text
2. Check prediction confidence
3. If the confidence is too low, ask user for clarification
4. Return final label

# Run the CLI App

python app.py

---

# Example: High Confidence Flow

Enter text to classify: The movie was excellent!

Input: The movie was excellent!
[InferenceNode] Predicted label: Positive | Confidence: 93%
Final Label: Positive (Accepted model prediction)


# Example: Fallback with Clarification

Enter text to classify: You should not watch this movie but you can if you have time.

Input: The movie was painfully slow and boring.
[InferenceNode] Predicted label: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?
User: Yes, it was definitely negative.
Final Label: Negative (Corrected via user clarification)

---

# Logs

All predictions and clarifications are logged in:

src/logs/logfile.txt

---

# Video Link : https://www.loom.com/share/7cca728a57bd4eadb75f4a500ef154c8?sid=f449d9f9-a848-4f7d-84d9-d318a3ca770b
