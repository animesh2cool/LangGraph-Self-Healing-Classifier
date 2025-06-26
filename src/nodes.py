from model_loader import load_model
import torch.nn.functional as F
import torch
import logging

model, tokenizer, device = load_model()

def inference_node(state):
    input_text = state["input_text"]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    label = "Positive" if predicted.item() == 1 else "Negative"

    return {
        **state,
        "initial_prediction": label,
        "confidence": confidence.item(),
        "prediction": label,
        "fallback_invoked": False
    }


def confidence_check_node(state: dict) -> dict:
    threshold = 0.7
    confidence = state["confidence"]

    if confidence < threshold:
        logging.info(f"[ConfidenceCheckNode] Confidence {confidence:.2f} below threshold. Triggering fallback.")
        return {
            **state,
            "fallback": True,
            "fallback_invoked": True
        }
    else:
        logging.info(f"[ConfidenceCheckNode] Confidence {confidence:.2f} above threshold. Proceeding.")
        return {
            **state,
            "fallback": False,
            "fallback_invoked": False
        }


def fallback_node(state: dict) -> dict:
    print(f"[FallbackNode] Could you clarify your intent? Was this a negative review?")
    clarification = input("User: ").strip().lower()

    if "yes" in clarification:
        final_label = "Negative"
        note = "(Corrected via user clarification)"
    elif "no" in clarification:
        final_label = "Positive"
        note = "(Corrected via user clarification)"
    else:
        final_label = state.get("prediction", "Unknown")
        note = "(No clear clarification. Kept model prediction)"

    logging.info(f"[FallbackNode] User clarification received: {clarification} → Final label: {final_label}")

    return {
        **state,
        "clarification": clarification,
        "prediction": final_label,
        "fallback_invoked": True,
        "clarification_note": note
    }