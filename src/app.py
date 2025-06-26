import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
from transformers.utils import logging as hf_logging

# Hide warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()


from dag_graph import build_graph
import logging
from datetime import datetime

# Check log directory
os.makedirs("logs", exist_ok=True)

# Setup logger
logging.basicConfig(filename="logs/logfile.txt", level=logging.INFO)

# Build LangGraph DAG
dag = build_graph()

def log_result(input_text, label, confidence, fallback_used):
    logging.info(f"{datetime.now()} | Input: {input_text} | Label: {label} | Confidence: {confidence:.2f} | Fallback: {fallback_used}")

def main():
    print("LangGraph Self-Healing Classifier (Type 'exit' to quit)\n")

    while True:
        user_input = input("Enter text to classify: ")
        if user_input.lower() == "exit":
            break

        try:
            data = dag.invoke({"input_text": user_input})

            print(f"\nInput: {user_input}")
            print(f"[InferenceNode] Predicted label: {data.get('initial_prediction', 'N/A')} | Confidence: {data.get('confidence', 0)*100:.0f}%")

            if data.get("fallback_invoked", False):
                print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
                print(f"[FallbackNode] Could you clarify your intent? Was this a negative review?")
                print(f"User: {data.get('clarification', 'N/A')}")
                print(f"Final Label: {data['prediction']} (Corrected via user clarification)")
            else:
                print(f"Final Label: {data['prediction']} (Accepted model prediction)")

            print("-" * 50)

            # Log result
            log_result(user_input, data["prediction"], data.get("confidence", 0), data.get("fallback_invoked", False))

        except Exception as e:
            print(f"Error during classification: {e}")
            logging.error(f"{datetime.now()} | ERROR | {str(e)}")

if __name__ == "__main__":
    main()