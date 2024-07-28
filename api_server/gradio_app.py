import gradio as gr
import torch
from model_utils import load_model_for_inference

from src.metadata import metadata

# load the model;
model = load_model_for_inference(
    metadata.SAVED_MODELS_PATH / "latest-bf16.ckpt"
)
model.eval()


def predict(raw_instruction: str, temperature=0) -> str:
    flow = {"instruction": raw_instruction, "input": ""}
    with torch.no_grad():
        answer = model.get_response(flow, temperature=temperature)
    return answer


# the gradio interface;
iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Chat with GPT2",
    description="Demo for chat interface",
)


if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=5000)
