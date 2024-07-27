import json

from flask import Flask, jsonify, request
from model_utils import load_model_for_inference

from src.metadata import metadata

app = Flask(__name__)

model = load_model_for_inference(
    metadata.SAVED_MODELS_PATH / "latest-bf16.ckpt",
)
model.eval()


# heartbeat endpoint;
@app.route("/")
def index():
    return "hello world!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        flow = {
            "instruction": data.get("instruction", ""),
            "input": data.get("input", ""),
        }
        temperature = data.get("temperature", 0)
        if flow["instruction"] == "":
            flow["instruction"] = flow["input"]
            flow["input"] = ""
        if not (flow["instruction"] or flow["input"]):
            return jsonify(
                {
                    "statusCode": 400,  # Bad Request
                    "headers": {
                        "Content-Type": "application/json",
                    },
                    "body": "Empty string provided! Provide at least one of 'instruction' or 'input'",
                }
            )
        print(flow, temperature)
        answer = model.get_response(flow, temperature=temperature)
        print("answer computed")
        return jsonify(
            {
                "statusCode": 200,  # OK
                "headers": {
                    "Content-Type": "application/json",
                },
                "body": json.dumps({"answer": answer}),
            }
        )
    except Exception as e:
        print(repr(e))
        return jsonify(
            {
                "statusCode": 500,  # Internal Server Error;
                "headers": {
                    "Content-Type": "application/json",
                },
                "body": json.dumps({"error": repr(e)}),
            }
        )


def main():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
