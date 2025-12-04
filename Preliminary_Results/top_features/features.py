from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from flask_cors import CORS

# ------------ LLM CLASS ------------
class LocalLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", device="cpu"):
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded.")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": device})
        print("Model loaded!")

    def generate_features(self, product_name: str, brand: str):
        prompt = f"""
You are an expert product reviewer.

Product: {product_name}
Brand: {brand}

List the top 3 best features of this product and write a short usage example.
Return strictly in JSON:
{{ "feature1": "...", "feature2": "...", "feature3": "...", "usage_example": "..." }}
"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            result = {
                "feature1": "",
                "feature2": "",
                "feature3": "",
                "usage_example": text
            }

        return result



app = Flask(__name__)
CORS(app)

llm = LocalLLM()


@app.route("/generate-features", methods=["POST"])
def generate_features():
    data = request.get_json()

    product_name = data.get("product_name")
    brand = data.get("brand")

    if not product_name or not brand:
        return jsonify({"error": "Missing 'product_name' or 'brand'"}), 400

    result = llm.generate_features(product_name, brand)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


