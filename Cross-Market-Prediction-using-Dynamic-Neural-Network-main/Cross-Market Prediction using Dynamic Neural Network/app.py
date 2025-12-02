from flask import Flask, render_template, request, jsonify
import os
import json

app = Flask(__name__, static_folder="static", template_folder="templates")

# TODO: replace this placeholder with your model-loading/prediction logic.
# If your repo already has a prediction function, import it here, e.g.:
# from your_module import predict_from_features
def predict_from_input(input_data: dict) -> dict:
    # Example placeholder response to show the flow.
    # Replace this with the repo's real prediction call.
    return {
        "status": "ok",
        "input_received": input_data,
        "prediction": "placeholder_prediction"
    }

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Accept JSON or form data
        if request.is_json:
            payload = request.get_json()
        else:
            payload = request.form.to_dict()

        # Convert to dict as needed by your model/predict function
        result = predict_from_input(payload)
        # Return JSON response (used by the frontend)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
