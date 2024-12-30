import pickle

from flask import Flask, request, jsonify

# Load the saved model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Code by GK-Rao")

app = Flask(__name__)

print("welcome to new branch")
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data as JSON
    data = request.get_json()
    features = data["Year"]  # Expecting input as {"features": [[value1], [value2], ...]}

    # Make prediction
    predictions = model.predict([features])
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
