import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load data from the intents.csv file
data = pd.read_csv(r"D:\customer_chatbot\Customer-Support-Chatbot\intents.csv")

# Split the patterns into a list and extract responses
patterns = data["patterns"].apply(lambda x: x.split(";"))
responses = data["responses"]

# Flatten the patterns list for vectorization
flattened_patterns = [pattern for sublist in patterns for pattern in sublist]
flattened_intents = data["intent"].repeat([len(p) for p in patterns]).reset_index(drop=True)

# Data preprocessing using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(flattened_patterns)
y = flattened_intents

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Print the model accuracy
print(f"Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# Prediction function to classify user intent
def classify_intent(user_input):
    user_input_vector = vectorizer.transform([user_input])
    intent = model.predict(user_input_vector)[0]
    response = data.loc[data["intent"] == intent, "responses"].values[0]
    return response

# Define the chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = classify_intent(user_input)
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000)
