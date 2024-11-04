from flask import Flask, request, jsonify

app = Flask(__name__)

# Basic GET route for testing
@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Chatbot API!"

# Function to handle chatbot responses
def basic_chatbot(user_input):
    responses = {
        "hello": "Hi there! How can I help you?",
        "pricing": "Our pricing plans are available on our website.",
        "support": "Please contact our support team at support@example.com."
    }
    for keyword, response in responses.items():
        if keyword in user_input.lower():
            return response
    return "I'm sorry, I don't understand. Can you rephrase?"

# POST route for chatbot interaction
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = basic_chatbot(user_input)
    return jsonify({"response": response})

# Run the Flask application
if __name__ == "__main__":
    app.run(port=5000, debug=True)  # Debug mode enabled
