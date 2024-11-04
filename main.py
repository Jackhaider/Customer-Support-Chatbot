from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define the basic chatbot function
def basic_chatbot(user_input):
    # Define keyword-based responses
    responses = {
        "hello": "Hi there! How can I help you?",
        "pricing": "Our pricing plans are available on our website.",
        "support": "Please contact our support team at support@example.com."
    }
    
    # Check if any keyword is in the user's input
    for keyword, response in responses.items():
        if keyword in user_input.lower():
            return response
    return "I'm sorry, I don't understand. Can you rephrase?"

# Define the chat route to handle POST requests
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = basic_chatbot(user_input)
    return jsonify({"response": response})

# Run the app
if __name__ == "__main__":
    app.run(port=5000)
