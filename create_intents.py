import pandas as pd

# Define the data for intents.csv
data_content = {
    "intent": ["greeting", "pricing", "support", "goodbye", "thank_you"],
    "patterns": [
        "hello;hi;hey",
        "how much;cost;price",
        "need help;customer support",
        "bye;farewell;see you later",
        "thanks;thank you;appreciate it"
    ],
    "responses": [
        "Hello! How can I assist you?",
        "Our pricing information is available on our website.",
        "Our support team is here to help you!",
        "Goodbye! Have a great day!",
        "You're welcome! If you have any other questions, feel free to ask."
    ]
}

# Convert to DataFrame
new_data = pd.DataFrame(data_content)

# Save the DataFrame to intents.csv
new_data.to_csv(r"D:\customer_chatbot\Customer-Support-Chatbot\data\intents.csv", index=False)

print("intents.csv has been created successfully.")
