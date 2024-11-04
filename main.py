import pandas as pd
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

# Load data from the intents.csv file
data = pd.read_csv(r"D:\customer_chatbot\Customer-Support-Chatbot\intents.csv")

# Prepare the dataset
patterns = data["patterns"].apply(lambda x: x.split(";"))
responses = data["responses"]

# Flatten the patterns list for training
flattened_patterns = [pattern for sublist in patterns for pattern in sublist]
flattened_intents = data["intent"].repeat([len(p) for p in patterns]).reset_index(drop=True)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input patterns
train_encodings = tokenizer(flattened_patterns, truncation=True, padding=True, max_length=32)

# Create a tensor dataset
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Map intents to numerical labels
label_map = {label: idx for idx, label in enumerate(set(flattened_intents))}
num_labels = len(label_map)
encoded_labels = [label_map[intent] for intent in flattened_intents]

# Create the dataset
dataset = IntentDataset(train_encodings, encoded_labels)

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
model.eval()  # Set model to evaluation mode

# Define a prediction function to classify user intent
def classify_intent(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=32)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    response = data.loc[data["intent"] == list(label_map.keys())[predicted_class_id], "responses"].values[0]
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
