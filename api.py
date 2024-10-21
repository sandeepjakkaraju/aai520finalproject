from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from transformers import BertTokenizer

# Initialize the Flask app
app = Flask(__name__)

# Load the trained chatbot model and tokenizer
model = load_model('chatbot_model.h5')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to generate a response
def generate_response(input_text):
    # Tokenize the input text
    input_seq = tokenizer.encode(input_text, return_tensors='tf')

    # Generate a response from the model (use your decoding logic here)
    outputs = model.predict([input_seq,input_seq])  # Add your function here

    print(outputs)
 # Decode the generated token IDs into a string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response


# Define the endpoint for the chatbot API
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')
    
    if user_input:
        # Generate chatbot response
        response = generate_response(user_input)
        return jsonify({"response": response})
    
    return jsonify({"error": "No input provided"}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

