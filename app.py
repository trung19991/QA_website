from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering

app = Flask(__name__)

# Load model and tokenizer from saved paths
model_save_path = './model'
tokenizer_save_path = './tokenizer'

# Load the tokenizer and model
tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_save_path)
model = XLMRobertaForQuestionAnswering.from_pretrained(model_save_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def get_prediction(context, question):
    try:
        # Tokenize the input
        inputs = tokenizer.encode_plus(question, context, return_tensors='pt', truncation=True, padding=True).to(device)
        
        # Get model outputs
        outputs = model(**inputs)
        
        # Get the most likely beginning and end of answer with the argmax of the score
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        # Convert tokens to string
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        return answer
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction"

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        context = data['context']
        question = data['question']
        answer = get_prediction(context, question)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
