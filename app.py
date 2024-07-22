from flask import Flask, request, jsonify, send_from_directory
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering
from context_handler import ContextHandler

app = Flask(__name__)

# Load model and tokenizer from saved paths
model_save_path = './model'
tokenizer_save_path = './tokenizer'
squad_path = './squad.json'

# Load the tokenizer and model
tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_save_path)
model = XLMRobertaForQuestionAnswering.from_pretrained(model_save_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Load ContextHandler
context_handler = ContextHandler(squad_path)

def get_prediction(question):
    try:
        # Find the best context for the question
        context, question_id = context_handler.find_best_context(question)
        
        # Tokenize the input
        inputs = tokenizer.encode_plus(question, context, return_tensors='pt', truncation=True, padding=True).to(device)
        
        # Get model outputs
        outputs = model(**inputs)
        
        # Get the most likely beginning and end of answer with the argmax of the score
        answer_start = torch.argmax(outputs.start_logits).item()
        answer_end = torch.argmax(outputs.end_logits).item() + 1
        
        # Convert tokens to string
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        
        # Ensure the answer is extracted properly
        if answer.strip() == "":
            answer = "No answer found in the context."
        
        # Check if the context is valid
        if context.strip() == "":
            return {'answer': "No valid context found.", 'id': question_id}
        
        # Return answer with question ID
        return {'answer': answer, 'id': question_id}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'answer': "Error during prediction", 'id': None}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        question = data['question']
        result = get_prediction(question)
        return jsonify(result)
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

@app.route('/contexts', methods=['GET'])
def contexts():
    try:
        contexts = context_handler.contexts
        return jsonify({'contexts': contexts})
    except Exception as e:
        print(f"Error in /contexts route: {e}")
        return jsonify({'error': 'An error occurred while fetching contexts.'}), 500

@app.route('/similar_questions', methods=['POST'])
def similar_questions():
    try:
        data = request.json
        input_question = data['question']
        similar_questions = context_handler.find_similar_questions(input_question)
        return jsonify({'similar_questions': similar_questions})
    except Exception as e:
        print(f"Error in /similar_questions route: {e}")
        return jsonify({'error': 'An error occurred while fetching similar questions.'}), 500

if __name__ == '__main__':
    app.run(debug=True)