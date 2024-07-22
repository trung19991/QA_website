import json
import torch
from sentence_transformers import SentenceTransformer, util

class ContextHandler:
    def __init__(self, squad_path):
        self.squad_path = squad_path
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.contexts, self.questions, self.question_ids, self.answers = self.load_data()
        self.context_embeddings = self.embedder.encode(self.contexts, convert_to_tensor=True)
        self.question_embeddings = self.embedder.encode(self.questions, convert_to_tensor=True)
    
    def load_data(self):
        with open(self.squad_path, 'r', encoding='utf-8') as file:
            squad_data = json.load(file)
        
        contexts = []
        questions = []
        question_ids = []
        answers = []
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                contexts.append(context)
                for qa in paragraph['qas']:
                    questions.append(qa['question'])
                    question_ids.append(qa['id'])
                    answers.append(qa['answers'][0]['text'] if qa['answers'] else "No answer")
        
        return contexts, questions, question_ids, answers
    
    def find_best_context(self, question):
        question_embedding = self.embedder.encode(question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(question_embedding, self.context_embeddings)
        best_context_index = torch.argmax(similarities).item()
        
        if similarities[0][best_context_index] < 0.5:  # Adjust threshold as needed
            return None, None
        
        return self.contexts[best_context_index], self.answers[best_context_index]
    
    def find_similar_questions(self, input_question, threshold=0.5):
        input_question_embedding = self.embedder.encode(input_question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(input_question_embedding, self.question_embeddings)
        similar_questions = []
        for i, similarity in enumerate(similarities[0]):
            if similarity.item() >= threshold:
                similar_questions.append(self.questions[i])
        return similar_questions
