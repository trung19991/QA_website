import json
import requests
import os

# Constants
NUM_PARAGRAPHS = 20
NUM_QAS = 20

# Download the dataset
def download_dataset(url, cache_path='squad_data.json'):
    if not os.path.exists(cache_path):
        response = requests.get(url)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(response.json(), f)
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)

url = "https://trung19991.github.io/squad_data/squad.json"
data = download_dataset(url)

def create_lite_data(data, start_idx, end_idx, num_paragraphs=NUM_PARAGRAPHS, num_qas=NUM_QAS):
    lite_data = {"version": data["version"], "data": []}
    for article in data["data"][start_idx:end_idx]:
        lite_article = {"title": article["title"], "paragraphs": []}
        for paragraph in article["paragraphs"][:num_paragraphs]:
            lite_paragraph = {
                "context": paragraph["context"],
                "qas": paragraph["qas"][:num_qas]
            }
            lite_article["paragraphs"].append(lite_paragraph)
        lite_data["data"].append(lite_article)
    return lite_data

total_articles = len(data["data"])
train_size = int(total_articles * 0.7)
train_data = create_lite_data(data, 0, train_size)
val_data = create_lite_data(data, train_size, total_articles)

def save_data(data, filename):
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

save_data(train_data, "squad_data_train.json")
save_data(val_data, "squad_data_val.json")
print("Data has been split and saved into squad_data_train.json and squad_data_val.json")


import torch
from transformers import XLMRobertaForQuestionAnswering, XLMRobertaTokenizerFast, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

# Tokenization and Encoding
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        squad = json.load(f)
    contexts, questions, answers = [], [], []
    for group in squad['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers

train_contexts, train_questions, train_answers = read_data('squad_data_train.json')
valid_contexts, valid_questions, valid_answers = read_data('squad_data_val.json')

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2

add_end_idx(train_answers, train_contexts)
add_end_idx(valid_answers, valid_contexts)

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
valid_encodings = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_pos = answers[i]['answer_end'] - 1
        if end_pos >= 0:
            end_positions.append(encodings.char_to_token(i, end_pos))
        else:
            end_positions.append(None)
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(valid_encodings, valid_answers)

class SQuAD_Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SQuAD_Dataset(train_encodings)
valid_dataset = SQuAD_Dataset(valid_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

model = XLMRobertaForQuestionAnswering.from_pretrained("xlm-roberta-base")
model.to(device)
model.train()

N_EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 0
TOTAL_STEPS = len(train_loader) * N_EPOCHS

optim = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS)

for epoch in range(N_EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()
        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())

import torch
from transformers import XLMRobertaForQuestionAnswering, XLMRobertaTokenizerFast, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

# Tokenization and Encoding
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        squad = json.load(f)
    contexts, questions, answers = [], [], []
    for group in squad['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers

train_contexts, train_questions, train_answers = read_data('squad_data_train.json')
valid_contexts, valid_questions, valid_answers = read_data('squad_data_val.json')

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2

add_end_idx(train_answers, train_contexts)
add_end_idx(valid_answers, valid_contexts)

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
valid_encodings = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_pos = answers[i]['answer_end'] - 1
        if end_pos >= 0:
            end_positions.append(encodings.char_to_token(i, end_pos))
        else:
            end_positions.append(None)
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(valid_encodings, valid_answers)

class SQuAD_Dataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SQuAD_Dataset(train_encodings)
valid_dataset = SQuAD_Dataset(valid_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

model = XLMRobertaForQuestionAnswering.from_pretrained("xlm-roberta-base")
model.to(device)
model.train()

N_EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 0
TOTAL_STEPS = len(train_loader) * N_EPOCHS

optim = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS)

for epoch in range(N_EPOCHS):
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()
        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())
