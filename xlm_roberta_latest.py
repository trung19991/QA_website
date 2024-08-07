# -*- coding: utf-8 -*-
"""xlm_roberta_latest.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LenuxXedIOIBcOQodWLhoChorNm-r9si
"""

import json
import requests

# Constants
NUM_PARAGRAPHS = 20
NUM_QAS = 20

# Download the dataset
url = "https://trung19991.github.io/squad_data/squad.json"
response = requests.get(url)
data = response.json()

# Function to create a lite version of the data
def create_lite_data(data, start_idx, end_idx):
    lite_data = {
        "version": data["version"],
        "data": []
    }
    for article in data["data"][start_idx:end_idx]:
        lite_article = {
            "title": article["title"],
            "paragraphs": []
        }
        for paragraph in article["paragraphs"][:NUM_PARAGRAPHS]:
            lite_paragraph = {
                "context": paragraph["context"],
                "qas": paragraph["qas"][:NUM_QAS]  # Take the first NUM_QAS Q&A pairs
            }
            lite_article["paragraphs"].append(lite_paragraph)
        lite_data["data"].append(lite_article)
    return lite_data

# Split data into training (70%) and validation (30%) parts
total_articles = len(data["data"])
train_size = int(total_articles * 0.7)

# Create the training data
train_data = create_lite_data(data, 0, train_size)

# Create the validation data
val_data = create_lite_data(data, train_size, total_articles)

# Save the lite data to JSON files
with open("squad_data_train.json", "w", encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("squad_data_val.json", "w", encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print("Data has been split and saved into squad_data_train.json and squad_data_val.json")



"""# **Question Answering❓**
with fine-tuned BERT on SQuAD 2.0.  

Question answering comes in many forms. We’ll look at the particular type of extractive QA that involves answering a question about a passage by highlighting the segment of the passage that answers the question. This involves fine-tuning a model which predicts a start position and an end position in the passage. More specifically, we will fine tune the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model on the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

I have followed [this tutorial](https://huggingface.co/transformers/v3.2.0/custom_datasets.html#question-answering-with-squad-2-0) from the huggingface community for how to fine tune BERT on custom datasets which in our case is the SQuAD 2.0.

**Some first imports**
"""

import requests
import json
import torch
import os
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

"""**Connecting Google Drive in order to save the model**"""

if not os.path.exists('/content/drive/MyDrive/BERT-SQuAD'):
  os.mkdir('/content/drive/MyDrive/BERT-SQuAD')

pip install transformers

"""### **Download SQuAD 2.0 ⬇️**

SQuAD consists of two json files.

* train dataset
* validation dataset
"""

# !wget -nc https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# !wget -nc https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

"""## **Data preprocessing 💽**

In this section of data preprocessing, our goal is to get our data in the following form:

<div>
<img src="http://www.mccormickml.com/assets/BERT/SQuAD/input_formatting.png" width="650"/>
</div>


In short, we have to do the following:

1. Extract the data from the jsons files
2. Tokenize the data
3. Define the datasets
"""

# Load the training dataset and take a look at it
with open('squad_data_final.json', 'rb') as f:
  squad = json.load(f)

# Each 'data' dict has two keys (title and paragraphs)
squad['data'][0].keys()

# Find the group about Greece
gr = -1
for idx, group in enumerate(squad['data']):
  print(group['title'])
  if group['title'] == 'Greece':
    gr = idx
    print(gr)
    break

# let's check on Greece which is 186th (0-based indexing)
# we can see that we have a context and many questions and answers following
squad['data'][0]

# and this is the context given for NYC
squad['data'][0]

"""### **Get data 📁**

After we got a taste of the jsons files data format let's extract our data and store them into some data structures.
"""

def read_data(path):
  # load the json file
  with open(path, 'rb') as f:
    squad = json.load(f)

  contexts = []
  questions = []
  answers = []

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

"""Put the contexts, questions and answers for training and validation into the appropriate lists."""

train_contexts, train_questions, train_answers = read_data('squad_data_final.json')
valid_contexts, valid_questions, valid_answers = read_data('squad_data_final.json')

# print a random question and answer
print(f'There are {len(train_questions)} questions')
print(train_questions[-1])
print(train_answers[-1])

"""As you can see above, the answers are dictionaries whith the answer text and an integer which indicates the start index of the answer in the context. As the SQuAD does not give us the end index of the answer in the context we have to find it ourselves. So, let's get the character position at which the answer ends in the passage. Note that sometimes SQuAD answers are off by one or two characters, so we will also adjust for that."""

def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # sometimes squad answers are off by a character or two so we fix this
    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
      answer['answer_start'] = start_idx - 1
      answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
      answer['answer_start'] = start_idx - 2
      answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(valid_answers, valid_contexts)

# You can see that now we get the answer_end also
print(train_questions[-1])
print(train_answers[-1])

"""### **Tokenization 🔢**

As we know we have to tokenize our data in form that is acceptable for the BERT model. We are going to use the `BertTokenizerFast` instead of `BertTokenizer` as the first one is much faster. Since we are going to train our model in batches we need to set `padding=True`.
"""

from transformers import XLMRobertaTokenizerFast

tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
valid_encodings = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)

"""Let's see what we got after tokenizing our data."""

train_encodings.keys()

no_of_encodings = len(train_encodings['input_ids'])
print(f'We have {no_of_encodings} context-question pairs')

train_encodings['input_ids'][0]

"""Let's decode the first pair of context-question encoded pair and look into it."""

tokenizer.decode(train_encodings['input_ids'][0])

"""We can see that each word is assigned a number.

For example,

beyonce $\rightarrow$ 20773  
[CLS] $\rightarrow$ 101  
[SEP] $\rightarrow$ 102   
[PAD] $\rightarrow$ 0  

We see that the above form matches the one in the image we saw in the Data preprocessing section before.

Next we need to convert our character start/end positions to token start/end positions. Why is that? Because our words converted into tokens, so the answer start/end needs to show the index of start/end token which contains the answer and not the specific characters in the context.
"""

def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    # Handle potential negative end position
    end_pos = answers[i]['answer_end'] - 1
    if end_pos >= 0:
      end_positions.append(encodings.char_to_token(i, end_pos))
    else:
      end_positions.append(None)  # or handle as needed

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = tokenizer.model_max_length

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(valid_encodings, valid_answers)

train_encodings['start_positions'][:10]

"""### **Dataset definition 🗄️**

We have to define our dataset using the PyTorch Dataset class from `torch.utils` in order create our dataloaders after that.
"""

class SQuAD_Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)

train_dataset = SQuAD_Dataset(train_encodings)
valid_dataset = SQuAD_Dataset(valid_encodings)

"""### **Dataloaders 🔁**"""

from torch.utils.data import DataLoader

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)

"""## **Fine-Tuning ⚙️**

### **Model definition 🤖**

We are going to use the `bert-case-uncased` from the huggingface transformers.
"""

from transformers import XLMRobertaForQuestionAnswering

model = XLMRobertaForQuestionAnswering.from_pretrained("xlm-roberta-base")

"""### **Training 🏋️‍♂️**

Μy choices for some parameters:

* Use of `AdamW` which is a stochastic optimization method that modifies the typical implementation of weight decay in Adam, by decoupling weight decay from the gradient update. This helps to avoid overfitting which is necessary in this case were the model is very complex.

* Set the `lr=5e-5` as I read that this is the best value for the learning rate for this task.
"""

# Check on the available device - use GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

N_EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 0
TOTAL_STEPS = len(train_loader) * N_EPOCHS

optim = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=WARMUP_STEPS, num_training_steps=TOTAL_STEPS)

model.to(device)
model.train()

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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()
        scheduler.step()

        loop.set_description(f'Epoch {epoch+1}')
        loop.set_postfix(loss=loss.item())

"""**Save the model in my drive in order not to run it each time**"""

#model_path = '/content/drive/MyDrive/BERT-SQuAD'
#model.save_pretrained(model_path)
#tokenizer.save_pretrained(model_path)

"""**Respectively, load the saved model**"""

#from transformers import BertForQuestionAnswering, BertTokenizerFast

#model_path = '/content/drive/MyDrive/BERT-SQuAD'
#model = BertForQuestionAnswering.from_pretrained(model_path)
#tokenizer = BertTokenizerFast.from_pretrained(model_path)

#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#print(f'Working on {device}')

#model = model.to(device)

"""### **Testing ✅**

We are evaluating the model on the validation set by checking the model's predictions for the answer's start and end indexes and comparing with the true ones.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

model.eval()

acc = []
f1_scores = []
em_scores = []

for batch in tqdm(valid_loader):
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_true = batch['start_positions'].to(device)
        end_true = batch['end_positions'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)

        # Accuracy
        acc.append(((start_pred == start_true).sum() / len(start_pred)).item())
        acc.append(((end_pred == end_true).sum() / len(end_pred)).item())

        # F1 Score
        f1_start = f1_score(start_true.cpu(), start_pred.cpu(), average='macro')
        f1_end = f1_score(end_true.cpu(), end_pred.cpu(), average='macro')
        f1_scores.append((f1_start + f1_end) / 2)

        # Exact Match
        em = ((start_pred == start_true) & (end_pred == end_true)).float().mean().item()
        em_scores.append(em)

acc = sum(acc) / len(acc)
f1 = sum(f1_scores) / len(f1_scores)
em = sum(em_scores) / len(em_scores)

print(f"\n\nAccuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Exact Match: {em:.4f}\n")

print("T/P\tanswer_start\tanswer_end\n")
for i in range(len(start_true)):
    print(f"true\t{start_true[i]}\t{end_true[i]}\n"
          f"pred\t{start_pred[i]}\t{end_pred[i]}\n")

"""### **Ask questions 🙋**

We are going to use some functions from the [*official Evaluation Script v2.0*](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/) of SQuAD in order to test the fine-tuned model by asking some questions given a context. I have also looked at this [notebook](https://colab.research.google.com/github/fastforwardlabs/ff14_blog/blob/master/_notebooks/2020-06-09-Evaluating_BERT_on_SQuAD.ipynb#scrollTo=MzPlHgWEBQ8D) which evaluates BERT on SQuAD.
"""

def get_prediction(context, question):
  inputs = tokenizer.encode_plus(question, context, return_tensors='pt').to(device)
  outputs = model(**inputs)

  answer_start = torch.argmax(outputs[0])
  answer_end = torch.argmax(outputs[1]) + 1

  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

  return answer

def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re
  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)
  def white_space_fix(text):
    return " ".join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0

  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)

  return round(2 * (prec * rec) / (prec + rec), 2)

def question_answer(context, question,answer):
  prediction = get_prediction(context,question)
  em_score = exact_match(prediction, answer)
  f1_score = compute_f1(prediction, answer)

  print(f'Question: {question}')
  print(f'Prediction: {prediction}')
  print(f'True Answer: {answer}')
  print(f'Exact match: {em_score}')
  print(f'F1 score: {f1_score}\n')



"""**Beyoncé**"""

context = """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer,
          songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing
          and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child.
          Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time.
          Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide,
          earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""


questions = ["For whom the passage is talking about?",
             "When did Beyonce born?",
             "Where did Beyonce born?",
             "What is Beyonce's nationality?",
             "Who was the Destiny's group manager?",
             "What name has the Beyoncé's debut album?",
             "How many Grammy Awards did Beyonce earn?",
             "When did the Beyoncé's debut album release?",
             "Who was the lead singer of R&B girl-group Destiny's Child?"]

answers = ["Beyonce Giselle Knowles - Carter", "September 4, 1981", "Houston, Texas",
           "American", "Mathew Knowles", "Dangerously in Love", "five", "2003",
           "Beyonce Giselle Knowles - Carter"]

for question, answer in zip(questions, answers):
  question_answer(context, question, answer)

"""**Athens**"""

context = """Athens is the capital and largest city of Greece. Athens dominates the Attica region and is one of the world's oldest cities,
             with its recorded history spanning over 3,400 years and its earliest human presence starting somewhere between the 11th and 7th millennium BC.
             Classical Athens was a powerful city-state. It was a center for the arts, learning and philosophy, and the home of Plato's Academy and Aristotle's Lyceum.
             It is widely referred to as the cradle of Western civilization and the birthplace of democracy, largely because of its cultural and political impact on the European continent—particularly Ancient Rome.
             In modern times, Athens is a large cosmopolitan metropolis and central to economic, financial, industrial, maritime, political and cultural life in Greece.
             In 2021, Athens' urban area hosted more than three and a half million people, which is around 35% of the entire population of Greece.
             Athens is a Beta global city according to the Globalization and World Cities Research Network, and is one of the biggest economic centers in Southeastern Europe.
             It also has a large financial sector, and its port Piraeus is both the largest passenger port in Europe, and the second largest in the world."""

questions = ["Which is the largest city in Greece?",
             "For what was the Athens center?",
             "Which city was the home of Plato's Academy?"]

answers = ["Athens", "center for the arts, learning and philosophy", "Athens"]

for question, answer in zip(questions, answers):
  question_answer(context, question, answer)

"""**Angelos**"""

 install Flask

