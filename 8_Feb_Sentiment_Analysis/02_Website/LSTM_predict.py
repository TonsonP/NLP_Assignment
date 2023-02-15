from LSTM_model import *
import torch
import pickle
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import praw

reddit = praw.Reddit(client_id='j6aKGP9tTKvhM4lnupSwoQ', 
                     client_secret='YMDOCKQut_2kl1rvrM595ITveoktag', 
                     user_agent='Tonsonplaydota',
                     check_for_async=False)


# Load tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_md')

# Get vocab data
file = open('/root/projects/NLP/Assignment/8_Feb_Sentiment_Analysis/data/vocab.pkl','rb')
vocab = pickle.load(file)
device = 'cpu'

def get_post(sub_r):
    post_list = list()
    hot_posts = reddit.subreddit(sub_r).new(limit=50)
    for post in hot_posts:
        post_list.append(post.title)
    return post_list


def yield_tokens(data_iter): #data_iter, e.g., train
    for _, text in data_iter: 
        yield tokenizer(text)

#hyper-parameter
input_dim  = 17136
hid_dim    = 256
emb_dim    = 300         
output_dim = 5
num_layers = 2
bidirectional = True
dropout = 0.5

model = LSTM(input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout, 1).to(device)
weight_path = '/root/projects/NLP/Assignment/8_Feb_Sentiment_Analysis/weights/LSTM.pt'
model.load_state_dict(torch.load(weight_path))

text_pipeline  = lambda x: vocab(tokenizer(x))

def predict(text, text_length):
    with torch.no_grad():
        output = model(text, text_length).squeeze(1)
        predicted = torch.max(output.data, 1)[1]
        return predicted

def sentence_checking(test_list):
    predict_list = list()
    for sent in test_list:
        text = torch.tensor(text_pipeline(sent)).to(device)
        text_list = [x.item() for x in text]
        text = text.reshape(1, -1)
        text_length = torch.tensor([text.size(1)]).to(dtype=torch.int64)
        result = predict(text, text_length)
        predict_list.append(result.cpu().data.numpy()[0])
    return predict_list

def check_range(pred_list):
    pred_list_avg = sum(pred_list) / len(pred_list)

    if pred_list_avg < 3:
        return 'Negative'
    elif pred_list_avg > 3:
        return 'Positive'
    else:
        return "Ummmm, Don't know"
