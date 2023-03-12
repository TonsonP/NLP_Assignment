# Import necessary libraries
import torch
from torch import nn
import torch.nn.functional as F
import torchdata
import torchtext

from tqdm import tqdm
import random, math, time
from torch.autograd import Variable

import pickle
file = open('/root/projects/NLP/Assignment/Transformer_Generator/obj/vocab_transforms.pkl', 'rb')
vocab_transform = pickle.load(file)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#make our work comparable if restarted the kernel
# SEED = 555
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

# Clean dataset a bit.

from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import re

nlp = spacy.load('en_core_web_md')

def preprocessing(sentence):
    
    # Clear the html tag by using regular expression.
    sentence = re.sub("<[^>]*>", "", sentence) # Filter html tag
    sentence = re.sub("[^\x00-\x7F]+", "", sentence) # Filter non-English
    sentence = re.sub("/[^a-zA-Z0-9 ]/", "", sentence) # Filter out some symbol
    #It matches any character which is not contained in the ASCII character set (0-127, i.e. 0x0 to 0x7F)
    doc = nlp(sentence)
    cleaned_tokens = []
    
    # This time "I WILL NOT FILTERS OUT STOPWORD" during the preprocessing as I think it is
    # necessary for story. For the punctuation I think it should be fine to filter out.
    for token in doc:
        if token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
            token.pos_ != 'SYM' and token.pos_!= 'X':
                cleaned_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(cleaned_tokens)

from torchtext.vocab import build_vocab_from_iterator
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(preprocessing(text))


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads #make sure it's divisible....

        self.fc_q = nn.Linear(hid_dim,hid_dim) 
        self.fc_k = nn.Linear(hid_dim,hid_dim) 
        self.fc_v = nn.Linear(hid_dim,hid_dim) 

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, q, k, v, mask = None):
        batch_size = q.shape[0]
        
        Q = self.fc_q(q)
        K = self.fc_k(k)
        V = self.fc_v(v)
        
        #Q, K, V = [b, l, h]
        #reshape them into head_dim
        #reshape them to [b, n_headm, l, head_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        #Q, K, V = [b, m_head, l, head_dim]

        #e = QK/sqrt(dk)
        e =  torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        #e = [b, n_heads, ql, kl]
        
        # torch.Size([64, 8, 50, 50])
        # torch.Size([64, 1, 1, 50, 256])

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e10)

        a = torch.softmax(e, dim=-1)
        #a = [batch size, n_heads, ql, kl]
                    
        #eV
        x = torch.matmul(self.dropout(a),V)
        #x : [b, n_heads, ql, head_di]

        x = x.permute(0, 2, 1, 3).contiguous()
        #x: [b, ql, n_heads, head_dim]

        #concat them together
        x = x.view(batch_size, -1, self.hid_dim)
        #x = [b, ql, h]

        x = self.fc(x)
        #x = [b, ql, h]

        return x, a
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        #x = [batch size, seq len, hid dim]
        
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.norm_ff = nn.LayerNorm(hid_dim) #second yellow box
        self.norm_maskedatt = nn.LayerNorm(hid_dim) #first red box
        
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.ff = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, trg_mask):
        #trg      : [b, l, h]
        #enc_src  : [b, sl, h]
        #trg_mask : [b, 1, tl, tl]
        #src_mask : [b, 1, 1, sl]

        #1st box : mask multi, add & norm
        _trg, attention = self.self_attention(trg, trg, trg, trg_mask) #Q, K, V
        _trg    = self.dropout(_trg)
        _trg    = trg + _trg
        trg     = self.norm_maskedatt(_trg)

        #2rd box : ff, add & norm
        _trg    = self.ff(trg)
        _trg    = self.dropout(_trg)
        _trg    = trg + _trg
        trg     = self.norm_ff(_trg)

        return trg, attention
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads,
                 pf_dim, dropout, device, src_pad_idx,trg_pad_idx, max_length = 100):
        super().__init__()
        self.pos_emb = nn.Embedding(max_length, hid_dim)
        self.trg_emb = nn.Embedding(output_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
                            [
                            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                            for _ in range(n_layers)
                            ]
                            )
        self.fc = nn.Linear(hid_dim, output_dim)
        self.device = device
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        #src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_mask : [batch size, 1, 1, trg len]
        
        trg_len = trg_mask.shape[-1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device =self.device)).bool() #lower triangle
        #trg_sub_mask = [trg len, trg len]
        trg_mask = trg_mask & trg_sub_mask 
        #trg_mask : [batch size, 1, trg len, trg len]
        return trg_mask     
    
    def forward(self, x):
        #src : = [batch size, trg len]
        #enc_src : hidden state from encoder = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = x.shape[0]
        trg_len = x.shape[1]
        
        src_mask = self.make_trg_mask(x)

        #pos
        pos = torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos = [batch size, trg len]

        pos_emb = self.pos_emb(pos) #[batch size, trg len, hid dim]
        trg_emb = self.trg_emb(x) #[batch size, trg len, hid dim]

        x = pos_emb + trg_emb * self.scale #[batch size, trg len, hid dim]
        x = self.dropout(x)
        
        for layer in self.layers: #output, hidden
            trg, attention = layer(x, src_mask)
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc(trg)
        #output = [batch size, trg len, output dim]

        return output, attention
    
class inference():
    def __init__(self,decoder):
        self.decoder = decoder

    #use during inference
    #encapsulates beam_decode or greedy_decode
    def decode(self, src, src_len, trg, hidden, method='beam-search'):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #src len = [batch size]

        # encoder_outputs, hidden = self.encoder(src, src_len) 
        #encoder_outputs = [src len, batch size, hid dim * 2]  (*2 because of bidirectional)(every hidden states)
        #hidden = [batch size, hid dim]  #final hidden state
       
        hidden = hidden.unsqueeze(0)
        #hidden = [1, batch size, hid dim]
        
        if method == 'beam-search':
            return self.beam_decode(src, trg, hidden)
        else:
            return self.greedy_decode(trg, hidden)

    def greedy_decode(self, trg, decoder_hidden, encoder_outputs=None):
        #trg = [batch size, src len]
        #decoder_hiddens = [1, batch size, hid dim]
        #encoder_outputs = [src len, batch size, hid dim * 2]

        seq_len, batch_size = trg.size()
        decoded_batch = torch.zeros((batch_size, seq_len))
        # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).cuda()
        decoder_input = trg[0, :] # sos
        # print(decoder_input.shape)

        # mask = self.create_mask(trg[:, idx].unsqueeze(1))

        for t in range(seq_len): 
            prediction, decoder_hidden = self.decoder(decoder_input)
            topv, topi = prediction.data.topk(1)  # [32, 10004] get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach().view(-1)

        return decoded_batch #(batch size, length)

    def beam_decode(self, src_tensor, target_tensor, decoder_hiddens, encoder_outputs=None):
        #src_tensor      = [src len, beam_decodebatch size]
        #target_tensor   = [trg len, batch size]
        #decoder_hiddens = [1, batch size, hid dim]
        #encoder_outputs = [src len, batch size, hid dim * 2]
        
        target_tensor = target_tensor.permute(1, 0)
        #target_tensor = [batch size, trg len]
        
        #how many parallel searches
        beam_width = 3
        
        #how many sentence do you want to generate
        topk = 1  
        
        #final generated sentence
        decoded_batch = []
                
        #Another difference is that beam_search_decoding has 
        #to be done sentence by sentence, thus the batch size is indexed and reduced to only 1.  
        #To keep the dimension same, we unsqueeze 1 dimension for the batch size.
        for idx in range(target_tensor.size(0)):  # batch_size
            
            #decoder_hiddens = [1, batch size, dec hid dim]
            decoder_hidden = decoder_hiddens[:, idx, :]
            #decoder_hidden = [1, dec hid dim]
            
            #encoder_outputs = [src len, batch size, enc hid dim * 2]
            # encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)
            #encoder_output = [src len, 1, enc hid dim * 2]
            
            # mask = self.create_mask(src_tensor[:, idx].unsqueeze(1))
            # print("mask shape: ", mask.shape)
            
            #mask = [1, src len]

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([SOS_IDX]).to(device)

            # Number of sentence to generate
            endnodes = []  #hold the nodes of EOS, so we can backtrack
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()  #this is a min-heap

            # start the queue
            nodes.put((-node.eval(), node))  #we need to put - because PriorityQueue is a min-heap
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                # score is log p divides by the length scaled by some constants
                score, n = nodes.get()
                            
                # wordid is simply the numercalized integer of the word
                decoder_input  = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_IDX and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                # decoder_input = SOS_IDX
                # decoder_hidden = [1, hid dim]
                # encoder_output = [src len, 1, hid dim * 2]
                # mask = [1, src len]
                
                # prediction, decoder_hidden, _ = self.decoder(decoder_input)
                #prediction     = [1, output dim]  #1 because the batch size is 1
                #decoder hidden = [1, hid dim]

                #so basically prediction is probabilities across all possible vocab
                #we gonna retrieve k top probabilities (which is defined by beam_width) and their indexes
                #recall that beam_width defines how many parallel searches we want
                log_prob, indexes = torch.topk(prediction, beam_width)
                # log_prob      = (1, beam width)
                # indexes       = (1, beam width)
                
                nextnodes = []  #the next possible node you can move to

                # we only select beam_width amount of nextnodes
                for top in range(beam_width):
                    pred_t = indexes[0, top].reshape(-1)  #reshape because wordid is assume to be []; see when we define SOS
                    log_p  = log_prob[0, top].item()
                                    
                    #decoder hidden, previous node, current node, prob, length
                    node = BeamSearchNode(decoder_hidden, n, pred_t, n.logp + log_p, n.len + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # Once everything is finished, choose nbest paths, back trace them
            
            ## in case it does not finish, we simply get couple of nodes with highest probability
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            #look from the end and go back....
            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace by looking at the previous nodes.....
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]  #reverse it....
                utterances.append(utterance) #append to the list of sentences....

            decoded_batch.append(utterances)

        return decoded_batch  #(batch size, length)
    
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h        = hiddenstate  #define the hidden state
        self.prevNode = previousNode  #where does it come from
        self.wordid   = wordId  #the numericalized integer of the word
        self.logp     = logProb  #the log probability
        self.len      = length  #the current length; first word starts at 1

    def eval(self, alpha=0.7):
        # the score will be simply the log probability penaltized by the length 
        # we add some small number to avoid division error
        # read https://arxiv.org/abs/1808.10006 to understand how alpha is selected
        return self.logp / float(self.len + 1e-6) ** (alpha)
    
    #this is the function for comparing between two beamsearchnodes, whether which one is better
    #it is called when you called "put"
    def __lt__(self, other):
        return self.len < other.len

    def __gt__(self, other):
        return self.len > other.len
    
output_dim  = len(vocab_transform)
hid_dim = 256
dec_layers = 12
dec_heads = 8
dec_pf_dim = 512
dec_dropout = 0.1

SRC_PAD_IDX = 1
TRG_PAD_IDX = 1

model = Decoder(output_dim, 
              hid_dim, 
              dec_layers, 
              dec_heads, 
              dec_pf_dim, 
              dec_dropout, 
              device,SRC_PAD_IDX,TRG_PAD_IDX).to(device)

model.load_state_dict(torch.load('/root/projects/NLP/Assignment/Transformer_Generator/models/Decoder.pt'))

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    # hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src)

            # print(prediction.shape)
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_md')

def generate_sentence(sentence):
    generation = generate(sentence, 50, 1, model, tokenizer, 
                          vocab_transform, device)
    return ' '.join(generation)