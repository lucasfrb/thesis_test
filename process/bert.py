
import pandas as pd

from typing import Callable, List, Optional, Tuple

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from detect_sentiment import sentiment_analyzer_scores
from statistics import mean
import gc
from numba import cuda 

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../pre_process')
from tokenizer import Normal_Tokenizer, Yake_Tokenizer, My_Tokenizer
from emoticons import EMOJIS
from post import Post

def spplit(string):
    return re.split('[,;\.?!\\\\* ]+', string)


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

filename = 'log.txt'

def save(headline, value):
    with open(filename, 'a') as f:
        try:
            f.write(headline + '\n')
            f.write(value + "\n")
        except:
            print("post cant be printed")

class Emojis():

    def _find_emojis(self, string):
        if string == '':
            return ''
        if string is None:
            return ''

        for key in EMOJIS:
            builder = [' ' + key, ' ' + key + ' ', key + ' ']
            if any(b in string for b in builder):
                #print(string)
                string = string.replace(key, EMOJIS[key])

        return string

    def transform(self, lista):
        print("emojis")
        save('PHASE', 'EMOJIS')
        l = list()
        for sub in lista:
            sub_list = list()
            for post in sub:
                if post.real_form.text == '':
                    #save('retrieve emojis - BEFORE', str(post))
                    post.title = self._find_emojis(post.title)
                    post.text = self._find_emojis(post.text)
                    post.real_form.title = post.title
                    post.real_form.text = post.text
                    sub_list.append(post)
                    #save('retrieve emojis - AFTER', str(post))
                    
                else:
                    post.title = post.real_form.title
                    post.text = post.real_form.text
                    sub_list.append(post)
                    #save('shortcut', post.id)
            l.append(sub_list)
        
        save("\nEMOJIS", "OVER\n")
        return l
        
        

    def fit(self, X, y=None):
        return self


class Token():

    def __init__(self, type):
        if type == "normal":
            self.token = Normal_Tokenizer()
        elif type == "yake":
            self.token = Yake_Tokenizer()
        elif type == "my":
            self.token = My_Tokenizer()

    def transform(self, lista):
        print("Token")
        save('PHASE', 'TOKENIZER')
        l = list()
        for sub in lista:
            sub_list = list()
            for post in sub:
                if post.title == "None":
                    post.title = ""
                if post.real_form.text == '':
                    #save('tokenizer - BEFORE', str(post))
                    post.title = self.token.process(post.title) if post.title != "None" or post.title != None else ""
                    post.text = self.token.process(post.text) if post.text != "None" else ""
                    
                    sub_list.append(post)
                    post.real_form.title = post.title
                    post.real_form.text = post.text
                    #save('tokenizer - AFTER', str(post))
                else:
                    post.title = post.real_form.title
                    post.text = post.real_form.text
                    sub_list.append(post)
                    #save('shortcut', str(post))

            l.append(sub_list)
        
        save("\nTOKENIZER", "OVER\n")
        return l


    def fit(self, X, y=None):
        
        return self



class Sentiment():

    '''
    def iterate(self, sub):
        n = 500
        if len(sub) >= n:
            return sub[0:n]
        else:
            return sub + ([Post()] * (n - len(sub)))
    '''
    
    def transform(self, lista):
        print("SENTIMENT DONE")
        save('PHASE', 'SENTIMENT')
        #print(lista[0])
        #print("here")
        #tt =  torch.tensor( [ sentiment_analyzer_scores(post) for post in lista[0] ] )
        #print(tt)
        #tt = F.pad(tt, (0, (500 - len(lista[0]))), 'constant', 0)
        #print(tt)
        #print(tt.size())
        #t = torch.stack([  torch.tensor( [ sentiment_analyzer_scores(post) for post in sub ] ) for sub in lista ]).to(dev)
        #print(t)
        #t = torch.stack([  F.pad( input = torch.tensor( [ sentiment_analyzer_scores(post) for post in sub ] ), pad = (0, (1000 - len(sub))), mode = 'constant', value = 0 ) for sub in lista ]).to(dev)
        #print("here2")9
        #print(torch.stack( [ torch.tensor([mean([ sentiment_analyzer_scores(post) for post in sub ])]) for sub in lista ]))
        return torch.stack( [ torch.tensor([mean([ sentiment_analyzer_scores(post) for post in sub ])]) for sub in lista ])
        #save('\nSENTIMENT', 'OVER\n')
        #t = t.cpu()
        #return t

        

    def fit(self, X, y=None):
        
        return self


'''
class Bert(BaseEstimator, TransformerMixin):

    def __init__(self, bert_tokenizer, bert_model,  max_length = 60, embedding_function : Optional[Callable[[torch.tensor], torch.tensor]] = None):
        self.tokenizer = bert_tokenizer
        self.model = bert_model
        self.max_length = max_length
        self.function = embedding_function

        if self.function is None:
            self.function = lambda x: x[0][:, 0, :].squeeze().to(dev)

    

    def _tokenize(self, text : str):
        #string = self._filter_string(text)

        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=self.max_length, truncation = True)["input_ids"]
        #print(tokenized_text)
        # Create an attention mask telling BERT to use all words
        attention_mask = [1] * len(tokenized_text)
        # bert takes in a batch so we need to unsqueeze the rows
        #print((torch.tensor(tokenized_text).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0),))
        return (torch.tensor(tokenized_text).unsqueeze(0).to(dev), torch.tensor(attention_mask).unsqueeze(0).to(dev),)
        


    def _tokenize_and_predict(self, post: str) -> torch.tensor:
        if post.real_form.embeddings != None:
            #print("shortcut")
            #save("shortcut", str(post.id))
            return post.real_form.embeddings.to(dev)

        self.model.to(dev)
        s = self.function(self.model(self._tokenize(str(post))[0], self._tokenize(str(post))[1]))
        post.real_form.embeddings = s.cpu()
        #save("calc embeddings", str(s))
        return s
        #emb = self.function(self.model(t, m))
        #torch.cuda.empty_cache()
        #del t, m
        #gc.collect()
        #torch.cuda.empty_cache()

        #post.real_form.embeddings = emb.cpu()
        
        #return emb
    

    def _each_sub(self, sub ):
        with torch.no_grad():
            if sub == []:
                return torch.zeros([1, 768], dtype=torch.int32).to(dev)
            else:

                return torch.sum(torch.stack([ self._tokenize_and_predict(post) for post in sub ]).to(dev), 0).to(dev) / len(sub)
            
            

    def transform(self, lista):
        print("BERT")
        #save("PHASE", "BERT")
        #check_cuda()
        #device = cuda.get_current_device()
        #device.reset()
        #check_cuda()
        #torch.cuda.ipc_collect()
        #torch.cuda.empty_cache()
        #print(torch.cuda.memory_summary())
        with torch.no_grad():
            
            
            return torch.stack([ self._each_sub(sub) for sub in lista ]).to(dev).cpu()

            #return torch.stack([ self._each_sub(sub) for sub in lista ]).cpu()
            #tensor = tensor.cpu()
            #return tensor


    
    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self

'''

class BigBird():

    def __init__(self, mod):
        self.model = mod

    def __each_subject(self, sub):
        self.model.to(dev)
        return torch.tensor(self.model.encode(' '.join(map(lambda x : str(x), sub))))
            

    def transform(self, lista):
        
        #lista = [ ' '.join(map(lambda x : str(x), subject)) for subject in lista ]
        return torch.stack([ self.__each_subject(sub) for sub in lista]).cpu()
    
    
    def fit(self, X, y=None):
        """No fitting necessary so we just return ourselves"""
        return self


if __name__ == "__main__":
    
    string = "Eu sou o Lucas"
    lista = [ ['Eu sou o Lucas'] ]
    
    """ set the tokenizer and model parameters """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    # create the bert
    bert_transformer = Bert(tokenizer, bert_model)

    embeddings = bert_transformer.transform(lista)

    print(embeddings)
    





    


    