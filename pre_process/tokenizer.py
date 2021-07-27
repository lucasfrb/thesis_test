from typing import SupportsComplex
from numpy.lib.shape_base import split
import yake
import re
import os
#from Levenshtein import distance as lev
# pickle

def analyze_token(t):
    if t == "":
        return False
    if not (ord(t[0]) >= 65 and ord(t[0]) <= 90 or ord(t[0]) >= 97 and ord(t[0]) <= 122):
        return False
    if "//" in t:
        return False
    
    return True



def spplit(string):
    return re.split('[,;\?%&()\\\\* ]+', string)


class Tokenizer:

    def __init__(self):
        self.rules = [func for func in dir(Tokenizer) if callable(getattr(Tokenizer, func)) if func.startswith("rule")]

    def rule_lower_case(self, string):
        return string.lower()

    
    def rule_rm_stopwords(self, tokens):
        stopwords = load_stopwords()
        for token in tokens:
            if token in stopwords:
                tokens.remove(token)
        
        return tokens
    

    def process(self, string):
        if string == '' or string is None:
            return ''

       

class Normal_Tokenizer(Tokenizer):

    def process(self, string):
        super().process(string)

        string = self.rule_lower_case(string)
        tokens = spplit(string)
        tokens = self.rule_rm_stopwords(tokens)
        tokens = filter(analyze_token, tokens)
        return ' '.join(tokens)


class Yake_Tokenizer(Tokenizer):

    def process(self, string):
        super().process(string)

        kw_extractor = yake.KeywordExtractor(top = 5)
        tokens = [ token[0] for token in kw_extractor.extract_keywords(string) ]
        return ' '.join(tokens)

class My_Tokenizer(Tokenizer):

    imp_words = dict()

    def __init__(self):

        p = os.path.join("..", "pre_process", "imp_words.txt")
        with open(p, encoding="utf-8") as f:
            for line in f:
                word, *rest = line.strip().split()
                #print(f"{word} {rest}")
                key = len(rest) + 1
                if key not in self.imp_words:
                    self.imp_words[key] = [' '.join(list([word] + rest))]
                else:
                    self.imp_words[key].append(' '.join(list([word] + rest)))


    def combinations(self, tokens, k):
        lista = list()
        for i in range(len(tokens) - k):
            lista.append(' '.join(tokens[i:i+k]))
        return lista

    def process(self, string):
        super().process(string)
        string = self.rule_lower_case(string)
        tokens = spplit(string)
        

        keys = list(self.imp_words.keys())
        keys.sort(reverse = True)

        #print(self.imp_words)

        new_string_list = list()
        for key in keys:
            tokens_k = self.combinations(tokens, key)
            values = self.imp_words[key]
            for i,token in enumerate(tokens_k):
                for imp_token in values:
                    if token == imp_token and not token in new_string_list:
                        new_string_list.extend([tokens_k[i - 1], token, tokens_k[i + 1]])
        
        tokens = self.rule_rm_stopwords(new_string_list)     
        new_string = ' '.join(tokens)

        return new_string

        


        

        

def load_stopwords(filename = 'stopwords.txt'):
    stopwords = list()
    with open(filename) as file:
        for line in file:
            stopwords.append(line.strip().lower())
    
    return stopwords


if __name__ == "__main__":
    t = My_Tokenizer()
    print(t.process("it #8217;s not much but I think you are a hero after all that shit you went through u still have hope most people don #8217;t, I don #8217;t know you but this inspires me to do something with my life thank you"))

