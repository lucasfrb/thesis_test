from typing import SupportsComplex
from numpy.lib.shape_base import split
import yake
import re
import os
from Levenshtein import distance as lev
# pickle

def analyze_token(t):
    if t == "":
        return False
    if not (ord(t[0]) >= 65 and ord(t[0]) <= 90 or ord(t[0]) >= 97 and ord(t[0]) <= 122):
        return False
    if "//" in t:
        return False
    
    return True


def spplit2(string):
    tokens = list()
    string = ""
    in_string = False
    analyzed = False
    for ch in string:
        if ch == '!':
            tokens.append(ch)
        
        if ord(ch) >= 65 and ord(ch) <= 90 or ord(ch) >= 97 and ord(ch) <= 122:
            in_string = True
            analyzed = False
        else:
            in_string = False


        if in_string:
            string += ch
        else:
            if not analyzed:
                if analyze_token(string):
                    tokens.append(string)
                    string = ""
                analyzed = True

    return tokens

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

        '''
        string = self.rule_lower_case(string)
        #tokens = spplit(string)
        #tokens = self.rule_rm_stopwords(tokens)
        #tokens = filter(analyze_token, tokens)
        #return ' '.join(tokens)
        kw_extractor = yake.KeywordExtractor(top = 5)
        tokens = [ token[0] for token in kw_extractor.extract_keywords(string) ]
        return ' '.join(tokens)
        '''

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
        tokens = self.rule_rm_stopwords(tokens)

        keys = list(self.imp_words.keys())
        keys.sort(reverse = True)

        #print(self.imp_words)

        new_string_list = list()
        for key in keys:
            tokens_k = self.combinations(tokens, key)
            values = self.imp_words[key]
            for token in tokens_k:
                for imp_token in values:
                    if lev(token, imp_token) < 2:
                        new_string_list.append(imp_token)
        
                    
        new_string = ' '.join(new_string_list)

        return new_string

        


        

        

def load_stopwords(filename = 'stopwords.txt'):
    stopwords = list()
    with open(filename) as file:
        for line in file:
            stopwords.append(line.strip().lower())
    
    return stopwords


if __name__ == "__main__":
    t = My_Tokenizer()
    print(t.process("I have just found a critical situation. #998 http://knsdkjs.com"))

