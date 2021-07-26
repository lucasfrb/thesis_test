import xml.etree.cElementTree as ET
from post import Post, Date
from tokenizer import Tokenizer
from emoticons import EMOJIS
from detect_sentiment import sentiment_analyzer_scores
import re

def spplit(string):
    return re.split('[,;\.?!\\\\* ]+', string)

class Subject:

    def __init__(self, r):
        self.root = r
        self.posts = list()
        self.tokenizer = Tokenizer()
        self.gt = "-1"
        self._parse()
             

    def _parse(self):
        for element in self.root.iter():
            if element.tag == 'ID':
               self.id = element.text
            elif element.tag == 'WRITING':
                post = Post()
            elif element.tag == 'TITLE':                
                post.title = element.text

                
            elif element.tag == 'DATE':
                d = Date(element.text)
                post.date = d
            elif element.tag == 'TEXT':

                #self._find_emojis2(element.text)
                post.text = element.text

                
            elif element.tag == 'INFO':
                post.info = element.text

                
                self.posts.append(post)
        

    def __eq__(self, another):
        return self.id == another.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        s = self.id 
        
        return s

    def __str__(self):
        return self.__repr__()

# empath , TextBlob, liwc, adaboost
