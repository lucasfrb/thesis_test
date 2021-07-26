from corpus_reader import CorpusReader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import re

def spplit2(string):
    return re.split('."^[,;\?%&()\\\\* ]+', string)

def spplit(string):
    return re.split('\W', string)

class Extractor():

    imp_words = list()


    def __init__(self, path, threshold):
        self.path = path
        self.threshold = threshold

    
    def run(self):
        corpus_reader = CorpusReader(self.path)
        corpus_reader.load()
        analyser = SentimentIntensityAnalyzer()

        num_subs = len(corpus_reader.subjects)
        for i, sub in enumerate(corpus_reader.subjects):
            print(f"Number os subjects left : {num_subs - i}")
            for post in sub.posts:
                score = analyser.polarity_scores(str(post))
                s = score['compound']
                if abs(s) > self.threshold:
                    string = spplit(str(post))
                    for j in range(3):
                        for i in range(len(string) - j):
                            score_word = analyser.polarity_scores(' '.join(string[i:(i + j)]))
                            word_compound = score_word['compound']
                            if abs(word_compound) > self.threshold:
                                if string[i] not in self.imp_words:
                                    self.imp_words.append(' '.join(string[i:(i + j)]))


if __name__ == '__main__':

    path = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRISK2020_T1_training_data', 'eRISK2020_T1_training_data', 'eRISK2020_training_data')
    e = Extractor(path, threshold = 0.2)
    e.run()

    with open("imp_words.txt", 'w') as f:
        for word in e.imp_words:
            word = str(word)
            f.write(f"{word} \n")








