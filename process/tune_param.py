import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../pre_process')

from detect_sentiment import sentiment_analyzer_scores
import dill as pickle

from sklearn.ensemble import AdaBoostClassifier
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from corpus_reader import CorpusReader
from transformers import BertTokenizer, BertModel
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from bert import  Sentiment, Token, Emojis, BigBird
from sklearn.calibration import CalibratedClassifierCV
import os
from sklearn.feature_extraction.text import CountVectorizer
from timeit import default_timer as timer
import torch
import concurrent.futures
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

def train6():

    with open("log.txt", 'w') as f:
        pass
    #path1 = os.path.join( '..', '..',  'dataset', 'eRISK2020_T1_training_data', 'train') 
    #path1 = os.path.join( '..', 'data', 'erisk-2021-t2', 'td') 
    path1 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRISK2020_T1_training_data', 'eRISK2020_T1_training_data', 'eRISK2020_training_data')
    path2 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRisk2020_T1_test_data', 'eRisk2020_T1_test_data', 'T1')
    print("Creating Corpus Reader for training")
    corpus_reader_train = CorpusReader(path1)
    corpus_reader_train.load()
    print("Corpus Reader for training created")
    corpus_reader_test = CorpusReader(path2)
    corpus_reader_test.load()
    print("Corpus Reader for testing created")

    emo = Emojis()
    token = Token()

    """ set the tokenizer and model parameters """
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = SentenceTransformer('paraphrase-mpnet-base-v2')
    #device = torch.device("cuda")


    #bert_model.to(device)
    
    # create the bert
    bert_transformer = BigBird(bert_model)


    sentiment = Sentiment()

    
    """ training the model """
    print("Initializing Training")
    #n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
    parameters = { 'classifier__n_estimators':[50, 100, 500, 1000], 'classifier__learning_rate' : [ 0.001, 0.01, 0.1, 1.0], 'classifier__max_depth' : [1, 3, 5, 10]}
    classifier = GradientBoostingClassifier()
    


    
    
    model = Pipeline(
    [
        ('emojis', emo),
        #('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            ("sentiment", sentiment),
        ])),
        
    
        ("classifier", classifier),
    ]
    )
    
    clf = GridSearchCV(model,  parameters)
    
 
    
    batch_size = 40

    num_users = len(corpus_reader_train.subjects)
    #print(num_users)
    for j in range(50, 2000, 50):
        count = 0
        all_texts = list()
        all_gt = list()
        for i in range(0, num_users, batch_size):
            #print(i)
            
            all_texts.append([ subject.posts[0:j]  for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
            all_gt.append([ subject.gt for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
            count += 1

        print(all_gt[0])
        for i in range(len(all_texts)):
            clf.fit(all_texts[i], all_gt[i])

    
    num_users = len(corpus_reader_test.subjects)
    
    #print(num_users)
    for j in range(50, 2000, 50):
        all_texts = list()
        all_gt = list()
        count = 0
        for i in range(0, num_users, batch_size):
            print(i)
            all_texts.append([ subject.posts[0:j]  for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
            all_gt.append([ subject.gt for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
            count += 1

        
        for i in range(len(all_texts)):
            clf.fit(all_texts[i], all_gt[i])
    

    
    print("End of training")
    return clf
    

if __name__ == '__main__':
    model = train6()

    print(model.best_params_)
    with open("param.txt", 'a') as f:
        f.write(str(model.best_params_))
