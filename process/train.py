
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../pre_process')

from detect_sentiment import sentiment_analyzer_scores
import dill as pickle

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier




class SVM(svm.LinearSVC):


    def fit(self, X, y, sample_weight=None):
        print(X[0])
        super().fit(X, y, sample_weight)




def train_model1(classifier):

    
    
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
    token = Token("normal")

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
    #classifier = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    #clf = CalibratedClassifierCV(classifier)
    #classifier = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    #classifier = AdaBoostClassifier(learning_rate = 0.01, n_estimators = 100)

    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    
    
    model = Pipeline(
    [
        ('emojis', emo),
        ('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            #("sentiment", sentiment),
        ])),
        
    
        ("classifier", classifier),
    ]
    )
    
    
    
    
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
            model.fit(all_texts[i], all_gt[i])

    
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
            model.fit(all_texts[i], all_gt[i])
    

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open(f'model1_{classifier.__class__.__name__}.sav', 'wb')
    pickle.dump(model, dbfile)
    return model

def train_model2(classifier):

    
    
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
    token = Token("yake")

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
    #classifier = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    #clf = CalibratedClassifierCV(classifier)
    #classifier = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    #classifier = AdaBoostClassifier(learning_rate = 0.01, n_estimators = 100)

    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    
    
    model = Pipeline(
    [
        ('emojis', emo),
        ('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            #("sentiment", sentiment),
        ])),
        
    
        ("classifier", classifier),
    ]
    )
    
    
    
    
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
            model.fit(all_texts[i], all_gt[i])

    
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
            model.fit(all_texts[i], all_gt[i])
    

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open(f'model2_{classifier.__class__.__name__}.sav', 'wb')
    pickle.dump(model, dbfile)
    return model

def train_model3(classifier):

    
    
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
    token = Token("normal")

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
    #classifier = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    #clf = CalibratedClassifierCV(classifier)
    #classifier = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    #classifier = AdaBoostClassifier(learning_rate = 0.01, n_estimators = 100)

    #clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    
    
    model = Pipeline(
    [
        ('emojis', emo),
        ('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            ("sentiment", sentiment),
        ])),
        
    
        ("classifier", classifier),
    ]
    )
    
    
    
    
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
            model.fit(all_texts[i], all_gt[i])

    
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
            model.fit(all_texts[i], all_gt[i])
    

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open(f'model3_{classifier.__class__.__name__}.sav', 'wb')
    pickle.dump(model, dbfile)
    return model



def train_model4(classifier):

    
    
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
    token = Token("yake")

    """ set the tokenizer and model parameters """
    
    bert_model = SentenceTransformer('paraphrase-mpnet-base-v2')
    
    # create the bert
    bert_transformer = BigBird(bert_model)

    sentiment = Sentiment()

    
    """ training the model """
    print("Initializing Training")
        
    
    model = Pipeline(
    [
        ('emojis', emo),
        ('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            ("sentiment", sentiment),
        ])),
        
    
        ("classifier", classifier),
    ]
    )
    
    batch_size = 40

    num_users = len(corpus_reader_train.subjects)
    
    count = 0
    all_texts = list()
    all_gt = list()
    for i in range(0, num_users, batch_size):
        
        all_texts.append([ subject.posts  for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        model.fit(all_texts[i], all_gt[i])

    
    '''
    num_users = len(corpus_reader_test.subjects)
    
    
    all_texts = list()
    all_gt = list()
    count = 0
    for i in range(0, num_users, batch_size):
        
        all_texts.append([ subject.posts  for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        model.fit(all_texts[i], all_gt[i])
    
    '''
    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open(f'model4_{classifier.__class__.__name__}.sav', 'wb')
    pickle.dump(model, dbfile)
    return model




#model = train()

if __name__ == "__main__":

    classifier1 = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    classifier2 = AdaBoostClassifier(learning_rate = 0.01, n_estimators = 100) 


    args = sys.argv

    model = args[1]
    classifier = args[2]

    if model == 'model1':
        if classifier == 'svm':
            train_model1(classifier1)
        elif classifier == 'adaboost':
            train_model1(classifier2)
    elif model == 'model2':
        if classifier == 'svm':
            train_model2(classifier1)
        elif classifier == 'adaboost':
            train_model2(classifier2)
    elif model == 'model3':
        if classifier == 'svm':
            train_model3(classifier1)
        elif classifier == 'adaboost':
            train_model3(classifier2)
    elif model == 'model4':
        if classifier == 'svm':
            train_model4(classifier1)
        elif classifier == 'adaboost':
            train_model4(classifier2)
    

