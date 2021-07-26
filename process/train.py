
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



def run(args):

    model = args['model']
    data = args['data']
    labels = args['labels']


    model.fit(data, labels)


def train():

    with open("log.txt", 'w') as f:
        pass
    path1 = os.path.join( '..', '..',  'dataset', 'eRISK2020_T1_training_data', 'train') 
    #path1 = os.path.join( '..', 'data', 'erisk-2021-t2', 'td') 
    #path1 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRISK2020_T1_training_data', 'eRISK2020_T1_training_data', 'eRISK2020_training_data')
    #path2 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRisk2020_T1_test_data', 'eRisk2020_T1_test_data', 'T1')
    print("Creating Corpus Reader for training")
    corpus_reader_train = CorpusReader(path1)
    corpus_reader_train.load()
    print("Corpus Reader for training created")
    #corpus_reader_test = CorpusReader(path2)
    #corpus_reader_test.load()
    print("Corpus Reader for testing created")

    emo = Emojis()
    token = Token()

    """ set the tokenizer and model parameters """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    #device = torch.device("cuda")


    #bert_model.to(device)
    
    # create the bert
    bert_transformer = Bert(tokenizer, bert_model)


    sentiment = Sentiment()

    
    """ training the model """
    print("Initializing Training")
    classifier = svm.LinearSVC(C=1.0, class_weight="balanced", dual = False)
    clf = CalibratedClassifierCV(classifier)


    
    
    model = Pipeline(
    [
        ('emojis', emo),
        ('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            #("vectorizer", bert_transformer),
            ("sentiment", sentiment),
        ])),
        
    
        ("classifier", clf),
    ]
    )
    
    
    
    '''
    model = Pipeline(
    [
        
        ('tokenizer', token), 
        ("vectorizer", bert_transformer),
    
        ("classifier", clf),
    ]
    )
    '''
    
    batch_size = 85

    num_users = len(corpus_reader_train.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})

    '''
    num_users = len(corpus_reader_test.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})
    '''

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open('test1_emojis_token_sentiment.sav', 'wb')
    pickle.dump(model, dbfile)
    return model

def train2():

    with open("log.txt", 'w') as f:
        pass
    path1 = os.path.join( '..', '..',  'dataset', 'eRISK2020_T1_training_data', 'train') 
    #path1 = os.path.join( '..', 'data', 'erisk-2021-t2', 'td') 
    #path1 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRISK2020_T1_training_data', 'eRISK2020_T1_training_data', 'eRISK2020_training_data')
    #path2 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRisk2020_T1_test_data', 'eRisk2020_T1_test_data', 'T1')
    print("Creating Corpus Reader for training")
    corpus_reader_train = CorpusReader(path1)
    corpus_reader_train.load()
    print("Corpus Reader for training created")
    #corpus_reader_test = CorpusReader(path2)
    #corpus_reader_test.load()
    print("Corpus Reader for testing created")

    emo = Emojis()
    token = Token()

    """ set the tokenizer and model parameters """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    #device = torch.device("cuda")


    #bert_model.to(device)
    
    # create the bert
    bert_transformer = Bert(tokenizer, bert_model)


    sentiment = Sentiment()

    
    """ training the model """
    print("Initializing Training")
    classifier = svm.LinearSVC(C=1.0, class_weight="balanced", dual = False)
    clf = CalibratedClassifierCV(classifier)


    
    
    model = Pipeline(
    [
        ('emojis', emo),
        #('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            #("vectorizer", bert_transformer),
            ("sentiment", sentiment),
        ])),
        
    
        ("classifier", clf),
    ]
    )
    
    
    
    '''
    model = Pipeline(
    [
        
        ('tokenizer', token), 
        ("vectorizer", bert_transformer),
    
        ("classifier", clf),
    ]
    )
    '''
    
    batch_size = 85

    num_users = len(corpus_reader_train.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})

    '''
    num_users = len(corpus_reader_test.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})
    '''

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open('test1_emojis_sentiment.sav', 'wb')
    pickle.dump(model, dbfile)
    return model

def train3():

    with open("log.txt", 'w') as f:
        pass
    path1 = os.path.join( '..', '..',  'dataset', 'eRISK2020_T1_training_data', 'train') 
    #path1 = os.path.join( '..', 'data', 'erisk-2021-t2', 'td') 
    #path1 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRISK2020_T1_training_data', 'eRISK2020_T1_training_data', 'eRISK2020_training_data')
    #path2 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRisk2020_T1_test_data', 'eRisk2020_T1_test_data', 'T1')
    print("Creating Corpus Reader for training")
    corpus_reader_train = CorpusReader(path1)
    corpus_reader_train.load()
    print("Corpus Reader for training created")
    #corpus_reader_test = CorpusReader(path2)
    #corpus_reader_test.load()
    print("Corpus Reader for testing created")

    emo = Emojis()
    token = Token()

    """ set the tokenizer and model parameters """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    #device = torch.device("cuda")


    #bert_model.to(device)
    
    # create the bert
    bert_transformer = Bert(tokenizer, bert_model)


    sentiment = Sentiment()

    
    """ training the model """
    print("Initializing Training")
    classifier = svm.LinearSVC(C=1.0, class_weight="balanced", dual = False)
    clf = CalibratedClassifierCV(classifier)


    
    
    model = Pipeline(
    [
        ('emojis', emo),
        #('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            ("sentiment", sentiment),
        ])),
        
    
        ("classifier", clf),
    ]
    )
    
    
    
    '''
    model = Pipeline(
    [
        
        ('tokenizer', token), 
        ("vectorizer", bert_transformer),
    
        ("classifier", clf),
    ]
    )
    '''
    
    batch_size = 85

    num_users = len(corpus_reader_train.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})

    '''
    num_users = len(corpus_reader_test.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})
    '''

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open('test1_bert_emojis_sentiment.sav', 'wb')
    pickle.dump(model, dbfile)
    return model

def train4():

    with open("log.txt", 'w') as f:
        pass
    path1 = os.path.join( '..', '..',  'dataset', 'eRISK2020_T1_training_data', 'train') 
    #path1 = os.path.join( '..', 'data', 'erisk-2021-t2', 'td') 
    #path1 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRISK2020_T1_training_data', 'eRISK2020_T1_training_data', 'eRISK2020_training_data')
    #path2 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRisk2020_T1_test_data', 'eRisk2020_T1_test_data', 'T1')
    print("Creating Corpus Reader for training")
    corpus_reader_train = CorpusReader(path1)
    corpus_reader_train.load()
    print("Corpus Reader for training created")
    #corpus_reader_test = CorpusReader(path2)
    #corpus_reader_test.load()
    print("Corpus Reader for testing created")

    emo = Emojis()
    token = Token()

    """ set the tokenizer and model parameters """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    #device = torch.device("cuda")


    #bert_model.to(device)
    
    # create the bert
    bert_transformer = Bert(tokenizer, bert_model)


    sentiment = Sentiment()

    
    """ training the model """
    print("Initializing Training")
    classifier = svm.LinearSVC(C=1.0, class_weight="balanced", dual = False)
    clf = CalibratedClassifierCV(classifier)


    
    
    model = Pipeline(
    [
        ('emojis', emo),
        ('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            ("sentiment", sentiment),
        ])),
        
    
        ("classifier", clf),
    ]
    )
    
    
    
    '''
    model = Pipeline(
    [
        
        ('tokenizer', token), 
        ("vectorizer", bert_transformer),
    
        ("classifier", clf),
    ]
    )
    '''
    
    batch_size = 85

    num_users = len(corpus_reader_train.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})

    '''
    num_users = len(corpus_reader_test.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})
    '''

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open('test1_bert_emojis_token_sentiment.sav', 'wb')
    pickle.dump(model, dbfile)
    return model

def train5():

    with open("log.txt", 'w') as f:
        pass
    path1 = os.path.join( '..', '..',  'dataset', 'eRISK2020_T1_training_data', 'train') 
    #path1 = os.path.join( '..', 'data', 'erisk-2021-t2', 'td') 
    #path1 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRISK2020_T1_training_data', 'eRISK2020_T1_training_data', 'eRISK2020_training_data')
    #path2 = os.path.join('..', 'data', 'erisk-2021-t2', 'training', 'eRisk2020_T1_test_data', 'eRisk2020_T1_test_data', 'T1')
    print("Creating Corpus Reader for training")
    corpus_reader_train = CorpusReader(path1)
    corpus_reader_train.load()
    print("Corpus Reader for training created")
    #corpus_reader_test = CorpusReader(path2)
    #corpus_reader_test.load()
    print("Corpus Reader for testing created")

    emo = Emojis()
    token = Token()

    """ set the tokenizer and model parameters """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    #device = torch.device("cuda")


    #bert_model.to(device)
    
    # create the bert
    bert_transformer = Bert(tokenizer, bert_model)


    sentiment = Sentiment()

    
    """ training the model """
    print("Initializing Training")
    classifier = svm.LinearSVC(C=1.0, class_weight="balanced", max_iter=30000 )
    clf = CalibratedClassifierCV(classifier)


    
    
    model = Pipeline(
    [
        ('emojis', emo),
        ('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            #("sentiment", sentiment),
        ])),
        
    
        ("classifier", clf),
    ]
    )
    
    
    
    '''
    model = Pipeline(
    [
        
        ('tokenizer', token), 
        ("vectorizer", bert_transformer),
    
        ("classifier", clf),
    ]
    )
    '''
    
    batch_size = 85

    num_users = len(corpus_reader_train.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        #print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_train.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    print(all_gt[0])
    for i in range(len(all_texts)):
        model.fit(all_texts[i], all_gt[i])

    '''
    num_users = len(corpus_reader_test.subjects)
    count = 0
    all_texts = list()
    all_gt = list()
    #print(num_users)
    for i in range(0, num_users, batch_size):
        print(i)
        all_texts.append([ subject.posts[0:1000]  for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))]  ])
        all_gt.append([ subject.gt for subject in corpus_reader_test.subjects[(batch_size * count) : (batch_size * (count + 1))] ])
        count += 1

    
    for i in range(len(all_texts)):
        run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})
    '''

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open('test1_bert_emojis_token.sav', 'wb')
    pickle.dump(model, dbfile)
    return model

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
    #classifier = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    #clf = CalibratedClassifierCV(classifier)
    #classifier = svm.SVC(C = 1, gamma = 'scale', kernel = 'linear', probability = True)
    #classifier = AdaBoostClassifier(learning_rate = 0.01, n_estimators = 100)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    
    
    model = Pipeline(
    [
        ('emojis', emo),
        #('tokenizer', token), 
        ('union', FeatureUnion(transformer_list = [
            ("vectorizer", bert_transformer),
            ("sentiment", sentiment),
        ])),
        
    
        ("classifier", clf),
    ]
    )
    
    
    
    '''
    model = Pipeline(
    [
        
        ('tokenizer', token), 
        ("vectorizer", bert_transformer),
    
        ("classifier", clf),
    ]
    )
    '''
    
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
            run({'model' : model,'data' :  all_texts[i], 'labels' : all_gt[i]})
    

    
    print("End of training")
    
    # Its important to use binary mode
    dbfile = open('final_bigbird_emojis_sentiment_gbt.sav', 'wb')
    pickle.dump(model, dbfile)
    return model




#model = train()

if __name__ == "__main__":
    #train()
    #train2()
    #train3()
    #train4()
    #train5()
    train6()

