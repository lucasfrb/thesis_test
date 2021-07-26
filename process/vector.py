

from sklearn.linear_model import LogisticRegression
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../pre_process')



from corpus_reader import CorpusReader
from transformers import BertTokenizer, BertModel
from sklearn import svm
from sklearn.pipeline import Pipeline
from bert import Bert, Apply_PCA, Normalize, Vectorizer
from sklearn.calibration import CalibratedClassifierCV
import os
from sklearn.feature_extraction.text import CountVectorizer
from matrix import Matrix
import measures




def read_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for json_obj in f:
            data.append(json.loads(json_obj))
    return data


def get_all_tweets(data):
    tweets = []
    for user in data:
        full_tweets = []
        for tweet in user["text"]:
            full_tweets.extend(tweet)
        tweets.append(" ".join(full_tweets))
    return tweets


def write_results(output_path, results):
    with open(os.path.join(output_path, "results.tsv"), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(results)

def run_simulation(args):
    matrix = args['matrix']
    vec = args['vec']
    clas = args['class']
   
    
    for i in range(5, 300):
        print("POST {}".format(i))
        posts = dict()
        for entry in matrix.results:
            #print(entry.blocked)
            #print(len(entry.subject.posts))
            if not entry.blocked and len(entry.subject.posts) > i:
                strings = [str(t) for t in entry.subject.posts[:(i + 1)]]
                posts[entry] = strings
        
        #print(list(posts.values()))
        if list(posts.values()) != []: 
            subs = list(posts.keys())    
            list_posts = list(posts.values())
            probs = clas.predict_proba(vec.transform([ ''.join(map(lambda x : str(x), sub)) for sub in list_posts]))
            

            for i in range(len(list_posts)):
                decision = '0' if probs[i][0] > 0.5 else '1'
                for e, l in posts.items():  
                    if l == list_posts[i]:
                        matrix.write_test_result(e.subject.id, decision)
        
        else:
            break
       
    

        
    return matrix

def main():
    path = os.path.join('..', '..', 'dataset', 'eRISK2020_T1_training_data', 'td')

    print("Creating Corpus Reader for training")
    corpus_reader_train = CorpusReader(path)
    corpus_reader_train.load()
    print("Corpus Reader for training created")

    path = os.path.join('..', '..', 'dataset', 'T1_test_data', 'td') 
    gt_name = 'T1_erisk_golden_truth.txt'
    corpus_reader_test = CorpusReader(path, gt_name)
    corpus_reader_test.load()


    all_texts = [ ''.join(map(lambda x : str(x), subject.posts)) for subject in corpus_reader_train.subjects]
    all_gt = [ subject.gt for subject in corpus_reader_train.subjects ]

    count_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w+', ngram_range=(1, 2))
    bow = dict()
    bow["train"] = (count_vectorizer.fit_transform(all_texts), all_gt)

    lr_classifier = LogisticRegression(solver='liblinear')
    lr_classifier.fit(*bow["train"])

    
    matrix = Matrix(len(corpus_reader_test.subjects), corpus_reader_test.subjects)
    args = {'matrix' : matrix, 'vec' : count_vectorizer, 'class' : lr_classifier}
    
    matrix = run_simulation(args)
    
    print(matrix)

     # analyze results
    precision = measures.calc_precision(corpus_reader_test.subjects, matrix)
    recall = measures.calc_recall(corpus_reader_test.subjects, matrix)
    f1 = measures.calc_f1(precision, recall)
    ERDE = measures.calc_ERDE(corpus_reader_test.subjects, matrix)

    


if __name__ == '__main__':

    main()