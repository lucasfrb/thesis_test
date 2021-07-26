import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../pre_process')
from corpus_reader import CorpusReader

import measures
from matrix import Matrix
from bert import  Sentiment, Emojis, Token
import os
import dill as pickle
import torch
from train import train6


def run_simulation(args):
    matrix = args['matrix']
    model = args['model']
    prob_threshold = args['prob_threshold']
    min_texts_trigger_alert = args['min_texts_trigger_alert']
    max_texts = args['max_texts']
    
    
    for i in range(min_texts_trigger_alert, max_texts, 3):
        print("POST {}".format(i))
        posts = dict()
        for id in matrix.results:
            entry = matrix.results[id]
            #print(entry.blocked)
            #print(len(entry.subject.posts))
            if not entry.blocked and len(entry.subject.posts) > i:
                strings = [post for post in entry.subject.posts[:(i + 1)]]
                posts[entry] = strings
        
        #print(list(posts.values()))
        if list(posts.values()) != []: 
            subs = list(posts.keys())    
            list_posts = list(posts.values())
            predictions = model.predict_proba(list_posts)
            with open("sub_results.txt", 'a') as f:
                for i,pred in enumerate(predictions):
                    f.write(f"{i} : {pred[0]} - {pred[1]}\n")
            #print(list(predictions[0 : 700]))
            #print(" Num 1's - {}".format(str(list(predictions).count('1'))))
            for i in range(len(predictions[0:700])):
                print("{} - {} - {}".format(subs[i].subject, predictions[i], subs[i].subject.gt))
            for i in range(len(list_posts)):
                #decision = predictions[i]
                decision = '0' if predictions[i][0] > prob_threshold else '1'
                for e, l in posts.items():  
                    if l == list_posts[i]:
                        matrix.write_test_result(e.subject.id, decision)
        
        else:
            print("cortar")
            break

    return matrix

def write_results(filename, header, results):
    with open(filename, 'a') as f:
        f.write("*** SIMULATION ***\n")
        for key in header:
            f.write("{} - {}, ".format(key, header[key]))
        f.write("\n")
        for key in results:
            f.write("{} : {}\n".format(key, results[key]))


if __name__ == '__main__':

    
    MODEL1_NAME = 'final_bigbird_emojis_sentiment_gbt.sav'
    MODEL2_NAME = 'test1_emojis_token_sentiment.sav'
    MODEL3_NAME = 'test1_bert_emojis_token.sav'
    MODEL4_NAME = 'test1_bigbird_emojis.sav'
    MODEL5_NAME = 'test1_bigbird_emojis_sentiment.sav'
    MODEL6_NAME = 'test1_bigbird_emojis_sentiment_svm.sav'
    MODEL7_NAME = 'test1_yake2_bigbird_emojis_sentiment_svm.sav'


    # load the model from disk
    mod1 = pickle.load(open(MODEL1_NAME, 'rb'))
    #mod2 = pickle.load(open(MODEL2_NAME, 'rb'))
    #mod3 = pickle.load(open(MODEL3_NAME, 'rb'))
    #mod4 = pickle.load(open(MODEL4_NAME, 'rb'))
    #mod5 = pickle.load(open(MODEL5_NAME, 'rb'))
    #mod6 = pickle.load(open(MODEL6_NAME, 'rb'))
    #mod7 = pickle.load(open(MODEL7_NAME, 'rb'))
    #device = torch.device("cuda")
    #no_vader.to(device)
    
    

    path = os.path.join( '..', 'data', 'erisk-2021-t2') 
    #path = os.path.join( '..', '..',  'dataset', 'T1_test_data', 'test')

    gt_name = 'golden_truth.txt'

    corpus_reader_test = CorpusReader(path)
    corpus_reader_test.load()

    with open("file.txt", 'w') as f:
        for sub in corpus_reader_test.subjects:
            f.write("{} - {}\n".format(sub.id, sub.gt))

    filename = "RESULTS_TEST_more_model3_no_token_param.txt"

    #clean file
    with open(filename, 'w') as file:
        pass

    # find the greatest number of posts
    posts_max = max([ len(s.posts) for s in corpus_reader_test.subjects ])
    print(posts_max)

    #mod6 = train6()

    
    
    model = {mod1 : "mod1"}    
    probs = [  0.8 ]
    #probs = [0.52]
    mi = 20
    ma = 1000

    for m in model:
        for prob in probs:
            matrix = Matrix(len(corpus_reader_test.subjects), corpus_reader_test.subjects)
            args = {'matrix' : matrix, 'model' : m, 'min_texts_trigger_alert' : mi , 'max_texts' : ma, 'prob_threshold' : prob}

            matrix = run_simulation(args)
            #print(matrix)

            # analyze results
            precision = measures.calc_precision(corpus_reader_test.subjects, matrix)
            print(precision)
            recall = measures.calc_recall(corpus_reader_test.subjects, matrix)
            print(recall)
            f1 = measures.calc_f1(precision, recall)
            print(f1)
            ERDE5 = measures.calc_ERDE5(corpus_reader_test.subjects, matrix)
            print(ERDE5)
            ERDE50 = measures.calc_ERDE50(corpus_reader_test.subjects, matrix)
            print(ERDE50)
            Flatency = measures.calc_FLatency(corpus_reader_test.subjects, matrix, f1)
            print(Flatency)

            header = {'Model' : model[m], 'Minimun number of Texts to trigger an Alert' : mi, 'Maximum number of texts read by subject' : ma, 'probability' : prob}
            results = {'PRECISION' : precision, 'RECALL' : recall, 'F1' : f1, 'ERDE5' : ERDE5, 'ERDE50' : ERDE50, 'Flatency' : Flatency}
            write_results(filename, header, results)
            matrix.serialize('2')

    '''

    model = {mod7 : "mod7"}    
    mi = 10
    ma = 700

    for m in model:
        matrix = Matrix(len(corpus_reader_test.subjects), corpus_reader_test.subjects)
        args = {'matrix' : matrix, 'model' : m, 'min_texts_trigger_alert' : mi , 'max_texts' : ma}

        matrix = run_simulation(args)
        #print(matrix)

        # analyze results
        precision = measures.calc_precision(corpus_reader_test.subjects, matrix)
        print(precision)
        recall = measures.calc_recall(corpus_reader_test.subjects, matrix)
        print(recall)
        f1 = measures.calc_f1(precision, recall)
        print(f1)
        ERDE5 = measures.calc_ERDE5(corpus_reader_test.subjects, matrix)
        print(ERDE5)
        ERDE50 = measures.calc_ERDE50(corpus_reader_test.subjects, matrix)
        print(ERDE50)
        Flatency = measures.calc_FLatency(corpus_reader_test.subjects, matrix, f1)
        print(Flatency)

        header = {'Model' : model[m], 'Minimun number of Texts to trigger an Alert' : mi, 'Maximum number of texts read by subject' : ma}
        results = {'PRECISION' : precision, 'RECALL' : recall, 'F1' : f1, 'ERDE5' : ERDE5, 'ERDE50' : ERDE50, 'Flatency' : Flatency}
        write_results(filename, header, results)

        '''




