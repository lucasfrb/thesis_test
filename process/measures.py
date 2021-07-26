
from math import e
import statistics

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def calc_precision(subjects, matrix):
    den = len([ sub for sub in matrix.results.values() if sub.analyze_subject() == '1'])
    num = len(intersection([ sub.subject for sub in matrix.results.values() if sub.analyze_subject() == '1'], [ sub for sub in subjects if sub.gt == '1' ]))

    if den == 0:
        print("0 no denominador")
        return 0

    return round(num / den, 3)

def calc_recall(subjects, matrix):
    den = len([ sub for sub in subjects if sub.gt == '1' ])
    num = len(intersection([ sub.subject for sub in matrix.results.values() if sub.analyze_subject() == '1'], [ sub for sub in subjects if sub.gt == '1' ]))

    return round(num / den, 3)

def calc_f1(precision, recall):
    if (precision + recall) == 0:
        print("0 no denominador")
        return 0
        
    return round(2 * ((precision * recall) / (precision + recall)), 3)


def calc_ERDE5(subjects, matrix):
    FP = lambda x : round(len([sub for sub in subjects if sub.gt == '1']) / len(subjects), 3)
    FN = lambda x : 1
    TN = lambda x : 0
    TP = lambda x : 1 - (1 / (1 + e ** ((len(x.list) -1) - 5)))
    error = list()
    for id in matrix.results:
        entry = matrix.results[id]
        d = entry.analyze_subject()
        gt = entry.subject.gt
        if d == '1' and gt == '0':
            error.append(FP(entry))
        elif d == '0' and gt == '1':
            error.append(FN(entry))
        elif d == '1' and gt == '1':
            error.append(TP(entry))
        else:
            error.append(TN(entry))

    return round(statistics.mean(error), 3)

def calc_ERDE50(subjects, matrix):
    FP = lambda x : round(len([sub for sub in subjects if sub.gt == '1']) / len(subjects), 3)
    FN = lambda x : 1
    TN = lambda x : 0
    TP = lambda x : 1 - (1 / (1 + e ** ((len(x.list) -1) - 50)))
    error = list()
    for id in matrix.results:
        entry = matrix.results[id]
        d = entry.analyze_subject()
        gt = entry.subject.gt
        if d == '1' and gt == '0':
            error.append(FP(entry))
        elif d == '0' and gt == '1':
            error.append(FN(entry))
        elif d == '1' and gt == '1':
            error.append(TP(entry))
        else:
            error.append(TN(entry))

    return round(statistics.mean(error), 3)

def calc_FLatency(subjects, matrix, F):

    penalty = lambda x : -1 + (2 / (1 + e ** (-0.5 * (len(x.list) - 1))))

    error = list()
    for id in matrix.results:
        entry = matrix.results[id]
        d = entry.analyze_subject()
        gt = entry.subject.gt
        if d == '1' and gt == '1':
            error.append(penalty(entry))

    speed = (1 - statistics.median(error))

    return F * speed



