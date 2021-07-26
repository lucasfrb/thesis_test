import os
import xml.etree.cElementTree as ET

from subject import Subject
import concurrent.futures
from count import Count



class CorpusReader:

    def __init__(self, path, gt_file = 'golden_truth.txt', size = None):
        self.path = path
        self.subjects = list()
        self.gt_file = gt_file
        self.size = size
        #self.count_words = Count()


    def load(self):
        data_path = os.path.join(self.path, 'DATA')         
        increment = 0
        files = os.listdir(data_path)
        for i,f in enumerate(files):
            if i == self.size:
                print("aqui")
                break
            print(" Percentage : {:.2f}".format((increment / len(files)) * 100) if self.size is None else (increment / self.size))
            if f.endswith(".xml") and f.startswith("subject"):
                sub = Subject(ET.parse(os.path.join(data_path, f)).getroot())
                self.subjects.append(sub)
            increment += 1 
            
        self._load_gt()

    def _load_gt(self):
        data_path = os.path.join(self.path, self.gt_file)
        with open(data_path) as file:
            for line in file:
                sub, gt = line.strip().split(" ")
                s = self._get_subject(sub)
                if s is not None:
                    s.gt = gt

    def _get_subject(self, sub):
        for s in self.subjects:
            if s.id == sub:
                return s
        return None


if __name__ == "__main__":

    path1 = os.path.join('..', 'data', 'erisk-2021-t2', 'td')
    corpus_reader_train = CorpusReader(path1, size = 2)
    corpus_reader_train.load()
    
    all_texts = [ list(map(lambda x : str(x), subject.posts)) for subject in corpus_reader_train.subjects]
    all_gt = [ subject.gt for subject in corpus_reader_train.subjects ]

    for sub in corpus_reader_train.subjects:
        print(f"{sub} - {sub.gt}")




