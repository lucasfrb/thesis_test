 

class Matrix:

    def __init__(self, number_subjects, subjects):
        self.size = number_subjects
        
        self.results = dict()
        for s in subjects:
            entry = Entry(s)
            self.results[s.id] = entry

    
    
    def write_test_result(self, id, result):
        # 1 - self harm, 0 - control
        assert(len(result) == 1), 'prediction is returning not just one result'
        result = result[0].strip()
        assert(result == '0' or result == '1')
        self.results[id].add(result)

    

    def __str__(self):
        s = ""
        for entry in self.results:
            s += str(entry) + "\n"
        return s

    def serialize(self, arg):
        with open(f'matrix{arg}.txt', 'w') as f:
            for result in self.results:
                key = result
                value = self.results[key]  # Entry
                f.write(f"{key} - ")

                zeros = value.list.count('0')
                f.write(f'{zeros} - ')
                if value.analyze_subject() == '1':
                    f.write('self-harm\n')
                else:
                    f.write('control\n')
            
                



class Entry:

    def __init__(self, s):
        self.subject = s
        self.gt = self.subject.gt
        self.list = list()
        self.blocked = False

    def __eq__(self, another):
        return hasattr(another, 'subject') and hasattr(another, 'list') and hasattr(another, 'blocked') and self.subject == another.subject
   
    def __hash__(self):
        return hash(self.subject)

    def add(self, result):
        if not self.blocked:
            self.list.append(result)
        
        if result == '1':
            self.blocked = True
    
    def analyze_subject(self):
        if self.list == []:
            return '0'
        if '1' in self.list:
            return '1'
        elif all(v == '0' for v in self.list):
            return '0'
        
        raise Exception('List with wrong values')

    def __str__(self):
        return str(self.subject.id) + ' : ' + str(self.list)