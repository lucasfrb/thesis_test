import xml.etree.ElementTree as ET

class Post:

    count = 0

    def __init__(self):
        self.id = Post.count
        Post.count += 1
        self.date = ""
        self.title = ""
        self.text = ""
        self.real_form = RF()


        
    
    def __str__(self):
        if self.title == "None" or self.title is None or self.title == "":
            return self.text
        elif self.text == "None" or self.text is None or self.text == "":
            return self.title

        return self.title + " - " + self.text

    def __repr__(self):
        return self.__str__()



class RF:

    def __init__(self):
        self.title = ""
        self.text = ""
        self.vader = -5
        self.embeddings = None


    def __str__(self):
        return '{} - {}'.format(self.text, self.vader)

    def __repr__(self):
        return self.__str__()



class Date:

    def __init__(self, string):
        self.time = string.split(' ')[1]
        self.date = string.split(' ')[0]

    def __str__(self):
        return '{} - {}'.format(self.date, self.time)

    def __repr__(self):
        return self.__str__()

if __name__ == '__main__':

    p = Post()
    p2 = Post()

    print(p.id)
    print(p2.id)
