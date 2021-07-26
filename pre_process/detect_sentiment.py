from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from post import Post

filename = 'log.txt'

def save(headline, value):
    with open(filename, 'a') as f:
        f.write(headline + '\n')
        f.write(value + "\n")


def sentiment_analyzer_scores(post):
    #print(post.id)
    if post.real_form.vader == -5:
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(str(post))
        s = score['compound']
        post.real_form.vader = s
        #save('sentiment analysis - POST', str(post))
        #save('sentiment analysis - SCORE', str(s))
    else:
        s = post.real_form.vader
        #save('shortcut - POST', str(post))
        #save('shortcut', str(post.id))
    return  s


if __name__ == "__main__":

    sentence = "Hi Andrew, nice to meet you joy"


    print(sentiment_analyzer_scores(sentence))