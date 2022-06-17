import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
text = open('read.txt', encoding='utf-8').read()
lower_case = text.lower() #coverting the text to lowercase
clean_text = lower_case.translate(str.maketrans('','',string.punctuation)) #removing the punctuations

tokenized_words = word_tokenize(clean_text,"english") #tokenization

final_words = []
for word in tokenized_words :
    if word not in stopwords.words('english') :
        final_words.append(word)


emotion_list = []
with open('emotions.txt') as file :
    for line in file :
        clear_line = line.replace('\n','').replace("'",'').replace(',','').strip()
        word,emotion = clear_line.split(':')
        if word in final_words :
            emotion_list.append(emotion)


w = Counter(emotion_list)


def sentiment_analyze(senti_text) :
    score = SentimentIntensityAnalyzer().polarity_scores(senti_text)
    neg = score['neg']
    pos = score['pos']
    print(score)
    if pos > neg :
        print("Positive Sentiment")
    elif pos < neg :
        print("Negative Sentiment")
    else :
        print("Neutral Sentiment")

sentiment_analyze(clean_text)

fig , ax1 = plt.subplots()
ax1.bar(w.keys(),w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()







