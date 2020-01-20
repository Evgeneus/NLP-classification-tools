# from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import re

# punctuation marks to be removed
punctuation = set([',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=',
                   '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°',
                   '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â',
                   '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
                   '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è',
                   '¸', '¾', 'Ã', '⋅', '‘', '∞',  '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・',
                   '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', "``", "''"])

# english contractions
contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "can not", "'cause": "because",
                     "could've": "could have", "couldn't": "could not", "didn't": "did not",
                     "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                     "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                     "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                     "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                     "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                     "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                     "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                     "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                     "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                     "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                     "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                     "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                     "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                     "she'll've": "she will have", "she's": "she is", "should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                     "so's": "so as", "this's": "this is", "that'd": "that would",
                     "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                     "there'd've": "there would have", "there's": "there is", "here's": "here is",
                     "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                     "they'll've": "they will have", "they're": "they are", "they've": "they have",
                     "to've": "to have", "wasn't": "was not", "we'd": "we would",
                     "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                     "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                     "what'll've": "what will have", "what're": "what are", "what's": "what is",
                     "what've": "what have", "when's": "when is", "when've": "when have",
                     "where'd": "where did", "where's": "where is", "where've": "where have",
                     "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                     "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                     "won't": "will not", "won't've": "will not have", "would've": "would have",
                     "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                     "y'all'd": "you all would", "y'all'd've": "you all would have",
                     "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                     "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                     "you're": "you are", "you've": "you have", "don't": "do not"}

# English stopwords  but not including 'not'-ish words (for sentiment clf reason)
stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you are", "you have", "you will",
              "you would", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she is",
              'her', 'hers', 'herself', 'it', "it is", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
              'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'that will', 'these', 'those', 'am', 'is',
              'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
              'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
              'with', 'about', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
              'from', 'up', 'down', 'in', 'on', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
              'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
              'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will', 'just', 'do', 'should', "should have", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'ma'])


class Cleaner:
    '''
    Class for performing basic nlp text cleaning:
        - make lowercase
        - remove punctuation marks
        - replace english contractions by full words
        - substitute numbers
        - remove english stopwords
        - words lemmatization
        - words stemming

    # Example for cleaning text
    >>>text = 'At 63 1/2 months old my baby is walking.'
    >>>cleaner = Cleaner(with_lematz=True, with_stemming=False)
    >>>print(cleaner.clean_text(text))
    num2 num/num month old baby walking
    '''

    def __init__(self, with_lematz=False, with_stemming=False):
        self.with_lematz = with_lematz
        self.with_stemming = with_stemming
        self.punctuation = punctuation
        self.porter = PorterStemmer()
        self.wordnet_lemmatizer = WordNetLemmatizer()
        # init contraction cleaner
        self.contraction_dict = contraction_dict
        self.contraction_re = re.compile('(%s)' % '|'.join(self.contraction_dict.keys()))

    def clean_numbers(self, x):
        if bool(re.search(r'\d', x)):
            x = re.sub('[0-9]{5,}', 'num5', x)
            x = re.sub('[0-9]{4}', 'num4', x)
            x = re.sub('[0-9]{3}', 'num3', x)
            x = re.sub('[0-9]{2}', 'num2', x)
            x = re.sub(r"\b[\d.]+\b", 'num', x)

        return x

    def replace_contractions(self, x):
        def replace(match):
            return self.contraction_dict[match.group(0)]

        return self.contraction_re.sub(replace, x)

    def clean_basics(self, x):
        x = [word.strip() for word in nltk.word_tokenize(x.lower())
             if (word not in self.punctuation) and (word not in stop_words)]

        return x

    def clean_and_lemmatize(self, x):
        x = [self.wordnet_lemmatizer.lemmatize(word.strip()) for word in nltk.word_tokenize(x.lower())
             if (word not in self.punctuation) and (word not in stop_words)]

        return x

    def clean_and_stem(self, x):
        x = [self.porter.stem(word.strip()) for word in nltk.word_tokenize(x.lower())
             if (word not in self.punctuation) and (word not in stop_words)]

        return x

    def clean_text(self, x):
        x = self.replace_contractions(x)
        x = self.clean_numbers(x)
        if self.with_lematz is False and self.with_stemming is False:
            x = self.clean_basics(x)
        elif self.with_lematz is True:
            x = self.clean_and_lemmatize(x)
        elif self.with_stemming is True:
            x = self.clean_and_stem(x)

        return ' '.join(x)