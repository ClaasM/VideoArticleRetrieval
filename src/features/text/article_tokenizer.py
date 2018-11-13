import contractions
import time

# stopwords.words('english') is a pretty slow operation, so its copied here
stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've',
             'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
             'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'}

# Augment with some more, application-specific stopwords
# For very few websites, javascript wasn't recognized correctly by boilerpipe
stopwords = stopwords.union({'said', 'say', 'var', 'also'})
# 0: size (0.0237), px (0.0109), map (0.0092), mappingadds (0.0080), group (0.0072),



def tokenize(text):
    """
    Does tokenization, de-noising and normalization of the text.
    :param text:
    :return:
    """
    # The import takes about 10% of total compute time of this function.
    # Unfortunately its necessary to import here due to https://github.com/nltk/nltk/issues/947
    from nltk import word_tokenize
    from nltk.stem import PorterStemmer

    # Replace contractions in string of text (e.g. Didn't -> did not)
    text = contractions.fix(text)
    # To lowercase
    text = text.lower()
    # Remove all non-alphabetic characters (97 <= ord(x) <= 122)
    text = "".join(filter(lambda x: x.isalpha() or x.isspace(), text))
    # Use nltk to tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords]
    # Stem
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return words
