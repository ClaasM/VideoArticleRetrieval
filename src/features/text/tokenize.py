import contractions


def tokenize(text):
    """
    Does tokenization, de-noising and normalization of the text.
    :param text:
    :return:
    """
    # The import takes about 10% of total compute time of this function.
    # Unfortunately its necessary to import here due to https://github.com/nltk/nltk/issues/947
    from nltk import word_tokenize
    from nltk.corpus import stopwords
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
    words = [word for word in words if word not in stopwords.words('english')]
    # Stem
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return words
