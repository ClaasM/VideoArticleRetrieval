import numpy as np

class Text2Vec:

    def __init__(self, datafile, ndims=0, L1_normalize=0, L2_normalize=0):
        self.datafile = datafile
        self.ndims = ndims
        self.L1_normalize = L1_normalize
        self.L2_normalize = L2_normalize


    def embedding(self, query):
        vec = self.mapping(query)
        if vec is not None:
            vec = np.array(vec)
        return vec

    def do_L1_norm(self, vec):
        L1_norm = np.linalg.norm(vec, 1)
        return 1.0 * np.array(vec) / L1_norm

    def do_L2_norm(self, vec):
        L2_norm = np.linalg.norm(vec, 2)
        return 1.0 * np.array(vec) / L2_norm


# word2vec + average pooling
class AveWord2Vec(Text2Vec):

    # datafile: the path of pre-trained word2vec data
    def __init__(self, datafile, ndims=0, L1_normalize=0, L2_normalize=0):
        Text2Vec.__init__(self, datafile, ndims, L1_normalize, L2_normalize)
        self.word2vec = BigFile(datafile)
        if ndims != 0:
            assert self.word2vec.ndims == self.ndims, "feat dimension is not match %d != %d" % (
                self.word2vec.ndims, self.ndims)
        else:
            self.ndims = self.word2vec.ndims

    def preprocess(self, query, clear):
        if clear:
            words = clean_str(query)
        else:
            words = query.strip().split()
        return words

    def mapping(self, query, clear=True):
        words = self.preprocess(query, clear)

        # print query, '->', words
        renamed, vectors = self.word2vec.read(words)
        renamed2vec = dict(zip(renamed, vectors))

        if len(renamed) != len(words):
            vectors = []
            for word in words:
                if word in renamed2vec:
                    vectors.append(renamed2vec[word])

        if len(vectors) > 0:
            vec = np.array(vectors).mean(axis=0)

            if self.L1_normalize:
                return self.do_L1_norm(vec)
            if self.L2_normalize:
                return self.do_L2_norm(vec)
            return vec
        else:
            return None

# word2vec + average pooling + fliter stop words
class AveWord2VecFilterStop(AveWord2Vec):

    # datafile: the path of pre-trained word2vec data
    def __init__(self, datafile, ndims=0, L1_normalize=0, L2_normalize=0):
        Text2Vec.__init__(self, datafile, ndims, L1_normalize, L2_normalize)
        self.word2vec = BigFile(datafile)
        if ndims != 0:
            assert self.word2vec.ndims == self.ndims, "feat dimension is not match %d != %d" % (
                self.word2vec.ndims, self.ndims)
        else:
            self.ndims = self.word2vec.ndims

    def preprocess(self, query, clear):
        if clear:
            words = clean_str_filter_stop(query)
        else:
            words = query.strip().split()
        return words

    # Bag-of-words


class BoW2Vec(Text2Vec):

    # datafile: the path of bag-of-words vocabulary file
    def __init__(self, datafile, ndims=0, L1_normalize=0, L2_normalize=0):
        Text2Vec.__init__(self, datafile, ndims, L1_normalize, L2_normalize)
        word_vob = map(str.strip, open(datafile).readlines())
        self.word2index = dict(zip(word_vob, range(len(word_vob))))
        if ndims != 0:
            assert len(word_vob) == self.ndims, "feat dimension is not match %d != %d" % (len(word_vob), self.ndims)
        else:
            self.ndims = len(word_vob)
        printStatus(INFO + '.' + self.__class__.__name__, "%d words" % self.ndims)

    def preprocess(self, query):
        return clean_str(query)

    def mapping(self, query):
        words = self.preprocess(query)

        vec = [0.0] * self.ndims
        for word in words:
            if word in self.word2index:
                vec[self.word2index[word]] += 1
            # else:
            #     print word

        if sum(vec) > 0:
            if self.L1_normalize:
                vec = self.do_L1_norm(vec)
            if self.L2_normalize:
                vec = self.do_L2_norm(vec)
            return vec

        ###############################
        # sometimes need to modify
        # else:
        #     return None
        ###############################
        else:
            return vec

# Bag-of-words + fliter stop words
class BoW2VecFilterStop(BoW2Vec):

    def preprocess(self, query):
        return clean_str_filter_stop(query)


