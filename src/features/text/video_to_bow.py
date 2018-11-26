# For each video, this creates the 2048-vector of the tokens in each of the articles the video is embedded in.


freq_threshold = 5

def run():
    word2counter = {}
    len2counter ={}

    # output_vocab_file = os.path.join( rootpath, "%s/TextData/vocabulary/%s/word_vocab_%d.txt" % (collection, text_style, freq_threshold) )
    # output_vocab_counter_file = os.path.join( rootpath, "%s/TextData/vocabulary/%s/word_vocab_counter_%d.txt" % (collection, text_style, freq_threshold) )
    # 1000268201_693b08cb0e.jpg#enc#0 A child in a pink dress is climbing up a set of stairs in an entry way .

    for index, line in enumerate(open(input_file)):
        sid, sent = line.strip().split(" ", 1)
        sent = clean_str(sent)
        length = len(sent)
        len2counter[length] = len2counter.get(length, 0) + 1

        for word in sent:
            word2counter[word] = word2counter.get(word, 0) + 1

    sorted_wordCounter = sorted(word2counter.iteritems(), key = lambda a:a[1], reverse=True)


    output_line_vocab = [ x[0] for x in sorted_wordCounter if x[1] >= freq_threshold ]
    output_line_vocab_counter = [ x[0] + ' '  + str(x[1]) for x in sorted_wordCounter if x[1] >= freq_threshold ]

    open(output_vocab_file, 'w').write('\n'.join(output_line_vocab))
    open(output_vocab_counter_file, 'w').write('\n'.join(output_line_vocab_counter))

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()


text_batch = []
        for sent in raw_sent_batch:
            sent_bow_vec = bow2vec.mapping(sent)
            sent_w2v_vec = w2v2vec.mapping(sent)
            if sent_bow_vec is not None and sent_w2v_vec is not None:
                text_batch.append(list(sent_bow_vec) + list(sent_w2v_vec))