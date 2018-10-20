"""
Classifies all articles according to the LDA model
"""


def classify(text):

conn = psycopg2.connect(database="video_article_retrieval", user="postgres")
c = conn.cursor()
c.execute("SELECT source_url, text FROM articles WHERE text_extraction_status='Success' LIMIT 1000")
articles = [(source_url, tokenize.tokenize(text)) for (source_url, text) in c]

token_frequency = defaultdict(int)
for source_url, tokens in articles:
    for token in tokens:
        token_frequency[token] += 1

for article in articles:

    topics = lda[dictionary.doc2bow(tokens)]
    for topic in topics:
        # print(topic)
        topic_frequency[topic[0]] += topic[1]

import matplotlib.pyplot as plt
from matplotlib import rcParams

labels, values = [], []
for key in topic_frequency:
    values.append(topic_frequency[key])
    labels.append(", ".join(dictionary[term_id] for (term_id, _) in lda.get_topic_terms(topicid=key, topn=5)))
plt.pie(values, labels=labels, shadow=False, autopct='%1.1f%%', )
plt.axis('equal')
plt.savefig('topic_distribution.pdf', format="pdf", bbox_inches='tight')
plt.show()

def run():
    conn = psycopg2.connect(database="gdelt_social_video", user="postgres")

    c = conn.cursor()
    c.execute("SELECT DISTINCT source_url from article_videos")
    articles = c.fetchall()

    crawling_progress = CrawlingProgress(len(articles), update_every=10000)

    # parallel crawling and parsing to speed things up
    results = list()
    with Pool(32)  as pool:  # 16 seems to be around optimum
        for result in pool.imap_unordered(crawl_article, articles, chunksize=100):
            results.append(result[:100])
            crawling_progress.inc(1)

    print(Counter(results))

if __name__ == "__main__":
    run()