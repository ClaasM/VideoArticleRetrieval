import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def topic_network_graph(ax, ):

    graph = nx.Graph()

    for i in range(num_topics):
        for (term_id, term_probability) in lda.get_topic_terms(topicid=i, topn=5):
            graph.add_node(i, color="#75DDFF")
            graph.add_edge(i, dictionary[term_id], weight=term_probability * 100)

    plt.figure(figsize=(16, 16))

    pos = graphviz_layout(graph, prog="twopi", root='1')
    nodes = graph.nodes()

    # nodes
    nodes = graph.nodes(data=True)
    colors = [node[1]['color'] if 'color' in node[1] else "#FFFFFF" for node in nodes]
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=2000)

    # edges
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(graph, pos, edgelist=edges, width=weights)

    # labels
    sizes = [16 if 'color' in node[1] else 8 for node in nodes]
    nx.draw_networkx_labels(graph, pos, font_size=12, font_family='sans-serif')

    plt.axis('off')
    plt.savefig('lda_network_graph.pdf', format="pdf", bbox_inches='tight')
    plt.show()

def topic_distribution_pie(ax, ):
    db = MongoClient()['thesis-dev']
    topic_frequency = defaultdict(int)
    for tweet in db.tweets.find():

        topics = lda[dictionary.doc2bow(tokenize(preprocess(tweet['text'])))]
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
