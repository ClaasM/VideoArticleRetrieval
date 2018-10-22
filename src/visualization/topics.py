from collections import defaultdict

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def topic_network_graph(lda, dictionary, num_topics):
    graph = nx.Graph()

    for i in range(num_topics):
        for (term_id, term_probability) in lda.get_topic_terms(topicid=i, topn=5):
            graph.add_node(i, color="#75DDFF")
            graph.add_edge(i, dictionary[term_id], weight=term_probability * 100)

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


def topic_distribution_pie(articles, lda, dictionary):
    topic_frequency = defaultdict(int)
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


def print_topics(lda, dictionary, num_topics):
    for i in range(num_topics):
        print("\n%d: " % i, end='')
        for (term_id, term_probability) in lda.get_topic_terms(topicid=i, topn=5):
            print("%s (%.4f), " % (dictionary[term_id], term_probability), end='')
