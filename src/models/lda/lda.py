import psycopg2

data_connection = psycopg2.connect(database="gdelt_social_video", user="postgres")
features_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")


def get_average_topic(video_id, video_platform):
    """
    Given a video, this returns the average of the topic vectors of all articles embedding this video.
    :return:
    """

    c = data_connection.cursor()
    c.execute("SELECT source_url FROM article_videos WHERE video_id=%s AND platform=%s", [video_id, video_platform])
    topic_vectors = list()
    for source_url, in c:
        print(source_url)
        topic_vectors.append(get_topics(source_url))

    print(topic_vectors)
    # TODO return average


def get_topics(source_url):
    features_cursor = features_connection.cursor()
    query = "SELECT " + ",".join(get_topic_columns(10)) + " FROM topics WHERE source_url='%s'" % source_url
    features_cursor.execute(query)  # , [source_url]
    return features_cursor.fetchone()


def get_topic_columns(num_topics=10):
    """
    TODO use this everywhere
    :param num_topics:
    :return:
    """
    return ["topic_%d" % index for index in range(0, num_topics)]


if __name__ == "__main__":
    get_average_topic("fox26houston/10157181811750348", "facebook")
