import psycopg2
import pandas as pd

data_connection = psycopg2.connect(database="gdelt_social_video", user="postgres")
features_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")


def get_average_topic(video_id, video_platform):
    """
    Given a video, this returns the average of the topic vectors of all articles embedding this video.
    :return:
    """
    articles_cursor = data_connection.cursor()
    articles_cursor.execute("SELECT source_url FROM article_videos WHERE video_id=%s AND platform=%s",
                            [video_id, video_platform])
    articles = [source_url for source_url, in articles_cursor]
    columns = get_topic_columns(10)
    query = "SELECT " \
            + ",".join("avg(%s) AS %s" % (column, column) for column in columns) \
            + " FROM topics WHERE source_url IN " \
            + str(articles).replace("[", "(").replace("]", ")")
    avg_topic_cursor = features_connection.cursor()
    avg_topic_cursor.execute(query)
    return avg_topic_cursor.fetchone()


def get_topic_columns(num_topics=10):
    """
    TODO use this everywhere
    :param num_topics:
    :return:
    """
    return ["topic_%d" % index for index in range(0, num_topics)]





if __name__ == "__main__":
    print(get_average_topic("fox26houston/10157181811750348", "facebook"))
