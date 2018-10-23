"""
Because calculating the average topic of each video takes a while, it is saved in a separate table.
TODO after building a
"""
import psycopg2

if __name__ == '__main__':
    features_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")
    features_cursor = features_connection.cursor()

    data_connection = psycopg2.connect(database="gdelt_social_video", user="postgres")
    data_cursor = data_connection.cursor()

    features_cursor.execute("DROP TABLE IF EXISTS average_topics")
    features_cursor.execute("""CREATE TABLE average_topics (
      id TEXT NOT NULL,
      platform TEXT NOT NULL,
      topic_0 FLOAT DEFAULT 0,
      topic_1 FLOAT DEFAULT 0,
      topic_2 FLOAT DEFAULT 0,
      topic_3 FLOAT DEFAULT 0,
      topic_4 FLOAT DEFAULT 0,
      topic_5 FLOAT DEFAULT 0,
      topic_6 FLOAT DEFAULT 0,
      topic_7 FLOAT DEFAULT 0,
      topic_8 FLOAT DEFAULT 0,
      topic_9 FLOAT DEFAULT 0,

      PRIMARY KEY (id, platform)
    )""")

    columns = ["topic_%d" % index for index in range(0, 10)]
    features_cursor.execute("SELECT platform, id  FROM videos")
    videos = features_cursor.fetchall()

    for index, (platform, id) in enumerate(videos):
        # This has to be loaded into the program because its a different database
        data_cursor.execute("SELECT source_url FROM article_videos WHERE video_id=%s AND platform=%s",
                            [id, platform])
        articles = [source_url for source_url, in data_cursor]
        query = "INSERT INTO average_topics SELECT '%s', '%s'," % (id, platform) \
                + ",".join("avg(%s)" % column for column in columns) \
                + " FROM topics WHERE source_url IN " \
                + str(articles).replace("[", "(").replace("]", ")")
        features_cursor.execute(query)
        features_connection.commit()
        print(index)
