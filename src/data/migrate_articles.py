"""
Takes the articles from the dataset and creates a table to store their various analysis/status
Takes a couple minutes.
"""
import psycopg2

data_connection = psycopg2.connect(database="gdelt_social_video", user="postgres")
data_cursor = data_connection.cursor()
# We only want articles which contain videos that were also successfully downloaded and are present in the dataset.
data_cursor.execute("""SELECT DISTINCT av.source_url
                      FROM article_videos AS av LEFT JOIN videos v ON (av.video_id, av.platform) = (v.id, v.platform)
                      WHERE v.crawling_status = 'Success'""")

articles = data_cursor.fetchall()

# Connect to the database where the results will be saved
results_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")
results_cursor = results_connection.cursor()

for source_url, in articles:
    results_cursor.execute("INSERT INTO articles (source_url)  VALUES (%s)", [source_url])
    results_connection.commit()

# TODO migrate article_videos
