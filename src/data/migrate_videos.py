"""
Takes the videos from the dataset and creates a table to store their various analysis/status
Takes a couple seconds.
"""
import psycopg2

data_connection = psycopg2.connect(database="gdelt_social_video", user="postgres")
data_cursor = data_connection.cursor()
# We only want videos that have been crawled successfully
data_cursor.execute("SELECT id, platform FROM videos WHERE crawling_status='Success'")
videos = data_cursor.fetchall()

# Connect to the database where the results will be saved
results_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")
results_cursor = results_connection.cursor()

for id, platform in videos:
    results_cursor.execute("INSERT INTO videos (id, platform)  VALUES (%s, %s)", [id, platform])
    results_connection.commit()
