import psycopg2
import pandas as pd
from src import constants


def get_occurence_rate():
    """
    For each video, for each label, the sum of the probability of all occurrences of that label,
    divided by the duration of the video
    TODO this should probably be one query and should only return a cursor, with a connection as an argument
    :return: Dataframe
    """
    # Get all the label probability sums
    features_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")
    query = "SELECT platform, id, SUM(probability) as total, "
    query += ",".join("SUM(CASE WHEN class='%s' THEN probability END) as %s"
                      % (label, label.replace(" ", "_"))
                      for label in constants.COCO_CLASS_NAMES)
    query += " FROM object_detection_yolo GROUP BY (platform,id)"
    features = pd.read_sql(query, con=features_connection)
    features = features.fillna(0)

    # Get the duration of each video
    # TODO this can probably be done easier
    data_connection = psycopg2.connect(database="gdelt_social_video", user="postgres")
    data_cursor = data_connection.cursor()
    def get_duration(row):
        data_cursor.execute("SELECT duration FROM videos WHERE platform=%s AND id=%s", [row['platform'], row['id']])
        (duration,) = data_cursor.fetchone()
        return duration

    features['duration'] = features.apply(get_duration, axis=1)

    # Divide by said duration
    for label in constants.COCO_CLASS_NAMES:
        column_name = label.replace(" ", "_")
        features[column_name] = features[column_name] / features["duration"]

    return features
