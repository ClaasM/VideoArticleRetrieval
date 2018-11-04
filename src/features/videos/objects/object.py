import psycopg2
import pandas as pd
from src import constants


def get_occurence_sum(model="yolo"):
    features_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")
    query = "SELECT platform, id, "
    query += ",".join("SUM(CASE WHEN class='%s' THEN 1 END) as %s"
                      % (label, label.replace(" ", "_")) for label in constants.COCO_CLASS_NAMES)
    query += " FROM object_detection_%s GROUP BY (platform,id)" % model
    features = pd.read_sql(query, con=features_connection)
    return features.fillna(0)


def get_probability_sum(model="yolo"):
    # Get all the label probability sums
    features_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")
    query = "SELECT platform, id, "
    query += ",".join("SUM(CASE WHEN class='%s' THEN probability END) as %s"
                      % (label, label.replace(" ", "_")) for label in constants.COCO_CLASS_NAMES)
    query += " FROM object_detection_%s GROUP BY (platform,id)" % model
    features = pd.read_sql(query, con=features_connection)
    return features.fillna(0)


def get_probability_size_sum(model="yolo"):
    features_connection = psycopg2.connect(database="video_article_retrieval", user="postgres")
    # Calculates the size of the object in percent of the size of the frame.
    size_formula = "" # TODO
    query = "SELECT platform, id, "
    query += ",".join("SUM(CASE WHEN class='%s' THEN %s END) as %s"
                      % (label, size_formula, label.replace(" ", "_")) for label in constants.COCO_CLASS_NAMES)
    query += " FROM object_detection_%s GROUP BY (platform,id)" % model
    features = pd.read_sql(query, con=features_connection)
    return features.fillna(0)


def divide_by_duration(object_detection_dataframe):
    """
    For each video, for each label, the sum of the probability of all occurrences of that label,
    divided by the duration of the video
    TODO this should probably be one query and should only return a cursor, with a connection as an argument
    :return: Dataframe
    """

    # Get the duration of each video
    # TODO this can probably be done easier
    data_connection = psycopg2.connect(database="gdelt_social_video", user="postgres")
    data_cursor = data_connection.cursor()

    def get_duration(row):
        data_cursor.execute("SELECT duration FROM videos WHERE platform=%s AND id=%s", [row['platform'], row['id']])
        (duration,) = data_cursor.fetchone()
        return duration

    object_detection_dataframe['duration'] = object_detection_dataframe.apply(get_duration, axis=1)

    # Divide by said duration
    for label in constants.COCO_CLASS_NAMES:
        column_name = label.replace(" ", "_")
        object_detection_dataframe[column_name] = \
            object_detection_dataframe[column_name] / object_detection_dataframe["duration"]

    return object_detection_dataframe
