/*
Create a database called "video_article_retrieval" first:
CREATE DATABASE video_article_retrieval;
Connect to it:
\c video_article_retrieval
Then run this script to initialize it.
*/

-- TABLES

CREATE TABLE IF NOT EXISTS object_detection_tiny_yolo
(
  -- contains all objects in the videos detected by YOLO
  -- the dector is run every 1 second.
  id          TEXT  NOT NULL, --the video_id is extracted from the url when crawling it (to make querying for it faster).
  platform    TEXT  NOT NULL,
  second      INT   NOT NULL,
  class       TEXT  NOT NULL,
  probability FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS object_detection_yolo
(
  id          TEXT  NOT NULL,
  platform    TEXT  NOT NULL,
  second      INT   NOT NULL,
  class       TEXT  NOT NULL,
  probability FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS videos
(
  id                           TEXT NOT NULL, --the video_id is extracted from the url when crawling it (to make querying for it faster).
  platform                     TEXT NOT NULL,

  object_detection_yolo_status TEXT DEFAULT 'Not Processed',
  resnet_status                TEXT DEFAULT 'Not Processed',
  restnet_2048                 BYTEA,
  soundnet_status              TEXT DEFAULT 'Not Processed',
  soundnet_1024                BYTEA,
  i3d_rgb_status               TEXT DEFAULT 'Not Processed',
  i3d_rgb_1024                 BYTEA,
  i3d_flow_status              TEXT DEFAULT 'Not Processed',
  i3d_flow_1024                BYTEA,
  PRIMARY KEY (platform, id)
);

CREATE TABLE IF NOT EXISTS articles
(
  id                        SERIAL PRIMARY KEY,
  source_url                TEXT UNIQUE NOT NULL,
  text                      TEXT,
  text_extraction_status    TEXT DEFAULT 'Not Tried',
  tokens                    TEXT,
  bow_2048                  BYTEA,
  w2v_2048                  BYTEA,
  feature_extraction_status TEXT default 'Not Processed'
);

-- INDICES
-- Indices are only created where they are really needed, because they take up space and slow down inserts/deletes
CREATE INDEX IF NOT EXISTS object_detection_yolo_id_index
  ON public.object_detection_tiny_yolo (id);
CREATE INDEX IF NOT EXISTS object_detection_yolo_platform_index
  ON public.object_detection_tiny_yolo (platform);

CREATE INDEX IF NOT EXISTS object_detection_tiny_yolo_id_index
  ON public.object_detection_tiny_yolo (id);
CREATE INDEX IF NOT EXISTS object_detection_tiny_yolo_platform_index
  ON public.object_detection_tiny_yolo (platform);
