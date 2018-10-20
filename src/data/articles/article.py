import gzip
from urllib.parse import urlparse
import urllib
import os

FILE_ENDING = ".gzip"
ARTICLES_PATH = os.environ["DATA_PATH"] + "/interim/articles_text/"


def get_article_filepath(url):
    """
    we ignore the fragment identifier as per https://tools.ietf.org/html/rfc3986 TODO quote
    TODO this is duplicated code from the other project
    :param url:
    :return:
    """
    parsed = urlparse(url)
    # Start with tld/domain/subdomain1/.../path1/path2/index.html?a=b#abc
    path = parsed.hostname.split(".")
    path.reverse()
    path += list(filter(None, parsed.path.split("/")))  # no empty strings
    if parsed.query:
        file_name = urllib.parse.quote_plus(parsed.query)
    else:
        file_name = path[-1]
        path = path[:-1]
    file_name += FILE_ENDING
    file_path = os.path.join(ARTICLES_PATH, *path)
    return file_path, file_name


def load(url):
    file_path, file_name = get_article_filepath(url)
    with gzip.open(file_path + "/" + file_name, "rb") as file:
        return file.read().decode("utf-8")
