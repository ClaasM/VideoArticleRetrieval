import gzip
from urllib.parse import urlparse
import urllib
import os

FILE_ENDING = ".gzip"
ARTICLES_TEXT_PATH = os.environ["DATA_PATH"] + "/interim/articles_text/"
ARTICLES_HTML_PATH = os.environ["DATA_PATH"] + "/raw/articles/"


def get_article_html_filepath(url):
    file_path, file_name = url_to_path(url)
    file_path = os.path.join(ARTICLES_HTML_PATH, file_path)
    return file_path, file_name


def get_article_text_filepath(url):
    file_path, file_name = url_to_path(url)
    file_path = os.path.join(ARTICLES_TEXT_PATH, file_path)
    return file_path, file_name


def url_to_path(url):
    """
    we ignore the fragment identifier as per https://tools.ietf.org/html/rfc3986 TODO quote
    TODO this is duplicated code from the other project
    TODO this could also be a package
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
    file_path = os.path.join(*path)
    return file_path, file_name


def load(url):
    file_path, file_name = get_article_html_filepath(url)
    return load_file(file_path + "/" + file_name)


def load_file(file_path):
    """
    :param file_path: Path of the file, absolute or relative to this file
    :return:
    """
    with gzip.open(file_path, "rb") as file:
        return file.read().decode("utf-8")
