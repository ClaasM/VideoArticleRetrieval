import gzip


def load_gzip_text(path):
    with gzip.open(path, "rb") as file:
        return file.read().decode("utf-8")


def save_gzip_text(path, text):
    with gzip.open(path, "wb+") as file:
        file.write(text.encode("utf-8"))
