import glob
import os

from src.boilerpipe.extract import Extractor
from src import util

articles_path = os.environ["DATA_PATH"] + "/raw/articles/"
text_path = os.environ["DATA_PATH"] + "/interim/articles_text/"


# Create the text dir if it does not exist yet
def run():
    if not os.path.exists(text_path):
        os.makedirs(text_path)

    articles = glob.glob(os.path.join(articles_path, "http*"))

    for path, subdirs, files in os.walk(root):
        for name in files:
            print
            os.path.join(path, name)

    for file_path in articles:
        html = util.load_gzip_html(file_path)
        print(html[:100])
        text = Extractor(extractor='ArticleExtractor', html=html).getText()
        with open(os.path.join(text_path, file_path.split("/")[-1]), "w+") as f:
            f.write(text)


if __name__ == "__main__":
    run()
