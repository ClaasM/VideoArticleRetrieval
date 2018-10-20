"""
This could be a separate Repository as well.
A simplified version of python-boilerpipe
Fewer dependencies and LOC.
Not thread-safe, uses only ArticleExtractor, can only take an html string as input.
Easy to modify, however.
"""
import os

import jpype


class BoilerPipeArticleExtractor:
    def __init__(self):
        if not jpype.isJVMStarted():
            jars_path = os.path.dirname(os.path.abspath(__file__)) + "/jars"
            jars = [
                jars_path + "/boilerpipe-1.2.0.jar",
                jars_path + "/nekohtml-1.9.13.jar",
                jars_path + "/xerces-2.9.1.jar",
            ]
            args = "-Djava.class.path=%s" % os.pathsep.join(jars)
            jpype.startJVM(jpype.getDefaultJVMPath(), args)

        if not jpype.isThreadAttachedToJVM():
            jpype.attachThreadToJVM()

        self.InputSource = jpype.JClass('org.xml.sax.InputSource')
        self.StringReader = jpype.JClass('java.io.StringReader')
        self.HTMLHighlighter = jpype.JClass('de.l3s.boilerpipe.sax.HTMLHighlighter')
        self.BoilerpipeSAXInput = jpype.JClass('de.l3s.boilerpipe.sax.BoilerpipeSAXInput')

        self.extractor = jpype.JClass("de.l3s.boilerpipe.extractors.ArticleExtractor").INSTANCE

    def get_text(self, html):
        self.source = self.BoilerpipeSAXInput(self.InputSource(self.StringReader(html))).getTextDocument()
        self.extractor.process(self.source)
        return self.source.getContent()


test_gzip_html = "/Volumes/DeskDrive/data/raw/articles/co/tech/will-5bn-eu-fine-kill-android-2018-07.gzip"
if __name__ == "__main__":
    from src.data.articles import article as article_helper

    html = article_helper.load_file(test_gzip_html)
    extractor = BoilerPipeArticleExtractor()
    text = extractor.get_text(html)
    print(text)
