"""
Preprocessing the text extracted from the articles by boilerpipe:
-
"""

import re


def preprocess(text):
    # Remove all the other stuff
    return " ".join([word for word in re.split("[\s;,.#:-@!?'\"]", text) if word.isalpha()])
