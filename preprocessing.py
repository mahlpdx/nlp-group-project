import nltk
"""
Functions to read in corpora.
Functions to split corpora into train, validation (?), 
and test sets.
Functions to preprocess documents.
"""


def preprocess(document):
    """Preprocess a single document.

    Splits text, tokenizes, removes specials, phone numbers, emails.
    Uses custom stopword list.

    Inputs:
        Document (str): word document

    Return:
        (str): processed document

    """
    stop_words = [
        "and", "the", "is", "are", " this", "at", "of", "to", "in", "on",
        "for", "or", "a", "an", "as", "page", "by", "with", "our", "we", "that",
        "may"
    ]


if __name__ == '__main__':
    # Run test cases here
    print(1)
