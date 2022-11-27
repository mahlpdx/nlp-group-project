from preprocessing import *
from tfidf import *
import tensorflow_datasets as tfds
import tensorflow as tf

"""This is main script used to run the pipeline

Figure from paper:

Document Corpus --> Preprocessing --> multi-word terms --> compute tf-idf
                           |                                     /
                           |                                    /     
                           V                                   /
                   generate candidate sequences               /
                         \                                   /
                          \                                 /     
                           \                               /
                            > TF-IDF scores for sequences <
                                         |
                                         |
                                         V
                                Best scored summary                         
"""

if __name__ == '__main__':
    # Load dataset
    train, test = tfds.load(
        'multi_news',
        split=['train[:20%]', 'test[:20%]']
    )
    train = [
        (ex['document'].numpy().decode("utf-8"),
        ex['summary'].numpy().decode("utf-8") )
        for ex in list(train)
    ]
    test = [
        (ex['document'].numpy().decode("utf-8"),
        ex['summary'].numpy().decode("utf-8") )
        for ex in list(test)
    ]

    # Preprocess documents to create text corpus
    corpus = [preprocess(ex[0]) for ex in train]

