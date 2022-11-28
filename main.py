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
        split=['train[:1%]', 'test[:1%]']
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
    # Each sample contains multiple news articles so
    # we arbitrarily select the first one
    train_corpus = [
        preprocess(ex[0].split("|||||")[0]) for ex in train
    ]
    test_corpus = [
        preprocess(ex[0].split("|||||")[0]) for ex in test
    ] 
    
    # Term size to be used for corpus
    term_size = 1

    # Load mapping of term to idf
    print ("Processing idf_map...")
    idf_map = idf_mapping(train_corpus, term_size)
    print ("Completed idf map!")

    # Loop through test set to generate summaries
    print ("Generating summaries...")
    predicted_summaries = []
    for i in range(5):
        tf_map = tf_mapping(test_corpus[i], term_size)
        predicted_summaries.append(
            generate_summary(test_corpus[i], tf_map, idf_map, 1000, term_size)
        )
    print ("Completed summaries...")

    # Apply ROUGE evaluation to generated summaries

