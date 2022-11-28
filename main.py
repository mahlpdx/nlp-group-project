from preprocessing import *
from evaluation import *
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
    # 1.  Load dataset
    train, test = tfds.load(
        'multi_news',
        split=['train[:100]', 'test[:100]']
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

    # 2. Preprocess documents to create text corpus
    
    # Each sample contains multiple news articles so
    # we arbitrarily select the first one
    train_corpus = [
        preprocess(ex[0].split("|||||")[0]) for ex in train
    ]
    test_corpus = [
        preprocess(ex[0].split("|||||")[0]) for ex in test
    ] 
    
    # 3. Assign term size to be used for corpus
    term_size = 3

    # 4. Load mapping of term to idf
    print ("Processing idf_map...")
    idf_map = idf_mapping(train_corpus, term_size)
    print ("Completed idf map!")

    # 5. Loop through test set to generate summaries
    print ("Generating summaries...")
    predicted_summaries = []

    # Assign test size (number of samples)
    test_size = 5
    summary_size = 100
    for i in range(test_size):
        tf_map = tf_mapping(test_corpus[i], term_size)
        predicted_summaries.append(
            generate_summary(test_corpus[i], tf_map, idf_map, summary_size, term_size)
        )
    
    print ("Completed summaries!")
    print ("Metric evaluation...")
    
    # 6.  Apply ROUGE evaluation to generated summaries
    # Currently only applying to a single sample
    eval = Evaluation()
    scores = eval.evaluation(summary_str(predicted_summaries[0]), test[0][1])
    eval.score_display(scores)
    print ("Completed evaluation!")

