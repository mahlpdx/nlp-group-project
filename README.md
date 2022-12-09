# nlp-group-project

Group project for NLP
Approach: using this paper (https://aclanthology.org/2021.fnp-1.14.pdf) - which leverages tf-idf to do text summarization.

Our project’s motivating question: the paper uses tf-idf for large, financial documents, but does this approach scale well for smaller documents (such as reviews, news articles)?

Dataset: Multi-news. Open source dataset focused on news articles. Contains reference summaries. 

Evaluation: Use ROUGE, which is a family of metrics commonly used in text summarization research papers and compares generated summaries to reference summaries.
