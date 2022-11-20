# nlp-group-project

Group project for NLP
Approach: using this paper (https://aclanthology.org/2021.fnp-1.14.pdf) - which leverages tf-idf to do text summarization - and this repo (https://github.com/akashp1712/nlp-akash/blob/master/text-summarization/TF_IDF_Summarization.py) as reference, we will build out our text summarization model. 

Our project’s motivating question: the paper uses tf-idf for large, financial documents, but does this approach scale well for smaller documents (such as reviews, news articles)?

Dataset: TBD. Either open source dataset with documents and reference summaries OR we identify a document and use an open-source tool to create reference summaries.

Evaluation: Use ROUGE, which is a family of metrics commonly used in text summarization research papers and compares generated summaries to reference summaries.