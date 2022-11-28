from rouge_metric import PyRouge
"""
Functions to evaluate generated and reference
summaries with ROUGE metrics. 

Functions to calculate recall, precision and F-measure
for each rouge metric. 

Using this library: https://pypi.org/project/rouge-metric/
"""
"""
hypotheses = [
    'how are you\ni am fine',  # document 1: hypothesis
    'it is fine today\nwe won the football game',  # document 2: hypothesis
]
references = [[
    'how do you do\nfine thanks',  # document 1: reference 1
    'how old are you\ni am three',  # document 1: reference 2
], [
    'it is sunny today\nlet us go for a walk',  # document 2: reference 1
    'it is a terrible day\nwe lost the game',  # document 2: reference 2
]]
"""

def evaluation(hypotheses, references):
    # Evaluate document-wise ROUGE scores
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    scores = rouge.evaluate(hypotheses, references)
    print(scores)


if __name__ == '__main__':
    # Run test cases here
    generated_summary = ['My name is jacob and I like water']
    reference_summary = ['My title is jacob and I enjoy drinking']

    evaluation(generated_summary,reference_summary)
