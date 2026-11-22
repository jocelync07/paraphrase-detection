PAWS labeled_final: 5,000 training pairs, 1,000 test pairs (seed 42).

| Model | Accuracy | Precision | Recall | F1 (paraphrase) | Macro-F1 |
|---|---|---|---|---|---|
| Majority class | 0.571 | 0.000 | 0.000 | 0.000 | 0.363 |
| Naive Bayes (8 overlap features) | 0.722 | 0.677 | 0.674 | 0.675 | 0.716 |
| Logistic regression (same features) | 0.761 | 0.751 | 0.662 | 0.704 | 0.752 |
| Sentence-BERT, pre-trained (threshold) | 0.622 | 0.621 | 0.305 | 0.409 | 0.566 |
| Sentence-BERT, fine-tuned (threshold) | 0.732 | 0.677 | 0.718 | 0.697 | 0.728 |
| DistilBERT cross-encoder, fine-tuned | 0.579 | 0.512 | 0.389 | 0.442 | 0.552 |

Retrieval (query = sentence 1 of each test paraphrase; candidates = all test sentence 2s):

| Sentence-BERT | Recall@1 | Recall@5 | MRR | Similarity gap (para - non-para) |
|---|---|---|---|---|
| Pre-trained | 0.753 | 1.000 | 0.866 | 0.010 |
| Fine-tuned | 0.809 | 1.000 | 0.899 | 0.123 |

DistilBERT training:

| Epoch | Train loss | Test accuracy | Minutes |
|---|---|---|---|
| 1 | 0.6861 | 0.564 | 14.3 |
| 2 | 0.6571 | 0.571 | 7.7 |
| 3 | 0.5594 | 0.579 | 47.1 |
