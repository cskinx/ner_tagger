The NER model is based on SpaCy and trained from scratch using SpaCy.

__Note__: There was a bug in the code that is fixed now, which falsely multiplied the found entities in the text and produced wrong results. After fixing this, the performance increased significantly from <8% F-score to 95%.

## Requirements
Python 3.6 is required as well as the SpaCy python package (see `requirements.txt`).

## Run
To train a NER module, run `./train_ner.py --mode train --jsonl_path data/train.jsonl`
To evaluate the NER module, run `./train_ner.py --mode annotate --jsonl_path data/test.jsonl`

## Process
Each document is first split into sentences, and the entity annotations adjusted to sentence level training instances. 

## Evaluation
The results of the model can be found in `data/train_NER_out.jsonl` and `data/test_NER_out.jsonl`, respectively.

With 100 training epochs, the output for the training corpus is as follows:
```
cumulative
	Counts (Ann|NER): 69 | 71
	Precision: 77.46%
	Recall:    79.71%
	F-score:   78.57%
date_of_funding
	Counts (Ann|NER): 254 | 256
	Precision: 80.08%
	Recall:    80.08%
	F-score:   80.08%
headquarters_loc
	Counts (Ann|NER): 751 | 777
	Precision: 89.06%
	Recall:    91.66%
	F-score:   90.34%
investor
	Counts (Ann|NER): 1513 | 1763
	Precision: 83.78%
	Recall:    96.85%
	F-score:   89.84%
money_funded
	Counts (Ann|NER): 832 | 892
	Precision: 90.02%
	Recall:    95.48%
	F-score:   92.67%
org_in_focus
	Counts (Ann|NER): 1410 | 1484
	Precision: 88.07%
	Recall:    90.45%
	F-score:   89.25%
org_url
	Counts (Ann|NER): 118 | 129
	Precision: 79.84%
	Recall:    87.29%
	F-score:   83.40%
type_of_funding
	Counts (Ann|NER): 433 | 479
	Precision: 83.09%
	Recall:    90.66%
	F-score:   86.71%
valuation
	Counts (Ann|NER): 24 | 13
	Precision: 53.85%
	Recall:    29.17%
	F-score:   37.84%
year_founded
	Counts (Ann|NER): 240 | 256
	Precision: 91.02%
	Recall:    97.08%
	F-score:   93.95%
Evaluation for all entities:
	Counts (Ann|NER): 5644 | 6120
	Precision: 86.27%
	Recall:    92.44%
	F-score:   89.25%
```

And for the test corpus:
```
cumulative
	Counts (Ann|NER): 0 | 1
date_of_funding
	Counts (Ann|NER): 0 | 11
headquarters_loc
	Counts (Ann|NER): 0 | 26
investor
	Counts (Ann|NER): 0 | 76
money_funded
	Counts (Ann|NER): 0 | 29
org_in_focus
	Counts (Ann|NER): 0 | 29
org_url
	Counts (Ann|NER): 0 | 6
type_of_funding
	Counts (Ann|NER): 0 | 9
valuation
	Counts (Ann|NER): 0 | 0
year_founded
	Counts (Ann|NER): 0 | 14
Evaluation for all entities:
	Counts (Ann|NER): 0 | 201
```

Note that there is no valid evaluation here so far, since we trained on the full training set without separating a part of it into a development set, and we don't know the result for the test set.