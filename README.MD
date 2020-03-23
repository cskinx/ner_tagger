The NER model is based on SpaCy and trained from scratch using SpaCy.

## Requirements
Python 3.6 is required as well as the SpaCy python package (see `requirements.txt`).

## Run
To train a NER module, run `./train_ner.py --mode train --jsonl_path data/train.jsonl`
To evaluate the NER module, run `./test_ner.py --mode annotate --jsonl_path data/test.jsonl`

## Process
Each document is first split into sentences, and the entity annotations adjusted to sentence level training instances. 

## Evaluation
The results of the model can be found in `data/train_NER_out.jsonl` and `data/test_NER_out.jsonl`, respectively.
The output for the train corpus looks as follows:
```
cumulative
	Counts (Ann|NER): 69 | 0
	Precision: 0.00%
	Recall:    0.00%
	F-score:   0.00%
date_of_funding
	Counts (Ann|NER): 254 | 6946
	Precision: 2.43%
	Recall:    66.02%
	F-score:   4.69%
headquarters_loc
	Counts (Ann|NER): 751 | 17360
	Precision: 3.39%
	Recall:    77.37%
	F-score:   6.49%
investor
	Counts (Ann|NER): 1513 | 38625
	Precision: 3.18%
	Recall:    80.22%
	F-score:   6.12%
money_funded
	Counts (Ann|NER): 832 | 21164
	Precision: 3.40%
	Recall:    85.41%
	F-score:   6.54%
org_in_focus
	Counts (Ann|NER): 1410 | 36264
	Precision: 3.72%
	Recall:    85.99%
	F-score:   7.14%
org_url
	Counts (Ann|NER): 118 | 4293
	Precision: 2.42%
	Recall:    85.95%
	F-score:   4.71%
type_of_funding
	Counts (Ann|NER): 433 | 9803
	Precision: 3.33%
	Recall:    74.09%
	F-score:   6.37%
valuation
	Counts (Ann|NER): 24 | 0
	Precision: 0.00%
	Recall:    0.00%
	F-score:   0.00%
year_founded
	Counts (Ann|NER): 240 | 8060
	Precision: 2.84%
	Recall:    89.11%
	F-score:   5.51%
Evaluation for all entities:
	Counts (Ann|NER): 5644 | 142515
	Precision: 3.31%
	Recall:    80.30%
	F-score:   6.36%
```

And for the test corpus:
```
cumulative
	Counts (Ann|NER): 0 | 0
date_of_funding
	Counts (Ann|NER): 0 | 266
headquarters_loc
	Counts (Ann|NER): 0 | 484
investor
	Counts (Ann|NER): 0 | 1300
money_funded
	Counts (Ann|NER): 0 | 531
org_in_focus
	Counts (Ann|NER): 0 | 1209
org_url
	Counts (Ann|NER): 0 | 310
type_of_funding
	Counts (Ann|NER): 0 | 270
valuation
	Counts (Ann|NER): 0 | 0
year_founded
	Counts (Ann|NER): 0 | 565
Evaluation for all entities:
	Counts (Ann|NER): 0 | 4935
```


## Discussion
The NER model is obviously far from perfect and has a lot of false positives. However, the task is also quite difficult, since there are a lot of company names and it's easy to confuse e.g. an investor with a regular mentioned company name. 
The model was not trained to its full capacity, however, since the time limit only made it possible to train for 20 epochs. Since the loss was still decreasing, more training would definitely help. However, the model probably needs to be more sophisticated for the task to significantly improve the precision.