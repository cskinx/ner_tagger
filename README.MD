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
The output of the 

## Discussion
The NER model is obviously far from perfect and has a lot of false positives. However, the task is also quite difficult, since there are a lot of company names and it's easy to confuse e.g. an investor with a regular mentioned company name. 
The model was not trained to its full capacity, however, since the time limit only made it possible to train for 20 epochs. Since the loss was still decreasing, more training would definitely help. However, the model probably needs to be more sophisticated for the task to significantly improve the precision.