#!/usr/bin/env python
# coding: utf8

import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import data_wrangler
import argparse
from collections import defaultdict

def train(train_data, output_dir=None, n_iter=20):
    """Load the model, set up the pipeline and train the entity recognizer."""
    start_with_en = False
    if start_with_en:
        ## start with pretrained model
        nlp = spacy.load("en")
        ner = nlp.get_pipe("ner")
    else:
        ## create NER model from scratch
        nlp = spacy.blank("en")
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

    # collect all distinct labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly
        nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, anns in train_data[:200]:
        doc = nlp(text)
        if len(anns.get("entities")) > 0:
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def annotate(model_path, corpus_sents):
    """load model and annotate the given corpus."""
    nlp = spacy.load(model_path)
    annotated = []

    ents = []
    for text in corpus_sents:
        doc = nlp(text)
        for ent in doc.ents:
            ent_pos = text.index(ent.text)
            ent_end_pos = ent_pos + len(ent.text)
            ents.append((ent_pos, ent_end_pos, ent.label_))
        # if len(doc.ents) > 0:
        #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
        #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #     print(annotated)

        annotated.append((text, {"entities": ents}))

    return annotated

def evaluate(docs, ent_type = "all", only_counts=False):
    """ has a tuple for each doc: list of annotated entities
    as well as a list of NER entities."""
    ann_counts = 0
    ner_counts = 0

    TP = 0
    FP = 0
    FN = 0
    for ann_ents, ner_ents in docs:
        ## iterate through NER entities
        for ner_start, ner_end, ner_type in ner_ents:
            if ent_type == "all" or ent_type == ner_type:
                ner_counts += 1
                found = False
                for ann_start, ann_end, ann_type in ann_ents:
                    if ann_start == ner_start and ann_end == ner_end and ner_type == ann_type:
                        found = True
                if found:
                    TP += 1
                else:
                    FP += 1

        ## iterate through annotated entities for false negatives
        for ann_start, ann_end, ann_type in ann_ents:
            if ent_type == "all" or ann_type == ent_type:
                ann_counts += 1
                found = False
                for ner_start, ner_end, ner_type in ner_ents:
                    if ann_start == ner_start and ann_end == ner_end and ner_type == ann_type:
                        found = True
                if not found:
                    FN += 1

    if ent_type == "all":
        print("Evaluation for all entities:")
    else:
        print(f"{ent_type}")
    print(f"\tCounts (Ann|NER): {ann_counts} | {ner_counts}")
    if only_counts:
        return
    
    precision = 0
    if TP+FP > 0:
        precision = TP / (TP+FP)

    recall = 0
    if TP+FN > 0:
        recall = TP / (TP+FN)

    fscore = 0
    if precision + recall > 0:
        fscore = (2*precision*recall) / (precision+recall)

    print(f"\tPrecision: {precision*100:.2f}%")
    print(f"\tRecall:    {recall*100:.2f}%")
    print(f"\tF-score:   {fscore*100:.2f}%")

if __name__ == "__main__":
    ## parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", 
        required=True,
        help="path to jsonl corpus")
    parser.add_argument("--mode", 
        choices=["train", "annotate"],
        required=True,
        help="choose whether to train or annotate")
    args = parser.parse_args()

    if args.mode == "train":
        docs = data_wrangler.json_to_docs(args.jsonl_path)
        ner_corpus = data_wrangler.docs_to_ner_input(docs)
        train(corpus, output_dir="models")
    elif args.mode == "annotate":
        docs = data_wrangler.json_to_docs(args.jsonl_path)
        doc_tuples = []
        annotated_docs = []

        ## dirty test if we should evaluate metrics or not
        test_file = "test" in args.jsonl_path
        for doc in docs:
            ner_doc = data_wrangler.docs_to_ner_input([doc], incl_entities=False)
            adoc = annotate("models/", ner_doc)

            ner_doc = data_wrangler.ner_output_to_jsonl(adoc)
            annotated_docs.append(ner_doc)
            if not test_file:
                doc_tuples.append([doc.get("entities"), ner_doc.get("entities")])
            else:
                doc_tuples.append([[], ner_doc.get("entities")])
            # print(doc)
        labels = ["cumulative", "date_of_funding", "headquarters_loc",
            "investor", "money_funded", "org_in_focus", "org_url",
            "type_of_funding", "valuation", "year_founded", "all"]
        for ent_type in labels:
            evaluate(doc_tuples, ent_type, only_counts=test_file)
        data_wrangler.json_docs_to_file(annotated_docs, "data/train_NER_out.jsonl")