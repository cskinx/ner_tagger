import json
from collections import namedtuple
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English


class SentenceSplitter:
## takes a document string and returns it is a list of sentences
# https://stackoverflow.com/a/55819791

	def __init__(self):
		self.nlp = English()
		self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

	def split(self, txt):
		doc = self.nlp(txt)
		return [sent.string for sent in doc.sents]

	# def __init__(self):
	# 	self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	# def split(self, txt):
	# 	return self.tokenizer.tokenize(txt)

def json_to_docs(json_path):
	docs = []
	with open(json_path, "r", encoding="utf8") as f:
		for line in f:
			json_doc = json.loads(line)
			docs.append(json_doc)

	return docs

def docs_to_ner_input(docs, incl_entities=True):
	""" reads json file and transforms the given format
	into a more practical format. Outputs a tuple for each
	sentence in format:
	(<sentence_txt>, {"entities": [(<ent_start>, <ent_end>, <ent_type>)]}
	"""
	sent_splitter = SentenceSplitter()

	corpus_sents = []
	for doc in docs:
		txt = doc["text"]
		sents = sent_splitter.split(txt)
		if not incl_entities:
			train_sents = sents
		if incl_entities:
			entities = doc["entities"]

			## offsets so we can see in which sentence an entity belongs
			sent_offsets = []
			offset = 0
			sent_offsets.append(0)
			for sent in sents:
				offset += len(sent)
				sent_offsets.append(offset)

			train_sents = []
			for i, sent in enumerate(sents):
				sent_ents = []

				## current positions for this sentence in the doc
				sent_start = sent_offsets[i]
				sent_end = sent_offsets[i+1]

				## iterate through entities and add to this sentence
				## if they appear in it
				for start, end, ent_type in entities:
					## correct offsets from document to sentence level
					ent_sent_start = start - sent_start
					ent_sent_end = end - sent_start
					# breakpoint()
					if start >= sent_start and start < sent_end:
						## handle entities where sentence splitting went wrong.
						## (e.g. if "Pvt. Ltd" is split into 2 sentences)
						if end <= sent_end:
							sent_ents.append((ent_sent_start, ent_sent_end, ent_type))
							# print(f"1: {sent_ents[-1]}")
						else: 
							sent_ents.append((ent_sent_start, sent_end - sent_start, ent_type))
							# print(f"2: {sent_ents[-1]}")
					elif end >= sent_start and end < sent_end:
						## only part of an entity
						sent_ents.append((0, ent_sent_end, ent_type))
						# print(f"3: {sent_ents[-1]}")

				train_sents.append((sent, {"entities": sent_ents}))

		corpus_sents += train_sents

	# for sent, annotations in corpus_sents:
	# 	for ent in annotations.get("entities"):
	# 		print(sent, " -- ", sent[ent[0]:ent[1]], ent[0], ent[1])
	return corpus_sents

def ner_output_to_jsonl(doc):
	"""basically just transform sentence annotations to 
	document annotations and write it to a jsonl file.
	ner_out is a list of documents, with a list of sentences each."""
	sent_offset = 0
	doc_ents = []
	for sent, annotations in doc:
		ents = annotations["entities"]
		for start, end, ent_type in ents:
			ent_start = sent_offset + start
			ent_end = sent_offset + end
			doc_ents.append([ent_start, ent_end, ent_type])
		sent_offset += len(sent)

	json_doc = {
		"text": "".join(sent for sent,_ in doc),
		"entities": doc_ents
	}
	return json_doc


def json_docs_to_file(docs, out_path):
	with open(out_path, "w", encoding="utf8") as f:
		for doc in docs:
			f.write(json.dumps(doc, ensure_ascii=False))
			f.write("\n")

if __name__ == '__main__':
	json_to_docs("data/train.jsonl")


