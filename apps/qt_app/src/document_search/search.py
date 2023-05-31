import PyPDF2
import os
import sys
import json

file_path = os.path.join(os.getcwd(), "dsr.pdf")
file_obj = open(file_path, 'rb')
pdf = PyPDF2.PdfReader(file_obj)

document_list = []
for i in range(len(pdf.pages)):
    document = {}
    page_ob = pdf.pages[i]
    text = page_ob.extract_text()
    document['text'] = text
    document['source'] = i
    document_list.append(document)

from cherche import data, retrieve, rank, qa
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Retrieve on fields title and article
retriever = retrieve.TfIdf(key="source", on=["text"], documents=document_list, k=30)

# Rank on fields title and article
ranker = rank.Encoder(
    key = "source",
    on = ["text"],
    encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
    k = 3,
)

question_answering = qa.QA(
    model = pipeline("question-answering",
         model = "deepset/tinyroberta-squad2",
         tokenizer = "deepset/tinyroberta-squad2"
    ),
    on = "text")

# Pipeline creation
search = retriever + ranker + document_list + question_answering

search.add(documents=document_list)
# Search documents for 3 queries.
answers = search(
    q = [
        "What is Navy doing?"
    ]
)

import pdb
pdb.set_trace()

# List of dicts
# documents = data.load_towns()

# # Retrieve on fields title and article
# retriever = retrieve.TfIdf(key="id", on=["title", "article"], documents=documents, k=30)

# import pdb
# pdb.set_trace()

# # Rank on fields title and article
# ranker = rank.Encoder(
#     key = "id",
#     on = ["title", "article"],
#     encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode,
#     k = 3,
# )

# # Pipeline creation
# search = retriever + ranker

# search.add(documents=documents)

# # Search documents for 3 queries.
# search(["Bordeaux", "Paris", "Toulouse"])

