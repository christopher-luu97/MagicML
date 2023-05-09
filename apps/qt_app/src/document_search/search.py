import PyPDF2
import os
import sys
import json
from marqo import Client

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


