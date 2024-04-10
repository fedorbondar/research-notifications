import pandas as pd
import numpy as np
from source.rule_based_methods.vocab import CheckGrammarWithVocabulary
from source.rule_based_methods.language_tool import CheckGrammarWithLanguageTool

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

import time
from transformers import BertTokenizer, BertForSequenceClassification
import torch

data = pd.read_csv("../examples/notifications.csv", sep=':')

# get predictions with vocab

start_vocab = time.time()

vocab_checker = CheckGrammarWithVocabulary()

vocab_result = np.array([1 if len(vocab_checker.check(notif)) > 0 else 0 for notif in data["Notification"]])

end_vocab = time.time()

print(f'Precision, recall, accuracy with vocabulary: {precision_score(data["Target"], vocab_result)}, '
      f'{recall_score(data["Target"], vocab_result)}, {accuracy_score(data["Target"], vocab_result)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], vocab_result)}')
print(f'Average time of single check: {(end_vocab - start_vocab) / len(data["Target"])}')

# get predictions with language_check

start_lang = time.time()

lang_checker = CheckGrammarWithLanguageTool()

lang_result = []
for i, notif in enumerate(data["Notification"]):
    if len(lang_checker.check(notif)) > 0:
        lang_result.append(1)
    else:
        lang_result.append(0)

    if i % 100 == 0:
        print(f"Notification {i} proceeded.")
lang_result_np = np.array(lang_result)

end_lang = time.time()

print(f'Precision, recall, accuracy with language_tool: {precision_score(data["Target"], lang_result_np)}, '
      f'{recall_score(data["Target"], lang_result_np)}, {accuracy_score(data["Target"], lang_result_np)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], lang_result_np)}')
print(f'Average time of single check: {(end_lang - start_lang) / len(data["Target"])}')

# get predictions with bert-multilingual

start_bert_multi = time.time()

bert_multi_dir = "neural_network_methods/finetuned_bert_multilingual"
tokenizer = BertTokenizer.from_pretrained(bert_multi_dir)
model_loaded = BertForSequenceClassification.from_pretrained(bert_multi_dir)
bert_multi_result = []
for i, notification in enumerate(data["Notification"]):
    encoded_dict = tokenizer.encode_plus(
        notification,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_id = torch.LongTensor(encoded_dict['input_ids'])
    attention_mask = torch.LongTensor(encoded_dict['attention_mask'])
    with torch.no_grad():
        outputs = model_loaded(input_id, token_type_ids=None, attention_mask=attention_mask)

    index = outputs[0].argmax()
    if index == 1:
        bert_multi_result.append(1)
    else:
        bert_multi_result.append(0)

    if i % 100 == 0:
        print(f"Notification {i} proceeded.")

bert_multi_result_np = np.array(bert_multi_result)

end_bert_multi = time.time()

print(f'Precision, recall, accuracy with bert-multilingual: {precision_score(data["Target"], bert_multi_result_np)}, '
      f'{recall_score(data["Target"], bert_multi_result_np)}, {accuracy_score(data["Target"], bert_multi_result_np)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], bert_multi_result_np)}')
print(f'Average time of single check: {(end_bert_multi - start_bert_multi) / len(data["Target"])}')

# get predictions with rubert

start_rubert = time.time()

rubert_dir = "neural_network_methods/finetuned_rubert"
tokenizer = BertTokenizer.from_pretrained(rubert_dir)
model_loaded = BertForSequenceClassification.from_pretrained(rubert_dir)
rubert_result = []
for i, notification in enumerate(data["Notification"]):
    encoded_dict = tokenizer.encode_plus(
        notification,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_id = torch.LongTensor(encoded_dict['input_ids'])
    attention_mask = torch.LongTensor(encoded_dict['attention_mask'])
    with torch.no_grad():
        outputs = model_loaded(input_id, token_type_ids=None, attention_mask=attention_mask)

    index = outputs[0].argmax()
    if index == 1:
        rubert_result.append(1)
    else:
        rubert_result.append(0)

    if i % 100 == 0:
        print(f"Notification {i} proceeded.")

rubert_result_np = np.array(rubert_result)

end_rubert = time.time()

print(f'Precision, recall, accuracy with rubert: {precision_score(data["Target"], rubert_result_np)}, '
      f'{recall_score(data["Target"], rubert_result_np)}, {accuracy_score(data["Target"], rubert_result_np)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], rubert_result_np)}')
print(f'Average time of single check: {(end_rubert - start_rubert) / len(data["Target"])}')

# results:
#                   precision           recall              accuracy            avg check time (sec)
# vocabulary        0.7995495495495496  0.2805215329909127  0.6052527646129542  0.00011095823646533358
# language_tool     0.9507598784194529  0.6179375740813907  0.7930489731437599  0.1745031308882986
# bert-multilingual 0.9941792782305006  0.6748320821809561  0.8355055292259084  0.06570928135377724
# rubert            0.9927498329740439  0.7649823942348093  0.9083298403249032  0.05890432809400329
