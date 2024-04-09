import pandas as pd
import numpy as np
from source.rule_based_methods.vocab import CheckGrammarWithVocabulary
from source.rule_based_methods.language_tool import CheckGrammarWithLanguageTool

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from transformers import BertTokenizer, BertForSequenceClassification
import torch

data = pd.read_csv("../examples/notifications.csv", sep=':')

vocab_checker = CheckGrammarWithVocabulary()
lang_checker = CheckGrammarWithLanguageTool()

# get predictions with vocab
vocab_result = np.array([1 if len(vocab_checker.check(notif)) > 0 else 0 for notif in data["Notification"]])

print(f'Precision, recall, accuracy with vocabulary: {precision_score(data["Target"], vocab_result)}, '
      f'{recall_score(data["Target"], vocab_result)}, {accuracy_score(data["Target"], vocab_result)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], vocab_result)}')

# get predictions with language_check
lang_result = []
for i, notif in enumerate(data["Notification"]):
    if len(lang_checker.check(notif)) > 0:
        lang_result.append(1)
    else:
        lang_result.append(0)

    if i % 100 == 0:
        print(f"Notification {i} proceeded.")
lang_result_np = np.array(lang_result)

print(f'Precision, recall, accuracy with language_tool: {precision_score(data["Target"], lang_result_np)}, '
      f'{recall_score(data["Target"], lang_result_np)}, {accuracy_score(data["Target"], lang_result_np)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], lang_result_np)}')

# get predictions with bert-multilingual

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

print(f'Precision, recall, accuracy with bert-multilingual: {precision_score(data["Target"], bert_multi_result_np)}, '
      f'{recall_score(data["Target"], bert_multi_result_np)}, {accuracy_score(data["Target"], bert_multi_result_np)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], bert_multi_result_np)}')

# get predictions with rubert

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

print(f'Precision, recall, accuracy with rubert: {precision_score(data["Target"], rubert_result_np)}, '
      f'{recall_score(data["Target"], rubert_result_np)}, {accuracy_score(data["Target"], rubert_result_np)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], rubert_result_np)}')

# results:
#                   precision           recall              accuracy
# vocabulary        0.7995495495495496  0.2805215329909127  0.6052527646129542
# language_tool     0.9507598784194529  0.6179375740813907  0.7930489731437599
# bert-multilingual 0.9941792782305006  0.6748320821809561  0.8355055292259084
# rubert            0.9919871794871795  0.7337020940339787  0.8639415481832543
