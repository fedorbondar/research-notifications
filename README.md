# research-notifications

Project with tools for notification text correction written in Python 3.9.

## Context

The purpose of this project is to optimize process of testing correctness of notifications, 
as comparison of notification with its exact pattern "as is" takes too much time to prepare
and support in ongoing test updates. To solve this problem a different approach of testing 
was proposed: 

1) Check correctness of data fields (dates, amounts, names of products, etc.)
2) Search for typos in notification (evaluation errors, typos in words, word duplication, etc.)
3) Check notification for grammatical and syntactical faults

Some examples of typos and grammar faults:

* Evaluation error
  ```text
  Поступил платёж на сумму ${(amount)|numberformat("0.##")} руб. 
  ```
* Duplication error
  ```text
  В ночь 24 октября спишется спишется 550 руб. по тарифу "Всё".
  ```
* Typo (duplicated letters)
  ```text
  Ваш баааланс: 600 руб. Спасибо, что пользуетесь услугами.
  ```
* Grammar or punctuation fault
  ```text
  Кстати пополнить баланс легк на сайте или в ЛК.
  ```

In order to implement this approach research was held, in which Classic NLP, Rule-based and 
Neural Network methods of text processing were analysed and compared by accuracy and performance 
on notification dataset. 

## Features

Before testing text correctness methods, a dataset of notifications was created. Although less than 500 
notifications with errors were provided, another 2000 examples were generated to balance correct ones.
So the final dataset consists of 5000+ unique notifications and labels (1 if notification is incorrect and 
0 otherwise). Formally, further research is an attempt to solve classification problem with different NLP 
approaches.

Full dataset can not be shown, because it is corporate property, but few examples of real training data 
samples can be found [here](https://github.com/fedorbondar/research-notifications/blob/main/examples/notifications.csv).

In this project 4 methods of checking text correctness were tested:

* [vocab](https://github.com/fedorbondar/research-notifications/blob/main/source/rule_based_methods/vocab.py)

Simple approach for a baseline. Basically, just a vocabulary of 1M+ Russian words and forms. 
Predicts notification as error if at least one of its words is outside vocabulary. 

* [language_tool](https://github.com/fedorbondar/research-notifications/blob/main/source/rule_based_methods/language_tool.py)

Classifier based on well-known rule-based grammar checking tool 
[LanguageTool](https://github.com/jxmorris12/language_tool_python), which is used as spell-checker in OpenOffice.
Predicts notification as error if at least one suggestion to replace anything occurs. 

* [bert-multilingual](https://github.com/fedorbondar/research-notifications/blob/main/source/neural_network_methods/training_bert_multilingual.ipynb)

A `bert-base-multilingual-uncased` model from [HuggingFace](https://huggingface.co/google-bert/bert-base-multilingual-uncased)
by Google fine-tuned on notifications dataset.

* [rubert](https://github.com/fedorbondar/research-notifications/blob/main/source/neural_network_methods/training_rubert.ipynb)

Another `transformers` model by [DeepPavlov](https://huggingface.co/DeepPavlov/rubert-base-cased) fine-tuned on 
notifications. RuBERT is based on BERT-multilingual, but it was additionally trained on Russian texts.

## Performance

As a result of methods accuracy & performance comparison on notifications dataset here is a table a metrics:

|                   | precision | recall | accuracy | MCC  | avg check time (sec) |
|-------------------|-----------|--------|----------|------|----------------------|
| vocabulary        | 0.80      | 0.28   | 0.61     | -    | 0.0001               |
| language_tool     | 0.95      | 0.62   | 0.79     | -    | 0.17                 |
| bert-multilingual | 0.99      | 0.67   | 0.83     | 0.70 | 0.07                 |
| rubert            | 0.99      | 0.76   | 0.90     | 0.84 | 0.06                 |

where MCC stands for [Matthew's Correlation Coefficient](https://en.wikipedia.org/wiki/Phi_coefficient).

Note: value of precision metric, which is nearly 1, stands that almost none false positive samples. 
It means that almost none correct notifications were classified as in-correct.

## Requirements

See [here](https://github.com/fedorbondar/research-keywords/blob/main/requirements.txt).

Best way to get all required packages at once is running the following line:

```shell
pip install -r requirements.txt
```

After installation of `nltk` you might also need to execute the 
following in python:
```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
```

## References

* "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation" by N. Reimers, I. Gurevych
  ([source](https://arxiv.org/abs/2004.09813))
* "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by J. Devlin, M.-W. Chang, K. Lee,
  K. Toutanova ([source](https://arxiv.org/pdf/1810.04805.pdf))

## License

[MIT License](https://github.com/fedorbondar/research-notifications/blob/main/LICENSE)