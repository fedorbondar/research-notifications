import pandas as pd
import numpy as np
from source.rule_based_methods.vocab import CheckGrammarWithVocabulary
from source.rule_based_methods.language_tool import CheckGrammarWithLanguageTool

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

data = pd.read_csv("../../examples/notifications.csv", sep=':')

vocab_checker = CheckGrammarWithVocabulary()
lang_checker = CheckGrammarWithLanguageTool()

vocab_result = np.array([1 if len(vocab_checker.check(notif)) > 0 else 0 for notif in data["Notification"]])
lang_result = np.array([1 if len(lang_checker.check(notif)) > 0 else 0 for notif in data["Notification"]])

print(f'Precision, recall, accuracy with vocabulary: {precision_score(data["Target"], vocab_result)}, '
      f'{recall_score(data["Target"], vocab_result)}, {accuracy_score(data["Target"], vocab_result)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], vocab_result)}')

print(f'Precision, recall, accuracy with language_tool: {precision_score(data["Target"], lang_result)}, '
      f'{recall_score(data["Target"], lang_result)}, {accuracy_score(data["Target"], lang_result)}')
print(f'Confusion matrix: \n{confusion_matrix(data["Target"], lang_result)}')
