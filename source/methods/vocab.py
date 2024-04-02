from nltk.tokenize import RegexpTokenizer
from re import match


class CheckGrammarWithVocabulary:
    vocabulary: dict[str, int] = {}
    tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+')

    def __init__(self):
        # https://github.com/danakt/russian-words
        with open('russian.txt') as f:
            words = f.readlines()
            for word in words:
                self.vocabulary[word[:-1]] = 1

    def check(self, notification: str) -> list[str]:
        tokens = self.tokenizer.tokenize(notification.lower())
        possible_grammar_errors = []
        for token in tokens:
            # only check russian words with vocabulary
            if token not in self.vocabulary and match(r'[а-я]', token):
                possible_grammar_errors.append(token)

        return possible_grammar_errors
