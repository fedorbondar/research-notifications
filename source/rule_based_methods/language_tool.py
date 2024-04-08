# https://github.com/jxmorris12/language_tool_python
from language_tool_python import LanguageTool


class CheckGrammarWithLanguageTool:
    language_tool: LanguageTool = LanguageTool('ru-RU')

    def check(self, notification: str) -> list[tuple[str, str]]:
        matches = self.language_tool.check(notification)
        # if error with no replacements, return '-'
        messages_and_replacements = [(match.message, '-') if not match.replacements else
                                     (match.message, match.replacements[0]) for match in matches]
        return messages_and_replacements
