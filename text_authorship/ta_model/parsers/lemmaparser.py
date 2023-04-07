from constants import ParsingParams, TokenType, DELETED
from parsemanager import ParseManager
from parsers.baseparser import BaseParser


@ParseManager.register_parser
class LemmaParser(BaseParser):

    col_name = 'lemmas'

    def __init__(self):
        super().__init__()

    def parse_token(self, params: ParsingParams):
        if TokenType.PUNKT in params.token_type \
                or TokenType.STOPWORD in params.token_type:
            return

        if TokenType.STOPTAG in params.token_type:
            self.tokens.append(DELETED)
            return

        self.tokens.append(params.anls.normal_form)
