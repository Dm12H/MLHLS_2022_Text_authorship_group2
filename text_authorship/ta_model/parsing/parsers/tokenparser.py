from constants import ParsingParams, TokenType, DELETED
from parsing.parsemanager import ParseManager
from parsing.parsers.baseparser import BaseParser


@ParseManager.register_parser
class TokenParser(BaseParser):

    col_name = 'tokens'

    def __init__(self):
        super().__init__()

    def parse_token(self, params: ParsingParams) -> None:
        if TokenType.STOPTAG in params.token_type:
            self.tokens.append(DELETED)
            return

        self.tokens.append(params.token)
