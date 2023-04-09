from constants import ParsingParams, TokenType, DELETED
from parsemanager import ParseManager
from parsers.baseparser import BaseParser


@ParseManager.register_parser
class WordParser(BaseParser):

    col_name = 'text_no_punkt'

    def __init__(self):
        super().__init__()

    def parse_token(self, params: ParsingParams) -> None:
        if TokenType.PUNKT in params.token_type:
            return

        if TokenType.STOPTAG in params.token_type:
            self.tokens.append(DELETED)
            return

        self.tokens.append(params.stripped)
