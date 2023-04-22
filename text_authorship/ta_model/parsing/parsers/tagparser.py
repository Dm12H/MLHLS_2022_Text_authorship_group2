from constants import ParsingParams, TokenType, DELETED
from parsing.parsemanager import ParseManager
from parsing.parsers.baseparser import BaseParser


@ParseManager.register_parser
class TagParser(BaseParser):

    col_name = 'tags'

    def __init__(self):
        super().__init__()

    def parse_token(self, params: ParsingParams) -> None:
        if TokenType.PUNKT in params.token_type \
                or TokenType.STOPWORD in params.token_type:
            return

        if TokenType.STOPTAG in params.token_type:
            self.tokens.append(DELETED)
            return

        self.tokens.append(f'{len(params.stripped)}_{params.anls.tag.POS}')
