from constants import ParsingParams


class BaseParser:

    col_name = 'base'

    def __init__(self):
        self.tokens = []

    def parse_token(self, params: ParsingParams):
        raise NotImplementedError
