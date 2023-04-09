from constants import ParsingParams
from typing import List


class BaseParser:

    col_name = 'base'

    def __init__(self):
        self.tokens: List[str] = []

    def parse_token(self, params: ParsingParams):
        raise NotImplementedError
