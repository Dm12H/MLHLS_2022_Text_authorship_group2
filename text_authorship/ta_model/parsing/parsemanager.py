from nltk import word_tokenize
from pymorphy2 import MorphAnalyzer
from parsing.parsers.baseparser import BaseParser
from typing import List, Set, Type

from constants import PUNKT, TokenType, ParsingParams, STOP_TAGS
from utils import _strip_str


class ParseManager:
    parsers: List[Type[BaseParser]] = []

    @classmethod
    def register_parser(cls, parser: BaseParser):
        cls.parsers.append(parser)
        return parser

    @classmethod
    def get_col_names(cls):
        cols = [parser.col_name for parser in cls.parsers]
        return cols

    @classmethod
    def parse_text(cls, morph: MorphAnalyzer, stopwords: Set[str], text: str):
        tools = [ParserCls() for ParserCls in cls.parsers]

        for token in word_tokenize(text, language='russian'):
            token_type = TokenType.NOFLAG
            stripped = _strip_str(token, PUNKT)

            if not stripped:
                token_type = token_type | TokenType.PUNKT

            anls = morph.parse(stripped)[0]

            if anls.normal_form in stopwords:
                token_type = token_type | TokenType.STOPWORD

            if any([tag in anls.tag for tag in STOP_TAGS]):
                token_type = token_type | TokenType.STOPTAG

            params = ParsingParams(token=token,
                                   stripped=stripped,
                                   anls=anls,
                                   token_type=token_type)

            for p in tools:
                p.parse_token(params)

        result = [' '.join(p.tokens) for p in tools]
        return result
