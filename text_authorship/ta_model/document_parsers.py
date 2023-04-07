import sys
import unicodedata

import nltk
from enum import Flag, auto
from collections import namedtuple
from nltk import word_tokenize
from pymorphy2 import MorphAnalyzer

PUNKT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')}


def _strip_str(s: str, chars):
    i, j = 0, len(s)

    while i < j and s[i] in chars:
        i += 1

    while i < j and s[j - 1] in chars:
        j -= 1

    return s[i:j]


nltk.download('stopwords')


STOP_TAGS = [
    'Abbr',
    'Name',
    'Surn',
    'Patr',
    'Geox',
    'Orgn',
    'Trad'
]

DELETED = 'deleted'


class TokenType(Flag):
    NOFLAG = 0
    PUNKT = auto()
    STOPWORD = auto()
    STOPTAG = auto()


ParsingParams = namedtuple('ParsingParams', 'token stripped anls token_type')



class BaseParser:

    col_name = 'base'

    def __init__(self):
        self.tokens = []

    def parse_token(self, params: ParsingParams):
        raise NotImplementedError


class ParseManager:
    parsers = []

    @classmethod
    def register_parser(cls, parser):
        cls.parsers.append(parser)
        return parser

    @classmethod
    def get_col_names(cls):
        cols = [parser.col_name for parser in cls.parsers]
        return cols

    @classmethod
    def parse_text(cls, morph: MorphAnalyzer, sw, text: str):
        tools = [P() for P in cls.parsers]

        for token in word_tokenize(text, language='russian'):
            token_type = TokenType.NOFLAG
            stripped = _strip_str(token, PUNKT)

            if not stripped:
                token_type = token_type | TokenType.PUNKT

            anls = morph.parse(stripped)[0]

            if anls.normal_form in sw:
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


@ParseManager.register_parser
class WordParser(BaseParser):

    col_name = 'text_no_punkt'

    def __init__(self):
        super().__init__()

    def parse_token(self, params: ParsingParams):
        if TokenType.PUNKT in params.token_type:
            return

        if TokenType.STOPTAG in params.token_type:
            self.tokens.append(DELETED)
            return

        self.tokens.append(params.stripped)


@ParseManager.register_parser
class LemmaParser(BaseParser):

    col_name = 'lemmas'

    def __init__(self):
        super().__init__()

    def parse_token(self, params: ParsingParams):
        if TokenType.PUNKT in params.token_type or TokenType.STOPWORD in params.token_type:
            return

        if TokenType.STOPTAG in params.token_type:
            self.tokens.append(DELETED)
            return

        self.tokens.append(params.anls.normal_form)


@ParseManager.register_parser
class TagParser(BaseParser):

    col_name = 'tags'

    def __init__(self):
        super().__init__()

    def parse_token(self, params: ParsingParams):
        if TokenType.PUNKT in params.token_type or TokenType.STOPWORD in params.token_type:
            return

        if TokenType.STOPTAG in params.token_type:
            self.tokens.append(DELETED)
            return

        self.tokens.append(f'{len(params.stripped)}_{params.anls.tag.POS}')


@ParseManager.register_parser
class TokenParser(BaseParser):

    col_name = 'tokens'

    def __init__(self):
        super().__init__()

    def parse_token(self, params: ParsingParams):
        if TokenType.STOPTAG in params.token_type:
            self.tokens.append(DELETED)
            return

        self.tokens.append(params.token)
