import sys
import unicodedata
from collections import namedtuple
from enum import Flag, auto

PUNKT = {chr(i)
         for i in range(sys.maxunicode)
         if unicodedata.category(chr(i)).startswith('P')}


class TokenType(Flag):
    NOFLAG = 0
    PUNKT = auto()
    STOPWORD = auto()
    STOPTAG = auto()


ParsingParams = namedtuple('ParsingParams', 'token stripped anls token_type')
DELETED = 'deleted'
STOP_TAGS = [
    'Abbr',
    'Name',
    'Surn',
    'Patr',
    'Geox',
    'Orgn',
    'Trad'
]
