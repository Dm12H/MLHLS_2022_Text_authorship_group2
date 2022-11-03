import itertools as it
import os
from functools import partial

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


def chapter_to_str(chapter):
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    text = [para.get_text() for para in soup.find_all('p', class_="p1")]
    return '\n\n'.join(text)


def get_books_as_text_iterator(writer, writers_dir, cutoff=-2):
    book_list = os.listdir(os.path.join(writers_dir, writer))
    full_book_path = partial(os.path.join, writers_dir, writer)
    books = map(lambda book_name: epub.read_epub(full_book_path(book_name)), book_list)
    chapters = [book.get_items_of_type(ebooklib.ITEM_DOCUMENT) for book in books][slice(None, cutoff)]
    return map(chapter_to_str, it.chain.from_iterable(chapters))
