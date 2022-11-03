import hashlib
import itertools as it
import os
from functools import partial
from configparser import ConfigParser

import ebooklib
import numpy as np
from bs4 import BeautifulSoup
from ebooklib import epub


def chapter_to_str(chapter):
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    text = [para.get_text() for para in soup.find_all('p', class_="p1")]
    return text


def get_books_as_text_iterator(writer, writers_dir, cutoff=-2):
    book_list = os.listdir(os.path.join(writers_dir, writer))
    full_book_path = partial(os.path.join, writers_dir, writer)
    books = map(lambda book_name: epub.read_epub(full_book_path(book_name)), book_list)
    chapters = [book.get_items_of_type(ebooklib.ITEM_DOCUMENT) for book in books][slice(None, cutoff)]
    return map(chapter_to_str, it.chain.from_iterable(chapters))


def get_dir_hash(directory):
    sha1 = hashlib.sha1()
    for root, folders, files in os.walk(directory):
        for folder in folders:
            for file in files:
                path = os.path.join(root, folder, file)
                with open(path, "rb") as f:
                    sha1.update(hashlib.sha1(f.read()).hexdigest())
    return sha1.hexdigest()


def hash_results(func, data_dir, writer, *args, out_path=None, **kwargs):
    # TODO replace Configparser with json/xml/yaml
    writers_hashfile = ConfigParser()
    hash_file = func.__name__+"_hashes.ini"
    writers_hashfile.read(os.path.join(out_path, hash_file))
    path = os.path.join(data_dir, writer)
    try:
        hash_info = writers_hashfile[writer]
        hash_existing = hash_info["hash"]
        hash_computed = get_dir_hash(path)
        if hash_existing == hash_computed:
            feature_vector = np.load(os.path.join(out_path, hash_info["path"]))
            return feature_vector
    except KeyError:
        hash_computed = get_dir_hash(path)
    result = func(data_dir, writer, *args, **kwargs)
    fname_to_save = os.path.join(out_path, f"{func.__name__}_{writer.lower()}.npy")
    np.save(file=fname_to_save, arr=result)
    writers_hashfile[writer] = {"hash": hash_computed, "path": fname_to_save}
    with open(os.path.join(out_path, hash_file), "w") as f:
        writers_hashfile.write(f)
