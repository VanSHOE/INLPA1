import re
import numpy as np
from pprint import pprint


def get_token_list(path: str) -> list:
    in_text = open(path, "r")
    # tokenize hashtags
    in_text = re.sub(r"#(\w+)", r"<HASHTAG> ", in_text.read())
    # tokenize mentions
    in_text = re.sub(r"@(\w+)", r"<MENTION> ", in_text)
    # tokenize urls
    in_text = re.sub(r"http\S+", r"<URL> ", in_text)
    # starting with www
    in_text = re.sub(r"www\S+", r"<URL> ", in_text)

    special_chars = [' ', '*', '!', '?', '.', ',', ';', ':', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '-',
                     '_', '=', '+', '`', '~', '@', '#', '$', '%', '^', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                     '9']

    # pad the special characters with spaces
    for char in special_chars:
        in_text = in_text.replace(char, ' ')

    # pad < and > with spaces
    in_text = in_text.replace('<', ' <')
    in_text = in_text.replace('>', '> ')

    return in_text.split()


print(get_token_list("corpus/Pride and Prejudice - Jane Austen.txt"))
