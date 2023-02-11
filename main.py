import re
import numpy as np
from pprint import pprint

NGRAM_SIZE = 4
ngramDicts = {}


def get_token_list(path: str) -> list:
    """
    Tokenizes the input text file
    :param path: path to the input text file
    :return: list of tokens
    """
    in_text = open(path, "r")
    # lower case it
    in_text = in_text.read().lower()
    # tokenize hashtags
    in_text = re.sub(r"#(\w+)", r"<HASHTAG> ", in_text)
    # tokenize mentions
    in_text = re.sub(r"@(\w+)", r"<MENTION> ", in_text)
    # tokenize urls
    in_text = re.sub(r"http\S+", r"<URL> ", in_text)
    # starting with www
    in_text = re.sub(r"www\S+", r"<URL> ", in_text)

    special_chars = [' ', '*', '!', '?', '.', ',', ';', ':', '(', ')', '[', ']', '{', '}', '/', '\\', '|', '-', '_',
                     '=', '+', '`', '~', '@', '#', '$', '%', '^', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                     '9']

    # pad the special characters with spaces
    for char in special_chars:
        in_text = in_text.replace(char, ' ')

    # pad < and > with spaces
    in_text = in_text.replace('<', ' <')
    in_text = in_text.replace('>', '> ')

    return in_text.split()


def construct_ngram(n: int, token_list: list) -> dict:
    """
    Constructs an n-gram dictionary from the input token list
    :param n: n-gram size
    :param token_list: list of tokens
    :return: n-gram dictionary
    """
    ngram_dict = {}

    for i in range(len(token_list) - n + 1):
        ngram_to_check = token_list[i:i + n]
        cur_dict = ngram_dict
        for j in range(n):
            if ngram_to_check[j] not in cur_dict:
                if j == n - 1:
                    cur_dict[ngram_to_check[j]] = 1
                else:
                    cur_dict[ngram_to_check[j]] = {}
            else:
                if j == n - 1:
                    cur_dict[ngram_to_check[j]] += 1
            cur_dict = cur_dict[ngram_to_check[j]]

    return ngram_dict


def dfs_count(ngram_dict: dict, n: int) -> int:
    """
    Performs a depth first search on the input n-gram dictionary to count the number of n-grams
    :param ngram_dict: n-gram dictionary
    :param n: n-gram size
    :return: number of n-grams
    """
    count = 0
    for key, value in ngram_dict.items():
        if isinstance(value, dict):
            count += dfs_count(value, n)
        else:
            count += 1
    return count


def kneser_ney_smoothing(ngram_dict: dict, h_n: int, n: int, d: float, ngram: list) -> float:
    """
    Performs Kneser-Ney smoothing on the input n-gram dictionary
    :param ngram_dict: n-gram dictionary
    :param h_n: Maximum order of ngram
    :param n: n-gram size
    :param d: discounting factor
    :param ngram: n-gram to be smoothed
    :return: smoothed probability
    """

    if n == h_n:
        count = 0
        # traverse ngram_dict tree to find count
        cur_dict = ngram_dict[n]
        for i in range(n):
            if ngram[i] in cur_dict:
                count = cur_dict[ngram[i]]
            else:
                count = 0
                break
            cur_dict = cur_dict[ngram[i]]

        denomCount = 0
        # traverse ngram_dict tree to find denomCount
        cur_dict = ngram_dict[n - 1]
        for i in range(n - 1):
            if ngram[i] in cur_dict:
                denomCount = cur_dict[ngram[i]]
            else:
                denomCount = 0
                break
            cur_dict = cur_dict[ngram[i]]

        if denomCount == 0:
            return kneser_ney_smoothing(ngram_dict, h_n, n - 1, d, ngram[1:])

        numerator = count
        denominator = denomCount
        return numerator / denominator

    historyCount = 0
    # traverse ngram_dict tree to find historyCount
    cur_dict = ngram_dict[n - 1]
    for i in range(n - 1):
        if ngram[i] in cur_dict:
            historyCount = cur_dict[ngram[i]]
        else:
            historyCount = 0
            break
        cur_dict = cur_dict[ngram[i]]

    lmbda = d / historyCount


tokens = get_token_list("corpus/Pride and Prejudice - Jane Austen.txt")

for n in range(NGRAM_SIZE):
    ngramDicts[n + 1] = construct_ngram(n + 1, tokens)

pprint(ngramDicts[3])
