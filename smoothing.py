import re
import numpy as np
from pprint import pprint
import random
import time
from alive_progress import alive_bar

NGRAM_SIZE = 4
ngramDicts = {}


def get_token_list(in_text: str) -> list:
    """
    Tokenizes the input text file
    :param in_text: input text
    :return: list of tokens
    """

    # lower case it
    in_text = in_text.lower()
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


def sentence_tokenizer(fullText: str) -> list:
    """
    Tokenizes the input text file into sentences
    :param fullText: input text
    :return: list of sentences
    """
    # lower case it
    fullText = fullText.lower()
    # tokenize hashtags
    fullText = re.sub(r"#(\w+)", r"<HASHTAG> ", fullText)
    # tokenize mentions
    fullText = re.sub(r"@(\w+)", r"<MENTION> ", fullText)
    # tokenize urls
    fullText = re.sub(r"http\S+", r"<URL> ", fullText)
    # starting with www
    fullText = re.sub(r"www\S+", r"<URL> ", fullText)

    sentenceEnders = ['.', '!', '?']

    # split on sentence enders handling cases such as Mr. etc

    fullText = fullText.replace('mr.', 'mr')
    fullText = fullText.replace('mrs.', 'mrs')
    fullText = fullText.replace('dr.', 'dr')
    fullText = fullText.replace('st.', 'st')
    fullText = fullText.replace('co.', 'co')
    fullText = fullText.replace('inc.', 'inc')
    fullText = fullText.replace('e.g.', 'eg')
    fullText = fullText.replace('i.e.', 'ie')
    fullText = fullText.replace('etc.', 'etc')
    fullText = fullText.replace('vs.', 'vs')
    fullText = fullText.replace('u.s.', 'us')

    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', fullText)

    sentences = [s.replace('\n', ' ') for s in sentences]
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s != '']
    sentences = [get_token_list(s) for s in sentences]
    return sentences


def rem_low_freq(tokens: list, threshold: int) -> list:
    """
    Removes tokens from the input list that occur less than the threshold and replace them with <UNK>
    :param tokens: list of tokens
    :param threshold: threshold
    :return: list of tokens with low frequency tokens removed
    """
    # get the frequency of each token
    freq = {}
    for token in tokens:
        if token in freq:
            freq[token] += 1
        else:
            freq[token] = 1

    # remove tokens with frequency less than threshold
    for token in list(freq.keys()):
        if freq[token] <= threshold:
            del freq[token]

    # replace all tokens not in freq with <UNK>
    for i in range(len(tokens)):
        if tokens[i] not in freq:
            tokens[i] = '<UNK>'

    return tokens


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

    # remove all entities in dictionary tree with count 1 and add <UNK> instead

    return ngram_dict


def dfs_count(ngram_dict: dict) -> int:
    """
    Performs a depth first search on the input n-gram dictionary to count the number of n-grams
    :param ngram_dict: n-gram dictionary
    :return: number of n-grams
    """
    count = 0
    for key, value in ngram_dict.items():
        if isinstance(value, dict):
            count += dfs_count(value)
        else:
            count += 1
    return count


def ngram_count(ngram_dict: dict, ngram: list) -> int:
    """
    Returns the count of the input n-gram
    :param ngram_dict: n-gram dictionary
    :param ngram: n-gram to be counted
    :return: count of the n-gram
    """
    cur_dict = ngram_dict[len(ngram)]
    if n == 1:
        if ngram[0] in cur_dict:
            return cur_dict[ngram[0]]
        else:
            return cur_dict['<UNK>']
    for i in range(len(ngram)):
        if ngram[i] in cur_dict:
            cur_dict = cur_dict[ngram[i]]
        else:
            return 0
    return cur_dict


def kneser_ney_smoothing(ngram_dict: dict, d: float, ngram: list) -> float:
    """
    Performs Kneser-Ney smoothing on the input n-gram dictionary
    :param ngram_dict: n-gram dictionary
    :param d: discounting factor
    :param ngram: n-gram to be smoothed
    :return: smoothed probability
    """
    # replace unknown in ngram with <UNK>
    for i in range(len(ngram)):
        ngram[i] = ngram[i].lower()
        if ngram[i] not in ngram_dict[1]:
            ngram[i] = '<UNK>'

    # print(f'Final ngram: {ngram}')
    if len(ngram) == 1:
        denom = dfs_count(ngram_dict[2])
        # count all bigrams ending with ngram[-1]
        count = 0

        for key, value in ngram_dict[2].items():
            if ngram[-1] in value:
                count += 1

        # print(f'Count: {count}, Denom: {denom}')
        return count / denom

    try:
        first = max(ngram_count(ngram_dict, ngram) - d, 0) / ngram_count(ngram_dict, ngram[:-1])
    except ZeroDivisionError:
        return 0

    try:
        cur_dict = ngram_dict[len(ngram)]
        # len of ngram - 1
        for i in range(len(ngram) - 1):
            cur_dict = cur_dict[ngram[i]]
        second_rhs = len(cur_dict)
    except KeyError:
        second_rhs = 0
    second = d * second_rhs / ngram_count(ngram_dict, ngram[:-1])

    return first + second * kneser_ney_smoothing(ngram_dict, d, ngram[1:])


def witten_bell_smoothing(ngram_dict: dict, ngram: list) -> float:
    """
    Performs Witten-Bell smoothing on the input n-gram dictionary
    :param ngram_dict: n-gram dictionary
    :param ngram: n-gram to be smoothed
    :return: smoothed probability
    """
    # replace unknown in ngram with <UNK>
    for i in range(len(ngram)):
        ngram[i] = ngram[i].lower()
        if ngram[i] not in ngram_dict[1]:
            ngram[i] = '<UNK>'

    if len(ngram) == 1:
        return ngram_count(ngram_dict, ngram) / len(ngram_dict[1])
    try:
        cur_dict = ngram_dict[len(ngram)]
        # len of ngram - 1
        for i in range(len(ngram) - 1):
            cur_dict = cur_dict[ngram[i]]
        lambda_inv_num = len(cur_dict)
    except KeyError:
        lambda_inv_num = 0

    try:
        lambda_inv_num = lambda_inv_num / (lambda_inv_num + ngram_count(ngram_dict, ngram[:-1]))
    except ZeroDivisionError:
        return 0
    lambd = 1 - lambda_inv_num

    first_term = lambd * ngram_count(ngram_dict, ngram) / ngram_count(ngram_dict, ngram[:-1])
    second_term = lambda_inv_num * witten_bell_smoothing(ngram_dict, ngram[1:])

    return first_term + second_term


def sentence_likelihood(ngram_dict: dict, sentence: str, smoothing: str, kneserd=0.75) -> float:
    """
    Calculates the likelihood of the input sentence
    :param ngram_dict: n-gram dictionary
    :param sentence: input sentence
    :param smoothing: smoothing method
    :param kneserd: discounting factor for Kneser-Ney smoothing
    :return: likelihood of the sentence
    """
    tokens = get_token_list(sentence)
    if smoothing == 'wb':
        likelihood = 1
        for i in range(len(tokens) - NGRAM_SIZE + 1):
            likelihood *= witten_bell_smoothing(ngram_dict, tokens[i:i + NGRAM_SIZE])
        return likelihood
    elif smoothing == 'kn':
        likelihood = 1
        for i in range(len(tokens) - NGRAM_SIZE + 1):
            likelihood *= kneser_ney_smoothing(ngram_dict, kneserd, tokens[i:i + NGRAM_SIZE])
        return likelihood


def perplexity(ngram_dict: dict, sentence: str, smoothing: str, kneserd=0.75) -> float:
    """
    Calculates the perplexity of the input sentence
    :param ngram_dict: n-gram dictionary
    :param sentence: input sentence
    :param smoothing: smoothing method
    :param kneserd: discounting factor for Kneser-Ney smoothing
    :return: perplexity of the sentence
    """
    prob = sentence_likelihood(ngram_dict, sentence, smoothing, kneserd)
    # print(sentence, prob)
    prob = max(prob, 1e-10)

    return pow(prob, -1 / len(get_token_list(sentence)))


if __name__ == '__main__':
    # path = "corpus/Pride and Prejudice - Jane Austen.txt"
    path = "corpus/Ulysses - James Joyce.txt"
    in_text = open(path, "r", encoding="utf-8")
    sentences = in_text.readlines()
    random.seed(time.time())
    random_sentences = random.sample(sentences, 1000)
    in_text.close()

    in_text = open(path, "r", encoding="utf-8")
    trainLines = in_text.readlines()
    # remove test lines from training lines

    for sentence in random_sentences:
        trainLines.remove(sentence)

    combined_text = "".join(trainLines)

    for i in range(len(random_sentences)):
        random_sentences[i] = random_sentences[i].strip()
    random_sentences = [sentence for sentence in random_sentences if len(get_token_list(sentence)) >= NGRAM_SIZE]
    tokens = rem_low_freq(get_token_list(combined_text), 1)

    for n in range(NGRAM_SIZE):
        ngramDicts[n + 1] = construct_ngram(n + 1, tokens)

    # get 1000 random sentences from the corpus using random library

    wb_perplexities = []
    with alive_bar(len(random_sentences)) as bar:
        for sentence in random_sentences:
            wb_perplexities.append(perplexity(ngramDicts, sentence, 'wb'))
            bar()

    wb_avg = sum(wb_perplexities) / len(wb_perplexities)
    print(f'Witten-Bell average perplexity: {wb_avg}')
    # calculate perplexity for each sentence using Kneser-Ney smoothing
    kn_perplexities = []
    with alive_bar(len(random_sentences)) as bar:
        for sentence in random_sentences:
            kn_perplexities.append(perplexity(ngramDicts, sentence, 'kn'))
            bar()

    # average
    # print(wb_perplexities)
    # print(kn_perplexities)

    kn_avg = sum(kn_perplexities) / len(kn_perplexities)

    print(f'Kneser-Ney average perplexity: {kn_avg}')