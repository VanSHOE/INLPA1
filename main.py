import re
import numpy as np
from pprint import pprint

pridePrej = open("corpus/Pride and Prejudice - Jane Austen.txt", "r")
# tokenize hashtags
pridePrej = re.sub(r"#(\w+)", r"<HASHTAG> ", pridePrej.read())
# tokenize mentions
pridePrej = re.sub(r"@(\w+)", r"<MENTION> ", pridePrej)
# tokenize urls
pridePrej = re.sub(r"http\S+", r"<URL> ", pridePrej)
# starting with www
pridePrej = re.sub(r"www\S+", r"<URL> ", pridePrej)

uniqueChar = set(pridePrej)
specialChars = [' ', '*', '!', '?', '.', ',', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>', '/', '\\', '|', '-',
                '_', '=', '+', '`', '~', '@', '#', '$', '%', '^', '&', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# pad the special characters with spaces
for char in specialChars:
    pridePrej = pridePrej.replace(char, ' ' + char + ' ')

pridePrejTokens = pridePrej.split()
print(pridePrejTokens)
