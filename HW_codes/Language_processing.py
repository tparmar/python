from collections import Counter
import os
import pandas as pd
import matplotlib.pyplot as plt
stats = pd.DataFrame(columns=("language", "author, ""title","length", "unique"))
text = "This is my text. We're keeping this text short to keep things manageable."
title_num = 1
def count_words_fast(text):
    text = text.lower()
    skips = ['.', ',', '?', ';', '!', "'", '"' ]
    for ch in skips:
        text = text.replace(ch, "")
    word_counts = Counter(text.split(' '))
    return word_counts
def count_words(text):
    word_counts = {}
    for word in text.split(" "):
        if word in word_counts:
            word_counts[word] +=1
        else:
            word_counts[word] = 1
def read_book(title_path):
    """
    This function will read a book a return it in a string.
    """
    with open(title_path, "r", encoding = "utf8") as current_file:
        text = current_file.read()
        text = text.replace("\n", "").replace("\r", "")
    return text
def word_stats(word_counts):
    """
    Return number of unique words of word frequencies.
    """
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)
book_dir = "./Books"
for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + "/" + language):
        for title in os.listdir(book_dir + "/" + language+ "/"+ author):
            inputfile = book_dir + "/" + language+ "/"+ author + "/" + title
            print(inputfile)
            text = read_book(inputfile)
            (num_unique, counts) = word_stats(count_words(text))
            stats.loc[title_num] = language, author.captalize(), title.replace(".txt", ""), sum(counts), num_unique
            title_num += 1
