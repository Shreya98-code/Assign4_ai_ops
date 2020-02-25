"""
Assignment 3
"""
import re
import os
import zipfile

from nl_stuff import TweetTokenizer


class Preprocessor:
    """
    Generates padded embeddings of tweet
    """
    def __init__(self, max_length_tweet = 50, max_length_dictionary=100000):

        """
        Initialize class
        :param max_length_tweet:
        :param max_length_dictionary:
        :param embeddings_dict:
        :param file_path
        """
        self.max_length_tweet = max_length_tweet
        self.max_length_dictionary = max_length_dictionary
                
        file_path = './Preprocessor.zip/word_list.txt'
        archive_path = os.path.abspath(file_path)
        split = archive_path.split(".zip/")
        archive_path = split[0] + ".zip"
        path_inside = split[1]
        archive = zipfile.ZipFile(archive_path, "r")
        self.embeddings = archive.read(path_inside).decode("utf8").split("\n")
        self.embeddings = self.embeddings[:max_length_dictionary]

        self.tokenizer = TweetTokenizer()
        #docs = ['glove_25d_1.txt', 'glove_25d_2.txt', 'glove_25d_3.txt']

        # i = 0

        # self.embeddings_dict = {}

        # for doc in docs:
        #     if i >= max_length_dictionary:
        #         break
        #     with open(doc, 'r') as file:
        #         for line in file:
        #             values = line.split()
        #             word = values[0]
        #             vector = np.asarray(values[1:], "float32")
        #             self.embeddings_dict[word] = vector
        #             i += 1
        #             if i >= max_length_dictionary:
        #                 break

    @staticmethod
    def remove_stop_words(tweet):
        """
        Remove stopwords
        """
        
        # import stopwords
        file_path = './Preprocessor.zip/english'
        archive_path = os.path.abspath(file_path)
        split = archive_path.split(".zip/")
        archive_path = split[0] + ".zip"
        path_inside = split[1]
        archive = zipfile.ZipFile(archive_path, "r")
        stopwords = archive.read(path_inside).decode("utf8").split("\n")

        
        patt = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
        tweet = patt.sub('', tweet)
        return tweet

    def clean_text(self, tweet):
        """
        Cleaning text
        """

        # URL
        tweet = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", '', tweet)

        tweet = tweet.lower()

        # Numbers
        tweet = re.sub(r"[0-9]+", '', tweet)

        # Stopwords
        tweet = self.remove_stop_words(tweet)

        # Removing #
        tweet = re.sub(r"#", '', tweet)

        # Removing handles
        tweet = re.sub(r"@[a-zA-Z0-9]+", '', tweet)

        return tweet

    def tokenize_text(self, tweet):

        """
        Tokenizing tweet
        """
        tokenized = self.tokenizer.tokenize(tweet)
        
        return tokenized

    def replace_token_with_index(self, token_list):
        """
        Replace token with embeddings index
        """
        list_of_index = []
        for token in token_list:
            try:
                token_index = self.embeddings.index(token)
                list_of_index.append(token_index)
            except ValueError:
                embed = self.embeddings.index('<unknown>')
                list_of_index.append(embed)
        return list_of_index

    def pad_sequence(self, index_list):
        """
        Pad tokenized sequence
        """
        length1 = len(index_list)

        if length1 < self.max_length_tweet:
            req_d = self.max_length_tweet - length1
            pad = [self.embeddings.index('<pad>')]
            index_list.extend(pad*req_d)
            token_padded = index_list

        elif length1 == self.max_length_tweet:
            token_padded = index_list

        else:
            token_padded = index_list[:self.max_length_tweet]

        return token_padded

    def tweet_process(self, tweet):

        """
        end to end function
        """

        cleaned = self.clean_text(tweet)
        tokenized = self.tokenize_text(cleaned)
        token_index = self.replace_token_with_index(tokenized)
        padded_index = self.pad_sequence(token_index)

        return padded_index



#sample = "I am doing absolutell fine"
#Twitter = Preprocessor()
#print Twitter.tweet_process(sample)
