import math

class Feature:

    def __init__(self, word):
        self.token = word
        self.weight_list = []


class Sentence:

    def __init__(self, num, txt):
        self.sentence_number = num
        self.sentence_text = txt
        self.avg_distance = 0
        self.feature_list = []
        self.representation = []
        self.cluster_index = 0

    def set_token_list(self, tkn_list):
        self.feature_list = tkn_list

    def get_token_list(self):
        return self.feature_list


class Cluster:

    def __init__(self, num):
        self.cluster_number = num
        self.mean = []
        self.members = []
        self.summary_members = 0

    def add_member(self, sentence_index):
        self.members.append(sentence_index)

    def remove_member(self, sentence_index):
        self.members.remove(sentence_index)

