'''
This code was written by Milad Moradi
Institute for Artificial Intelligence and Decision Support
Medical University of Vienna
'''

import sys, getopt
import nltk
nltk.download('punkt')
import json
import json_lines
import random
import math
from numpy import double

#-------------------- CLASSES

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
        
    def eucl_distance(self, second_vector):
        eucl_dist = 0
        i = 0
        for weight in self.representation:
            eucl_dist += (weight - second_vector[i]) ** 2
            i += 1
        
        eucl_dist = math.sqrt(eucl_dist)
        return eucl_dist;
    
    def cosine_similarity(self, second_vector):
        numerator = 0
        term1 = 0
        term2 = 0
        i = 0
        for weight in self.representation:
            numerator += weight * second_vector[i]
            term1 += weight ** 2
            term2 += second_vector[i] ** 2
            i += 1
            
        denominator = math.sqrt(term1) * math.sqrt(term2)
        cosine_sim = numerator / denominator
        return cosine_sim
    
    
    def set_token_list(self, tkn_list):
        self.feature_list = tkn_list
    
    def get_token_list(self):
        return self.feature_list;

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

#-------------------- THE MAIN SUMMARIZATION FUNCTION

def produce_summary(compression_rate, sentence_list, clusters, output_address):
    
    summary_size = math.ceil(len(sentence_list) * compression_rate) + 1
    print('\nSummary size: ', summary_size)

    i = 0
    for cluster in clusters:
        cluster.summary_members = round(summary_size * (len(cluster.members) / len(sentence_list)))
        i += 1
    
    #--------------------- Sentence selection
    
    #For every member of each cluster calculate the average distance to other members within the same cluster 
    for cluster in clusters:
        for sentence_index in cluster.members:
            temp_avg_distance = 0
            denominator = 0
        
            for other_member in cluster.members:
                if sentence_index != other_member:
                    temp_avg_distance += sentence_list[sentence_index].eucl_distance(sentence_list[other_member].representation)
                    denominator += 1
                
            if denominator != 0:
                temp_avg_distance /= denominator
            sentence_list[sentence_index].avg_distance = temp_avg_distance
        
    #------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------


    #Sort members of each cluster
    for cluster in clusters:
        #-----------------Bubble sort
        for i in range(0, len(cluster.members)):
            for j in range(i+1, len(cluster.members)):
                if sentence_list[cluster.members[i]].avg_distance > sentence_list[cluster.members[j]].avg_distance:
                    temp_index = cluster.members[i]
                    cluster.members[i] = cluster.members[j]
                    cluster.members[j] = temp_index
                
    #------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------

    summary_index=[]
    for cluster in clusters:
        for i in range(0, cluster.summary_members):
            summary_index.append(cluster.members[i])
        
        
    
    summary_index.sort()
    print('---------- Sorted selected sentences --------')
    for index in summary_index:
        print(index)


    #----------------------- Producing final summary
    final_summary = ''
    i = 0
    for index in summary_index:
        print(sentence_list[index].sentence_text)
        if i > 0:
            final_summary += ' '
        final_summary += sentence_list[index].sentence_text
        i += 1
        

    '''
    final_output = '<html>'
    final_output += '\n<head>'
    final_output += '\n</head>'
    final_output += '\n<body bgcolor="white">'
    final_output += '\n<a name="1">[1]</a> <a href="#1" id=1>'
    final_output += final_summary
    final_output += '\n</a>'
    final_output += '\n</body>'
    final_output += '\n</html>'
    '''

    output_file_text = open(output_address, 'w')
    output_file_text.write(final_summary)
    output_file_text.close()
    
    return

#-------------------- MAIN BODY OF SUMMARIZER

def main(argv):
    
    input_file = ''
    output_file = ''
    compression_rate = 0.3
    number_of_clusters = 4
    
    try:
        opts, args = getopt.getopt(argv, "hi:o:c:k:", ["inputfile=", "outputfile=", "compression_rate=", "number_of_clusters="])
        
    except getopt.GetoptError:
        print("Summarizer.py -i <InputFile> -o <OutputFile> -c <CompressionRate> -k <NumberOfClusters>")
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print("Summarizer.py -i <InputFile> -o <OutputFile>")
            sys.exit()
        elif opt in ("-i", "--inputfile"):
            input_file = arg
        elif opt in ("-o", "--outputfile"):
            output_file = arg
        elif opt in ("-c", "--compression_rate"):
            compression_rate = double(arg)
        elif opt in ("-k", "--number_of_clusters"):
            number_of_clusters = int(arg)
            
    print("Input file is:", input_file)
    print("Output file is:", output_file)
    print("Compression rate is:", compression_rate)
    print("Number of clusters is:", number_of_clusters)
    
    
    #-------------------- Preprocessing --------------------
    
    input_address = 'INPUT/' + input_file
    
    print('---------- Preprocessing started ----------\n')
    
    opened_file = open(input_address, encoding = "utf8")
    print("-----File opened-----")
    
    input_text = opened_file.read()
    print("-----File read-----")
    
    input_sentences = nltk.sent_tokenize(input_text)
    
    sentence_split_text = ''
    preprocessed_text = ''
    sentence_num = 1
    
    for sentence in input_sentences:
        tokenized_sentence = nltk.word_tokenize(sentence)
        
        if sentence_num > 1:
            sentence_split_text += '\n'
            preprocessed_text += '\n'
        
        sentence_split_text += sentence
        preprocessed_text += str(tokenized_sentence)
        sentence_num += 1
        
    temp_file_address = 'TEMP/temp_input.txt'
    temp_file_token_address = 'TEMP/temp_input_token.txt'
    temp_file = open(temp_file_address, 'w')
    temp_file_token = open(temp_file_token_address, 'w')
    temp_file.write(sentence_split_text)
    temp_file_token.write(preprocessed_text)
    temp_file.close()
    temp_file_token.close()
    
    #-------------------- Feature extraction --------------------
    print("-----Feature extraction-----")
    import subprocess
    #feature_eaxtraction_script = "python BERT/extract_features.py --input_file=TEMP/temp_input.txt --output_file=TEMP/temp_features.jsonl --vocab_file=BERT/vocab.txt --bert_config_file=BERT/bert_config.json --init_checkpoint=BERT/bert_model.ckpt --layers=-1 --max_seq_length=128 --batch_size=8"
    feature_eaxtraction_script = "python BERT/extract_features.py --input_file=TEMP/temp_input.txt --output_file=TEMP/temp_features.jsonl --vocab_file=BERT/vocab.txt --bert_config_file=BERT/bert_config.json --init_checkpoint=BERT/model.ckpt-1000000 --layers=-1 --max_seq_length=128 --batch_size=8"

    subprocess.call(feature_eaxtraction_script, shell=True)
    #exec(open("extract_features.py --input_file=Input.txt --output_file=Example100.jsonl --vocab_file=vocab.txt --bert_config_file=bert_config.json --init_checkpoint=bert_model.ckpt --layers=-1 --max_seq_length=128 --batch_size=8").read())
    
    
    #-------------------- Clustering --------------------
    
    print('---------- Text summarizer started ----------\n')
    
    input_address_text = 'TEMP/temp_input.txt'
    input_address_feature = 'TEMP/temp_features.jsonl'
    output_address = 'OUTPUT/' + output_file


    input_file = open(input_address_text)
    print("-----File opened-----")

    input_text = input_file.read()
    print("-----File read-----")

    input_sentences = nltk.sent_tokenize(input_text)

    sentence_list = []
    sentence_num = 0

    for sentence in input_sentences:
        sentence_num += 1
        temp_sentence = Sentence(sentence_num, sentence)
        sentence_list.append(temp_sentence)

    #--------------------

    sentence_num = 0
    with open(input_address_feature) as input_file:
        for line in json_lines.reader(input_file):
            feature_set = line['features']
        
            for feature in feature_set:
                if feature['token'] in ['[CLS]', '[SEP]']:
                    continue;
            
                temp_feature = Feature(feature['token'])
            
                for layer in feature['layers']:
                    temp_feature.weight_list = layer['values']
                    
                if sentence_num < len(sentence_list):
                    sentence_list[sentence_num].feature_list.append(temp_feature)
                            
            sentence_num += 1
        
    print('-----------------------------------------------')

    #-------------------- Compute a representation for every sentence

    for sentence in sentence_list:
        for weight in sentence.feature_list[0].weight_list:
            sentence.representation.append(0.0)
        
        for feature in sentence.feature_list:
            i = 0
            for weight in feature.weight_list:
                sentence.representation[i] += weight
                i += 1
            
        j = 0
        for weight in sentence.representation:
            sentence.representation[j] /= len(sentence.feature_list)
            j += 1


        
        
    #-------------------- Clustering algorithm
    print('\n---------- Clustering started ----------')

    clusters = []

        
    i = 0
    for sentence in sentence_list:
        temp_cluster = Cluster(i+1)
        temp_cluster.members.append(i)
        clusters.append(temp_cluster)
        i += 1
        
    

    #-------------------- Starting clustering algorithm
            
    end_of_clustering = False

    iteration = 1
    while (end_of_clustering != True):
    
        print('---------- Iteration: ', iteration)
                
        nearest_distance = 10000000000
        nearest_cluster1 = -1
        nearest_cluster2 = -1
                
        for i in range(0, len(clusters)):
            for j in range(0, len(clusters)):
                        
                if i < j:
                            
                    #----------Compute the distance between two clusters
                    denominator = 0
                    temp_distance = 0
                    for index1 in clusters[i].members:
                        for index2 in clusters[j].members:
                                    
                            temp_distance += sentence_list[index1].eucl_distance(sentence_list[index2].representation)
                            denominator += 1
                                    
                    if denominator != 0:
                        temp_distance /= denominator
                            
                    if temp_distance < nearest_distance:
                        nearest_distance = temp_distance
                        nearest_cluster1 = i
                        nearest_cluster2 = j
                            
        #----------Merge two nearest clusters
        clusters[nearest_cluster1].members = clusters[nearest_cluster1].members + clusters[nearest_cluster2].members
        clusters.remove(clusters[nearest_cluster2])
        print(nearest_cluster1, 'and', nearest_cluster2, 'merged')
                
        print('Number of clusters: ', str(len(clusters)))
                
        iteration += 1
        if len(clusters) <= number_of_clusters:
            end_of_clustering = True
            output_address = 'OUTPUT/' + output_file
            produce_summary(compression_rate, sentence_list, clusters, output_address)
                    
        #-------------------- End of clustering algorithm
    
    
    
    
    
    print("\n")
    print("---------- Finished ----------")



if __name__ == '__main__':
    main(sys.argv[1:])