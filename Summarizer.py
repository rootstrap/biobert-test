import sys, getopt
import nltk

nltk.download('punkt')
import json_lines
import math
import numpy as np
import subprocess
import xml.etree.ElementTree as ET
import re
import utils


def produce_summary(compression_rate, sentence_list, clusters, output_address):
    summary_size = math.ceil(len(sentence_list) * compression_rate) + 1
    print('\nSummary size: ', summary_size)

    i = 0
    for cluster in clusters:
        cluster.summary_members = round(summary_size * (len(cluster.members) / len(sentence_list)))
        i += 1

    # --------------------- Sentence selection

    # For every member of each cluster calculate the average distance to other members within the same cluster
    for cluster in clusters:
        for sentence_index in cluster.members:
            temp_avg_distance = 0
            denominator = 0

            for other_member in cluster.members:
                if sentence_index != other_member:
                    temp_avg_distance += np.linalg.norm(np.array(sentence_list[sentence_index].representation) \
                                                    - np.array(sentence_list[other_member].representation))
                    denominator += 1

            if denominator != 0:
                temp_avg_distance /= denominator
            sentence_list[sentence_index].avg_distance = temp_avg_distance

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Sort members of each cluster
    for cluster in clusters:
        # -----------------Bubble sort
        for i in range(0, len(cluster.members)):
            for j in range(i + 1, len(cluster.members)):
                if sentence_list[cluster.members[i]].avg_distance > sentence_list[cluster.members[j]].avg_distance:
                    temp_index = cluster.members[i]
                    cluster.members[i] = cluster.members[j]
                    cluster.members[j] = temp_index

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    summary_index = []
    for cluster in clusters:
        for i in range(0, cluster.summary_members):
            summary_index.append(cluster.members[i])

    summary_index.sort()
    print('---------- Sorted selected sentences --------')
    for index in summary_index:
        print(index)

    # ----------------------- Producing final summary
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


def clean_content(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    children = root.getchildren()
    if len(children) == 0:
        print('Error: Wrong format for file {}'.format(filename))
        return None
    # replace \n - to put together the sentences
    content = re.sub(r'\n', ' ', children[0].text)
    content = re.sub(r'\*+', ' ', content)
    content = " ".join(content.split())
    return content


def preprocessing(input_file):
    print('---------- Preprocessing started ----------')
    input_address = 'INPUT/' + input_file
    input_text = clean_content(input_address)
    input_sentences = nltk.sent_tokenize(input_text)

    input_sentences = list(filter(lambda sentence: len(sentence) > 0, input_sentences))
    tokenized_sentence = list(map(lambda sentence: str(nltk.word_tokenize(sentence)), input_sentences))

    filename = input_file.split('.')[0]
    temp_file_address = 'TEMP/temp_input_{}.txt'.format(filename)
    temp_file_token_address = 'TEMP/temp_input_token_{}.txt'.format(filename)
    temp_file = open(temp_file_address, 'w')
    temp_file_token = open(temp_file_token_address, 'w')
    temp_file.write('\n'.join(input_sentences))
    temp_file_token.write('\n'.join(tokenized_sentence))
    temp_file.close()
    temp_file_token.close()


def feature_extraction(filename):
    print("-----Feature extraction-----")
    feature_eaxtraction_script = "python BERT/extract_features.py --input_file=TEMP/temp_input_{}.txt \
                                --output_file=TEMP/temp_features_{}.jsonl --vocab_file=BERT/vocab.txt \
                                --bert_config_file=BERT/bert_config.json --init_checkpoint=BERT/model.ckpt-1000000 \
                                --layers=-1 --max_seq_length=128 --batch_size=8".format(filename, filename)
    subprocess.call(feature_eaxtraction_script, shell=True)


def get_sentences_representation(filename):
    input_address_text = 'TEMP/temp_input_{}.txt'.format(filename)
    input_address_feature = 'TEMP/temp_features_{}.jsonl'.format(filename)
    input_file = open(input_address_text)
    input_text = input_file.read()
    input_sentences = nltk.sent_tokenize(input_text)

    sentence_list = list(map(lambda s: utils.Sentence(s[0], s[1]), enumerate(input_sentences)))

    sentence_num = 0
    with open(input_address_feature) as input_file:
        for line in json_lines.reader(input_file):
            feature_set = line['features']

            for feature in feature_set:
                if feature['token'] in ['[CLS]', '[SEP]']:
                    continue;

                temp_feature = utils.Feature(feature['token'])

                for layer in feature['layers']:
                    # since it is layers=-1 it is only 1 layer
                    temp_feature.weight_list.extend(layer['values'])

                sentence_list[sentence_num].feature_list.append(temp_feature)

            sentence_num += 1
    # -------------------- Compute a representation for every sentence

    for sentence in sentence_list:

        sentence.representation = [0.0] * len(sentence.feature_list[0].weight_list)

        for feature in sentence.feature_list:
            i = 0
            for weight in feature.weight_list:
                sentence.representation[i] += weight
                i += 1

        sentence.representation = [r / 2 for r in sentence.representation]
    return sentence_list


def clustering(filename, number_of_clusters, compression_rate, output_file):
    print('---------- Clustering started ----------')
    sentence_list = get_sentences_representation(filename)
    clusters = []

    for i in range(0, len(sentence_list)):
        temp_cluster = utils.Cluster(i + 1)
        temp_cluster.members.append(i)
        clusters.append(temp_cluster)

    end_of_clustering = False

    iteration = 1
    while not end_of_clustering:
        nearest_distance = 10000000000
        nearest_cluster1 = -1
        nearest_cluster2 = -1

        for i in range(0, len(clusters)):
            for j in range(0, len(clusters)):

                if i < j:

                    # ----------Compute the distance between two clusters
                    denominator = 0
                    temp_distance = 0
                    for index1 in clusters[i].members:
                        for index2 in clusters[j].members:
                            temp_distance += np.linalg.norm(np.array(sentence_list[index1].representation) \
                                                            - np.array(sentence_list[index2].representation))
                            denominator += 1

                    if denominator != 0:
                        temp_distance /= denominator

                    if temp_distance < nearest_distance:
                        nearest_distance = temp_distance
                        nearest_cluster1 = i
                        nearest_cluster2 = j

        # ----------Merge two nearest clusters
        clusters[nearest_cluster1].members = clusters[nearest_cluster1].members + clusters[nearest_cluster2].members
        clusters.remove(clusters[nearest_cluster2])

        print('Number of clusters: ', str(len(clusters)), 'Iteration:', iteration)

        iteration += 1
        if len(clusters) <= number_of_clusters:
            end_of_clustering = True
            output_address = 'OUTPUT/' + output_file
            produce_summary(compression_rate, sentence_list, clusters, output_address)

        # -------------------- End of clustering algorithm


def summarize(input_file, number_of_clusters, compression_rate, output_file):
    preprocessing(input_file)
    filename = input_file.split('.')[0]
    feature_extraction(filename)
    clustering(filename, number_of_clusters, compression_rate, output_file)


def main(argv):
    input_file = ''
    output_file = ''
    compression_rate = 0.3
    number_of_clusters = 4

    try:
        opts, args = getopt.getopt(argv, "hi:o:c:k:",
                                   ["inputfile=", "outputfile=", "compression_rate=", "number_of_clusters="])

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
            compression_rate = np.double(arg)
        elif opt in ("-k", "--number_of_clusters"):
            number_of_clusters = int(arg)

    print("Input file is:", input_file)
    print("Output file is:", output_file)
    print("Compression rate is:", compression_rate)
    print("Number of clusters is:", number_of_clusters)

    summarize(input_file, number_of_clusters, compression_rate, output_file)
    print("\n")
    print("---------- Finished ----------")


if __name__ == '__main__':
    main(sys.argv[1:])
