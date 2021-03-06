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
import clustering


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


def is_itemize(sentence):
    if len(sentence.feature_list) == 1 and re.match('.|-', sentence.feature_list[0].token):
        return True
    if len(sentence.feature_list) == 2 and re.match('\d+|\(\d+\)', sentence.feature_list[0].token) and re.match('.|-', sentence.feature_list[1].token):
        return True
    return False


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


def produce_summary(compression_rate, sentence_list, n_clusters):
    summary_index = []
    # populate clusters
    for i in range(0, n_clusters):
        cluster = utils.Cluster(i)
        cluster.members = list(filter(lambda s: s.cluster_index == i, sentence_list))
        cluster.summary_members = math.ceil(len(cluster.members) * compression_rate)
        if cluster.summary_members == 0:
            cluster.summary_members = 1
        cluster.members = sorted(cluster.members, key=lambda x: x.avg_distance, reverse=False)
        summary_index.extend(cluster.members[:cluster.summary_members])

    summary_index = list(map(lambda x: x.sentence_number, summary_index))
    summary_index = sorted(summary_index)
    return summary_index


def generate_html(indexes, sentences, output_address):
    final_summary = '<!DOCTYPE html><html><body>'
    for index in range(0, len(sentences)):
        if index in indexes:
            print('\x1b[6;30;42m' + sentences[index].sentence_text + '\x1b[0m')
            final_summary += '<p><mark>' + sentences[index].sentence_text + '</mark></p>'
        else:
            print(sentences[index].sentence_text)
            final_summary += '<p>' + sentences[index].sentence_text + '</p>'
    final_summary += '</body></html>'
    output_file_text = open(output_address, 'w')
    output_file_text.write(final_summary)
    output_file_text.close()


def generte_final_summary(indexes, sentences, output_address):
    final_summary = ''
    i = 0
    for index in indexes:
        if i > 0:
            final_summary += ' '
        final_summary += sentences[index].sentence_text
        i += 1
    output_file_text = open(output_address, 'w')
    output_file_text.write(final_summary)
    output_file_text.close()


def summarize(input_file, compression_rate, output_file):
    preprocessing(input_file)
    filename = input_file.split('.')[0]
    feature_extraction(filename)
    original_sentence_list = get_sentences_representation(filename)
    sentence_list = list(filter(lambda x: is_itemize(x) == False, original_sentence_list))
    representation = list(map(lambda sentence: sentence.representation, sentence_list))
    n_clusters = clustering.elbow_test(representation, n_init=10, max_clusters=20, max_iter=10000)
    sentence_list = clustering.clustering_kmeans(sentence_list, n_clusters)
    summary_index = produce_summary(compression_rate, sentence_list, n_clusters)
    generte_final_summary(summary_index, original_sentence_list, 'OUTPUT/' + output_file)
    generate_html(summary_index, original_sentence_list, 'OUTPUT/' + output_file.split('.')[0] + '.html')


def main(argv):
    input_file = ''
    output_file = ''
    compression_rate = 0.5

    try:
        opts, args = getopt.getopt(argv, "hi:o:c:k:",
                                   ["inputfile=", "outputfile=", "compression_rate="])

    except getopt.GetoptError:
        print("Summarizer.py -i <InputFile> -o <OutputFile> -c <CompressionRate>")
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

    print("Input file is:", input_file)
    print("Output file is:", output_file)
    print("Compression rate is:", compression_rate)

    summarize(input_file, compression_rate, output_file)
    print("---------- Finished ----------")


if __name__ == '__main__':
    main(sys.argv[1:])
