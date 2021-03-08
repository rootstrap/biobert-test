# biobert-test

This code is for medical text summarization by extractive strategy, using bert with a pre-trained model. 

The original code is from [BERT-based-Summ](https://github.com/BioTextSumm/BERT-based-Summ). The associated paper for 
the code is [Deep contextualized embeddings for quantifying the informative content in biomedical text summarization
](https://www.researchgate.net/publication/336272974_Deep_contextualized_embeddings_for_quantifying_the_informative_content_in_biomedical_text_summarization)

The objective of the paper is to show how contextualized  embeddings produced  by  a  deep bidirectional language  model 
 can  be  utilized  to  quantify  the  informative content of sentences in biomedical text summarization 

(5) (PDF) Deep contextualized embeddings for quantifying the informative content in biomedical text summarization. Available from: https://www.researchgate.net/publication/336272974_Deep_contextualized_embeddings_for_quantifying_the_informative_content_in_biomedical_text_summarization [accessed Mar 05 2021].
ontext-sensitive embeddings for biomedical text summarization

Some modifications to the [Summarizer.py](Summarizer.py) script have been done. 

Due to compatibility issues it works with python3.6.   

## Setup Script 
The [setup.sh](setup.sh) script is created to automatically download all the dependencies for the project. It follows these steps:  
1. Clone [bert](https://github.com/google-research/bert.git) repository 
2. Download pre-trained biobert_v1.1_pubmed model 
3. Install python dependencies in a virtualenv 
4. Creates directories INPUT OUTPUT TEMP

```bash
    sh setup.sh
```

## Run the Summarizer script 
1. Locate the input file at INPUT directory
2. Run the script: 

Four parameters must be specified when running the script:    
- INPUT_FILE_NAME(-i) is the name of input file already copied to the INPUT directory.
- OUTPUT_FILE_NAME(-o) is the name of output file containing the summary that will be created in the OUTPUT directory.
- COMPRESSION_RATE(-c) specifies the size of summary and takes a value in the range (0, 1).
- NUMBER_OF_CLUSTERS(-k) specifies the number of final clusters in the clustering step.

```bash
     python3.6 Summarizer.py -i 14.txt -o 14.txt -c 0.5 -k 4
```
