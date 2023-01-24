import os
import re
import json
import glob
import tqdm
import random
from lxml import etree
from typing import List, Dict, Any, Tuple
from bs4 import BeautifulSoup
from dataclasses import dataclass
from argparse import ArgumentParser
from transformers import (BertTokenizerFast,
                          RobertaTokenizerFast,
                          XLMRobertaTokenizer,
                          PreTrainedTokenizer,
                          AutoTokenizer)
from nltk import (sent_tokenize as sent_tokenize_,
                  wordpunct_tokenize as wordpunct_tokenize_)

import sys


# repo root dir; change to your own
#root_dir = "/shared/lyuqing/Zeroshot-Event-Extraction"
root_dir = "/home/evelinamorim/UPorto/zero-shot-participant/zeroshot-pt/Zeroshot-Event-Extraction"
os.chdir(root_dir)

def read_ann_file(path: str,
                  language: str = 'english') -> List[Tuple[str, int, int]]:
    """Reads a ANN text file.
    
    Args:
        path (str): path to the input file.
        language (str): document language. Valid values: "english" or "chinese".

    Returns:
        List[Tuple[str, int, int]]: a list of sentences. Each item in the list
            is a tuple of three elements, sentence text, start offset, and end
            offset.
    """
    data = open(path, 'r', encoding='utf-8').read()
    # TODO: how to get the sentence offset
    # Re-tokenize sentences
    sentences = [s for sent in sentences
                 for s in sent_tokenize(sent, language=language)]

    return sentences



def convert(ann_file: str,
            apf_file: str,
            time_and_val: bool = False,
            language: str = 'english') -> Document:
    """Converts a document.

    Args:
        sgm_file (str): path to a SGM file.
        apf_file (str): path to a APF file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese. Defaults to 'english'.

    Returns:
        Document: a Document instance.
    """
    sentences = read_ann_file(ann_file, language=language)
    doc_id, source, entities, relations, events = read_apf_file(
        apf_file, time_and_val=time_and_val)

    # Reivse sentences
    if doc_id in DOCS_TO_REVISE_SENT:
        sentences = revise_sentences(sentences, doc_id)

    # Process entities, relations, and events
    sentence_entities = process_entities(entities, sentences)
    sentence_relations = process_relation(
        relations, sentence_entities, sentences)
    sentence_events = process_events(events, sentence_entities, sentences)

    # Tokenization
    sentence_tokens = [tokenize(s, ent, evt, language=language) for s, ent, evt
                       in zip(sentences, sentence_entities, sentence_events)]

    # Convert span character offsets to token indices
    sentence_objs = []
    for i, (toks, ents, evts, rels, sent) in enumerate(zip(
            sentence_tokens, sentence_entities, sentence_events,
            sentence_relations, sentences)):
        for entity in ents:
            entity.char_offsets_to_token_offsets(toks)
        for event in evts:
            event.trigger.char_offsets_to_token_offsets(toks)
        sent_id = '{}-{}'.format(doc_id, i)
        sentence_objs.append(Sentence(start=sent[1],
                                      end=sent[2],
                                      text=sent[0],
                                      sent_id=sent_id,
                                      tokens=[t for _, _, t in toks],
                                      entities=ents,
                                      relations=rels,
                                      events=evts))
    return Document(doc_id, sentence_objs)



def convert_batch(input_path: str,
                  output_path: str,
                  time_and_val: bool = False,
                  language: str = 'english'):
    """Converts a batch of documents.

    Args:
        input_path (str): path to the input directory. Usually, it is the path 
            to the LDC2006T06/data/English or LDC2006T06/data/Chinese folder.
        output_path (str): path to the output JSON file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese. Defaults to 'english'.
    """
    if language == 'english':
        sgm_files = glob.glob(os.path.join(
            input_path, '**', 'timex2norm', '*.sgm'))
    elif language == 'chinese':
        sgm_files = glob.glob(os.path.join(
            input_path, '**', 'adj', '*.sgm'))
    elif language == 'portuguese':
        sgm_files = glob.glob(os.path.join(
            input_path, '*.ann'))
    else:
        raise ValueError('Unknown language: {}'.format(language))

    print('Converting the dataset to JSON format')
    print('#ANN files: {}'.format(len(sgm_files)))
    progress = tqdm.tqdm(total=len(sgm_files))

    with open(output_path, 'w', encoding='utf-8') as w:
        for sgm_file in sgm_files:
            progress.update(1)
            apf_file = sgm_file.replace('.ann', '.apf.xml')
            doc = convert(sgm_file, apf_file, time_and_val=time_and_val,
                          language=language)
            w.write(json.dumps(doc.to_dict()) + '\n')
    progress.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input folder')
    parser.add_argument('-o', '--output', help='Path to the output folder')
    parser.add_argument('-b',
                        '--bert',
                        help='BERT model name',
                        default='neuralmind/bert-large-portuguese-cased')
    parser.add_argument('-c',
                        '--bert_cache_dir',
                        help='Path to the BERT cache directory')
    parser.add_argument('-l', '--lang', default='portuguese',
                        help='Document language')
    parser.add_argument('--time_and_val', action='store_true',
                        help='Extracts times and values')
    args = parser.parse_args()
    if args.lang not in ['chinese', 'english','portuguese']:
        raise ValueError('Unsupported language: {}'.format(args.lang))

    input_dir = os.path.join(args.input, args.lang.title())

    model_name = args.bert
    cache_dir = args.bert_cache_dir
    if model_name.startswith('neuralmind/bert-'):
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                cache_dir=cache_dir)
    elif model_name.startswith('bert-'):
            tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                                      cache_dir=cache_dir)
    elif model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name,
                                                         cache_dir=cache_dir)
    elif model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name,
                                                        cache_dir=cache_dir)
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))

    # Convert to doc-level JSON format
    json_path = os.path.join(args.output, '{}.json'.format(args.lang))
    convert_batch(input_dir, json_path, time_and_val=args.time_and_val,
                  language=args.lang)