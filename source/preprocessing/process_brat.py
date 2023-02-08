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
import spacy

from text2story.readers import read_brat


# repo root dir; change to your own
#root_dir = "/shared/lyuqing/Zeroshot-Event-Extraction"
root_dir = "/home/evelinamorim/UPorto/zero-shot-participant/zeroshot-pt/Zeroshot-Event-Extraction"
os.chdir(root_dir)

def read_txt_file(path: str,
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
    sentences = []
    if language == "portuguese":
        if not(spacy.util.is_package('pt_core_news_lg')):
            spacy.cli.download('pt_core_news_lg')
        nlp = spacy.load('pt_core_news_lg')

        doc = nlp(data)
        offset_start = 0

        for sent in doc.sents:
            offset_end = offset_start + len(sent)
            sentences.append([sent.text, offset_start, offset_end])
            offset_start = offset_end + 1


    return sentences


@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace('\n', ' ')

    def char_offsets_to_token_offsets(self, tokens: List[Tuple[int, int, str]]):
        """Converts self.start and self.end from character offsets to token
        offsets.

        Args:
            tokens (List[int, int, str]): a list of token tuples. Each item in
                the list is a triple (start_offset, end_offset, text).
        """
        start_ = end_ = -1
        for i, (s, e, _) in enumerate(tokens):
            if s == self.start:
                start_ = i
            if e == self.end:
                end_ = i + 1
        if start_ == -1 or end_ == -1 or start_ > end_:
            raise ValueError('Failed to update offsets for {}-{}:{} in {}'.format(
                self.start, self.end, self.text, tokens))
        self.start, self.end = start_, end_

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            dict: a dict of instance variables.
        """
        return {
            'text': recover_escape(self.text),
            'start': self.start,
            'end': self.end
        }

    def remove_space(self):
        """Removes heading and trailing spaces in the span text."""
        # heading spaces
        text = self.text.lstrip(' ')
        self.start += len(self.text) - len(text)
        # trailing spaces
        text = text.rstrip(' ')
        self.text = text
        self.end = self.start + len(text)

    def copy(self):
        """Makes a copy of itself.

        Returns:
            Span: a copy of itself."""
        return Span(self.start, self.end, self.text)


@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    entity_subtype: str
    mention_type: str
    value: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict: a dict of instance variables.
        """
        entity_dict = {
            'text': recover_escape(self.text),
            'entity_id': self.entity_id,
            'mention_id': self.mention_id,
            'start': self.start,
            'end': self.end,
            'entity_type': self.entity_type,
            'entity_subtype': self.entity_subtype,
            'mention_type': self.mention_type
        }
        if self.value:
            entity_dict['value'] = self.value
        return entity_dict


@dataclass
class RelationArgument:
    mention_id: str
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'mention_id': self.mention_id,
            'role': self.role,
            'text': recover_escape(self.text)
        }


@dataclass
class Relation:
    relation_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgument
    arg2: RelationArgument

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'relation_id': self.relation_id,
            'relation_type': self.relation_type,
            'relation_subtype': self.relation_subtype,
            'arg1': self.arg1.to_dict(),
            'arg2': self.arg2.to_dict(),
        }


@dataclass
class EventArgument:
    mention_id: str
    role: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'mention_id': self.mention_id,
            'role': self.role,
            'text': recover_escape(self.text),
        }


@dataclass
class Event:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    trigger: Span
    arguments: List[EventArgument]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'event_id': self.event_id,
            'mention_id': self.mention_id,
            'event_type': self.event_type,
            'event_subtype': self.event_subtype,
            'trigger': self.trigger.to_dict(),
            'arguments': [arg.to_dict() for arg in self.arguments],
        }


@dataclass
class Sentence(Span):
    sent_id: str
    tokens: List[str]
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'sent_id': self.sent_id,
            'tokens': [recover_escape(t) for t in self.tokens],
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations],
            'events': [event.to_dict() for event in self.events],
            'start': self.start,
            'end': self.end,
            'text': recover_escape(self.text).replace('\t', ' '),
        }

@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'doc_id': self.doc_id,
            'sentences': [sent.to_dict() for sent in self.sentences]
        }


def read_ann_file(path: str,
                  time_and_val: bool = False
                 ) -> Tuple[str, str, List[Entity], List[Relation], List[Event]]:
    """Reads an Ann file.

    Args:
        path (str): path to the input file.
        time_and_val (bool): extract times and values or not.
    
    Returns:
        doc_id (str): document ID.
        source (str): document source.
        entity_list (List[Entity]): a list of Entity instances.
        relation_list (List[Relation]): a list of Relation instances.
        event_list (List[Event]): a list of Events instances.
    """
    data = open(path, 'r', encoding='utf-8').read()
    
    # metadata
    doc_id = os.path.basename(path)
    doc_id = os.path.splitext(doc_id)[0]

    entity_list, relation_list, event_list = [], [], []

    reader = read_brat.ReadBrat()
    tok_lst = reader.process_file(path.replace(".ann",""))
    
    # a stack to collect the tokens labeled as partipant (entity)
    partipant_tok_lst = [] 
    # a stack to collect the tokens labeled as event 
    event_tok_lst = []
    for tok in tok_lst:
        for attr_item in tok.attr:
            ann_type = attr_item[0]
            attr_map = attr_item[1]

            if ann_type == "Participant":
                

            #entity_list.append(Entity(start, end, text,
            #                          entity_id, mention_id, entity_type,
            #                          entity_subtype, mention_type))

    
    
            #event_list.append(Event(event_id, mention_id,
            #                        event_type, event_subtype,
            #                        Span(trigger_start,
            #                             trigger_end + 1, trigger_text),
            #                        event_args))

    
    return doc_id, source, entity_list, relation_list, event_list


def convert(txt_file: str,
            ann_file: str,
            time_and_val: bool = False,
            language: str = 'english') -> Document:
    """Converts a document.

    Args:
        txt_file (str): path to a txt file.
        ann_file (str): path to a ann file.
        time_and_val (bool, optional): extracts times and values or not.
            Defaults to False.
        language (str, optional): document language. Available options: english,
            chinese. Defaults to 'english'.

    Returns:
        Document: a Document instance.
    """
    sentences = read_txt_file(ann_file, language=language)
    doc_id, source, entities, relations, events = read_ann_file(
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
            chinese, portuguese. Defaults to 'english'.
    """
    if language == 'english':
        txt_files = glob.glob(os.path.join(
            input_path, '**', 'timex2norm', '*.txt'))
    elif language == 'chinese':
        txt_files = glob.glob(os.path.join(
            input_path, '**', 'adj', '*.txt'))
    elif language == 'portuguese':
        txt_files = glob.glob(os.path.join(
            input_path, '*.txt'))
    else:
        raise ValueError('Unknown language: {}'.format(language))

    print('Converting the dataset to JSON format')
    print('#ANN files: {}'.format(len(txt_files)))
    progress = tqdm.tqdm(total=len(txt_files))

    with open(output_path, 'w', encoding='utf-8') as w:
        for txt_file in txt_files:
            progress.update(1)
            
            ann_file = txt_file.replace('.txt', '.ann')
            doc = convert(txt_file, ann_file, time_and_val=time_and_val,
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