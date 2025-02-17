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


import sys
import spacy
import numpy as np
from text2story.readers import read_brat
from text2story.core.utils import join_tokens

from nltk import (sent_tokenize as sent_tokenize_,
                  wordpunct_tokenize as wordpunct_tokenize_)


# repo root dir; change to your own
#root_dir = "/shared/lyuqing/Zeroshot-Event-Extraction"
root_dir = "/home/evelinamorim/UPorto/zero-shot-participant/zeroshot-pt/Zeroshot-Event-Extraction"
os.chdir(root_dir)

def mask_escape(text: str) -> str:
    """Replaces escaped characters with rare sequences.

    Args:
        text (str): text to mask.
    
    Returns:
        str: masked string.
    """
    return text.replace('&amp;', 'ҪҪҪҪҪ').replace('&lt;', 'ҚҚҚҚ').replace('&gt;', 'ҺҺҺҺ')

def unmask_escape(text: str) -> str:
    """Replaces masking sequences with the original escaped characters.

    Args:
        text (str): masked string.
    
    Returns:
        str: unmasked string.
    """
    return text.replace('ҪҪҪҪҪ', '&amp;').replace('ҚҚҚҚ', '&lt;').replace('ҺҺҺҺ', '&gt;')


def recover_escape(text: str) -> str:
    """Converts named character references in the given string to the corresponding
    Unicode characters. I didn't notice any numeric character references in this
    dataset.

    Args:
        text (str): text to unescape.
    
    Returns:
        str: unescaped string.
    """
    return text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')


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
            if len(sent) > 0:
                offset_start = sent[0].idx
                offset_end = offset_start + len(sent.text.lstrip())
                sentences.append([sent.text, offset_start, offset_end])
                

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

def process_relation(relations: List[Relation],
                     sentence_entities: List[List[Entity]],
                     sentences: List[Tuple[str, int, int]]
                    ) -> List[List[Relation]]:
    """Cleans and assigns relations

    Args:
        relations (List[Relation]): a list of Relation instances.
        sentence_entities (List[List[Entity]]): a list of sentence entity lists.
        sentences (List[Tuple[str, int, int]]): a list of sentences.

    Returns:
        List[List[Relation]]: a list of sentence relation lists.
    """
    sentence_relations = [[] for _ in range(len(sentences))]
    for relation in relations:
        mention_id1 = relation.arg1.mention_id
        mention_id2 = relation.arg2.mention_id
        for i, entities in enumerate(sentence_entities):
            arg1_in_sent = any([mention_id1 == e.mention_id for e in entities])
            arg2_in_sent = any([mention_id2 == e.mention_id for e in entities])
            if arg1_in_sent and arg2_in_sent:
                sentence_relations[i].append(relation)
                break
            elif arg1_in_sent != arg2_in_sent:
                break
    return sentence_relations

def process_entities(entities: List[Entity],
                     sentences: List[Tuple[str, int, int]]
                    ) -> List[List[Entity]]:
    """Cleans entities and splits them into lists

    Args:
        entities (List[Entity]): a list of Entity instances.
        sentences (List[Tuple[str, int, int]]): a list of sentences.

    Returns:
        List[List[Entity]]: a list of sentence entity lists.
    """
    sentence_entities = [[] for _ in range(len(sentences))]

    # assign each entity to the sentence where it appears
    for entity in entities:
        start, end = entity.start, entity.end
        for i, (_, s, e) in enumerate(sentences):
            if start >= s and end <= e:
                sentence_entities[i].append(entity)
                assigned = True
                break

    # remove overlapping entities
    sentence_entities_cleaned = [[] for _ in range(len(sentences))]
    for i, entities in enumerate(sentence_entities):
        if not entities:
            continue
        # prefer longer entities
        entities.sort(key=lambda x: (x.end - x.start), reverse=True)
        chars = [0] * max([x.end for x in entities])
        for entity in entities:
            overlap = False
            for j in range(entity.start, entity.end):
                if chars[j] == 1:
                    overlap = True
                    break
            if not overlap:
                chars[entity.start:entity.end] = [
                    1] * (entity.end - entity.start)
                sentence_entities_cleaned[i].append(entity)
        sentence_entities_cleaned[i].sort(key=lambda x: x.start)

    return sentence_entities_cleaned

def process_events(events: List[Event],
                   sentence_entities: List[List[Entity]],
                   sentences: List[Tuple[str, int, int]]
                  ) -> List[List[Event]]:
    """Cleans and assigns events.

    Args:
        events (List[Event]): A list of Event objects
        entence_entities (List[List[Entity]]): A list of sentence entity lists.
        sentences (List[Tuple[str, int, int]]): A list of sentences.
    
    Returns:
        List[List[Event]]: a list of sentence event lists.
    """
    sentence_events = [[] for _ in range(len(sentences))]
    # assign each event mention to the sentence where it appears
    for event in events:
        start, end = event.trigger.start, event.trigger.end
        for i, (_, s, e) in enumerate(sentences):
            sent_entities = sentence_entities[i]
            if start >= s and end <= e:
                event_cleaned = Event(event.event_id, event.mention_id,
                                      event.event_type, event.event_subtype,
                                      trigger=event.trigger.copy())
                sentence_events[i].append(event_cleaned)

    # remove overlapping events
    sentence_events_cleaned = [[] for _ in range(len(sentences))]
    for i, events in enumerate(sentence_events):
        if not events:
            continue
        events.sort(key=lambda x: (x.trigger.end - x.trigger.start),
                    reverse=True)
        chars = [0] * max([x.trigger.end for x in events])
        for event in events:
            overlap = False
            for j in range(event.trigger.start, event.trigger.end):
                if chars[j] == 1:
                    overlap = True
                    break
            if not overlap:
                chars[event.trigger.start:event.trigger.end] = [
                    1] * (event.trigger.end - event.trigger.start)
                sentence_events_cleaned[i].append(event)
        sentence_events_cleaned[i].sort(key=lambda x: x.trigger.start)

    return sentence_events_cleaned


def create_participant(part_id, part_tok_lst):

    entity_id = part_id
    mention_id = part_id


    attr_type, attr_map = part_tok_lst[0].attr[0]
    if "Participant_Type_Domain" in attr_map:
        entity_type = attr_map["Participant_Type_Domain"]
        entity_subtype = attr_map["Participant_Type_Domain"]
        mention_type = attr_map["Participant_Type_Domain"]
    else:
        entity_type, entity_subtype, mention_type = "", "", ""

    tok_txt_lst = [tok_part.text for tok_part in part_tok_lst]

    text = join_tokens(tok_txt_lst)

    start = part_tok_lst[0].offset
    end = start + len(text)

    return Entity(start, end, text,
                  entity_id, mention_id, entity_type,
                    entity_subtype, mention_type)

def create_event(event_id, event_tok_lst):

    mention_id = event_id
    ann_type, attr_map = event_tok_lst[0].attr[0]

    if "Class" in attr_map:
        event_type = attr_map["Class"]
    else:
        event_type = ""
    event_subtype = event_type

    mention_type = ann_type

    tok_txt_lst = [tok.text for tok in event_tok_lst]
    trigger_text = join_tokens(tok_txt_lst)

    trigger_start = event_tok_lst[0].offset
    trigger_end = trigger_start + len(trigger_text)

    #print("-->", trigger_text, trigger_start, trigger_end)


    return Event(event_id, mention_id,
                event_type, event_subtype,
                Span(trigger_start,
                    trigger_end, trigger_text))

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
    source = os.path.basename(path)
    doc_id = os.path.splitext(source)[0]

    entity_list, relation_list, event_list = [], [], []

    reader = read_brat.ReadBrat()
    tok_lst = reader.process_file(path.replace(".ann",""))
    
    # a stack to collect the tokens labeled as partipant (entity)
    participant_tok_lst = [] 
    # a stack to collect the tokens labeled as event 
    event_tok_lst = []

    old_participant_id = None
    old_event_id = None

    map_id_ann = {}

    for tok in tok_lst:

        for id_ann in tok.id_ann:
            if id_ann in map_id_ann:
                map_id_ann[id_ann].append(tok)
            else:
                map_id_ann[id_ann] = [tok]


        for idx_ann, attr_item in enumerate(tok.attr):
            ann_type = attr_item[0]
            attr_map = attr_item[1]

            id_ann = tok.id_ann[idx_ann]
            
            if ann_type == "Participant":


                if id_ann != old_participant_id and old_participant_id != None:
                    #print(id_ann, old_id_ann)
                    participant = create_participant(old_participant_id, participant_tok_lst)
                    entity_list.append(participant)
                    participant_tok_lst = []

                participant_tok_lst.append(tok)
        
                old_participant_id = id_ann

            if ann_type == "Event":

                if id_ann != old_event_id and old_event_id != None:
                    #print(id_ann, old_id_ann)

                    event = create_event(old_event_id, event_tok_lst)
                    event_list.append(event)
                    event_tok_lst = []

                event_tok_lst.append(tok)
        
                old_event_id = id_ann

    # the last participant was left in the list
    if old_event_id != None:
        participant = create_participant(old_participant_id, participant_tok_lst)
        entity_list.append(participant)

    # the last event..
    if old_event_id != None:
        event = create_event(old_event_id, event_tok_lst)
        event_list.append(event)

    rel_id_set = set()
    for tok in tok_lst:
        for rel in tok.relations:

            mention_id = rel.rel_id

            if mention_id in rel_id_set:
                continue

            relation_type = rel.rel_type
            relation_subtype = rel.rel_type

            if rel.argn == "arg1":

                arg_mention_id2 = tok.id_ann[0]
                arg_role2 = "Arg-2"
                tok_lst_txt = [tok_arg.text for tok_arg in map_id_ann[arg_mention_id2]]
                arg_text2 = join_tokens(tok_lst_txt)
                
                arg_mention_id1 = rel.toks[0].id_ann[0]
                arg_role1 = "Arg-1"
                tok_lst_txt = [tok_arg.text for tok_arg in rel.toks]
                arg_text1 = join_tokens(tok_lst_txt)
            else:
                arg_mention_id1 = tok.id_ann[0]
                arg_role1 = "Arg-1"
                tok_lst_txt = [tok.text for tok in map_id_ann[arg_mention_id1]]
                arg_text1 = join_tokens(tok_lst_txt)
                
                arg_mention_id2= rel.toks[0].id_ann[0]
                arg_role2 = "Arg-2"
                tok_lst_txt = [tok_arg.text for tok_arg in rel.toks]
                arg_text2 = join_tokens(tok_lst_txt)

            arg1 =  RelationArgument(arg_mention_id1, arg_role1, arg_text1)
            arg2 =  RelationArgument(arg_mention_id2, arg_role2, arg_text2)

            relation_list.append(Relation(mention_id, relation_type,
                                              relation_subtype, arg1, arg2))
            rel_id_set.add(mention_id)
                
    
    
            #event_list.append(Event(event_id, mention_id,
            #                        event_type, event_subtype,
            #                        Span(trigger_start,
            #                             trigger_end + 1, trigger_text),
            #                        event_args))

    
    return doc_id, source, entity_list, relation_list, event_list

def wordpunct_tokenize(text: str, language: str = 'english') -> List[str]:
    """Performs word tokenization. For English, it uses NLTK's 
    wordpunct_tokenize function. For Chinese, it simply splits the sentence into
    characters.
    
    Args:
        text (str): text to split into words.
        language (str): available options: english, chinese.

    Returns:
        List[str]: a list of words.
    """
    if language == 'chinese':
        return [c for c in text if c.strip()]
    return wordpunct_tokenize_(text)

def tokenize(sentence: Tuple[str, int, int],
             entities: List[Entity],
             events: List[Event],
             language: str = 'english'
            ) -> List[Tuple[int, int, str]]:
    """Tokenizes a sentence.
    Each sentence is first split into chunks that are entity/event spans or words
    between two spans. After that, word tokenization is performed on each chunk.

    Args:
        sentence (Tuple[str, int, int]): Sentence tuple (text, start, end)
        entities (List[Entity]): A list of Entity instances.
        events (List[Event]): A list of Event instances.

    Returns:
        List[Tuple[int, int, str]]: a list of token tuples. Each tuple consists
        of three elements, start offset, end offset, and token text.
    """

    text, start, end = sentence
    text = mask_escape(text)

    # split the sentence into chunks
    splits = {0, len(text)}

    for entity in entities:
        splits.add(entity.start - start)
        splits.add(entity.end - start)
    for event in events:
        splits.add(event.trigger.start - start)
        splits.add(event.trigger.end - start)
    splits = sorted(list(splits))
    chunks = [(splits[i], splits[i + 1], text[splits[i]:splits[i + 1]])
              for i in range(len(splits) - 1)]

    # tokenize each chunk
    chunks = [(s, e, t, wordpunct_tokenize(t, language=language))
              for s, e, t in chunks]

    # merge chunks and add word offsets
    tokens = []
    for chunk_start, chunk_end, chunk_text, chunk_tokens in chunks:
        last = 0
        chunk_tokens_ = []
        for token in chunk_tokens:
            token_start = chunk_text[last:].find(token)
            if token_start == -1:
                raise ValueError(
                    'Cannot find token {} in {}'.format(token, text))
            token_end = token_start + len(token)
            chunk_tokens_.append((token_start + start + last + chunk_start,
                                  token_end + start + last + chunk_start,
                                  unmask_escape(token)))
            last += token_end
        tokens.extend(chunk_tokens_)

    return tokens

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

    sentences = read_txt_file(txt_file, language=language)
    doc_id, source, entities, relations, events = read_ann_file(
        ann_file, time_and_val=time_and_val)


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