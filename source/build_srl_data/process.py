import text2story as t2s

from text2story.annotators import ALLENNLP

import glob
import os
import json
from argparse import ArgumentParser

def main(input_dir):

    json_files = glob.glob(os.path.join(
            input_dir, '*.json'))


    ALLENNLP.load("pt")

    for f in json_files:
        with open(f, "r") as fd:

            data = json.load(fd)
            doc_id = data["doc_id"]

            sentences = data["sentences"]
            for sent in sentences:
                sent_id = sent["sent_id"]
                sentence_text = sent["text"]

                narrative_doc = t2s.Narrative("pt", sentence_text, "2023-01-01")
                srl = ALLENNLP.pipeline["srl_pt"].predict_json(narrative_doc.text)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input folder')

    args = parser.parse_args()
    input_dir = args.input
    main(input_dir)