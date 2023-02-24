import text2story as t2s

from text2story.annotators import ALLENNLP

import glob
import os
import json
from argparse import ArgumentParser

def main(input_dir, output_dir):

    json_files = glob.glob(os.path.join(
            input_dir, '*.json'))


    ALLENNLP.load("pt")

    for f in json_files:
        output_file = os.path.basename(f)
        output_file = os.path.splitext(output_file)[0]
        output_file = os.path.join(output_dir, "verbSRL%s.json" % output_file)

        fd_output = open(output_file, "a")

        with open(f, "r") as fd:

            data = json.load(fd)
            data_out = {}
            doc_id = data["doc_id"]

            sentences = data["sentences"]
            for sent in sentences:
                sent_id = sent["sent_id"]
                sentence_text = sent["text"]

                narrative_doc = t2s.Narrative("pt", sentence_text, "2023-01-01")
                srl = ALLENNLP.pipeline["srl_pt"].predict_json(narrative_doc.text)

                if isinstance(srl, dict):
                    data_out["doc_id"] = doc_id
                    data_out["sent_id"] = sent_id
                    data_out["sentence"] = sentence_text
                    data_out["verbs"] = srl["verbs"]

                    json.dump(data_out,fd_output)

        fd_output.close()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('-i', '--input', help='Path to the input folder')
    parser.add_argument('-o', '--output', help='Path to the output folder')


    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    main(input_dir, output_dir)