"""
Converts BioASQ format to SQuAD format.
"""
import argparse
import json
import logging
from multiprocessing.pool import ThreadPool
from pathlib import Path
from xml.etree import ElementTree

import requests


def generate_questions(fname):
    """
    Generates questions from a JSON file in the BioASQ dataset.
    """
    with open(fname, 'r') as f:
        data = json.load(f)
    for question in data['questions']:
        yield question


def query_pubmed(docid):
    """
    Queries PuBMed API for XML metadata associated to a given document.
    """
    endpoint = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    payload = {
        'db': 'pubmed',
        'id': docid,
        'retmode': 'xml'
    }
    response = requests.get(endpoint, params=payload)
    try:
        xml_metadata = ElementTree.fromstring(response.text)
    except:
        print(response.text)
    return xml_metadata


def render_section(abstract_section):
    """
    Renders a section of the abstract as text. Potentailly including the
    section title.
    """
    # WARNING: This adds labels to each section if available. This appears to
    # what was done to (most of) the abstracts BioASQ. However sometimes this
    # strategy is preventing matches.
    label = abstract_section.get('Label')
    if label is not None:
        return f'{label}: {abstract_section.text}'
    else:
        return abstract_section.text


def extract_context(xml_metadata):
    """
    Extracts the abstract from PubMed XML metadata.
    """
    title = xml_metadata.find('.//ArticleTitle').text
    abstract_sections = xml_metadata.findall('.//AbstractText')
    abstract = ' '.join(render_section(x) for x in abstract_sections)
    context = f'TITLE: {title} ABSTRACT: {abstract}'
    return context


def squadify(question):
    """
    Converts a BioASQ-formatted instance to a SQuAD-formatted instance.
    """
    squad_instance = {
        'title': question['id'],
        'paragraphs': []
    }
    for snippet in question['snippets']:
        docid = snippet['document'].split('/')[-1]
        xml_metadata = query_pubmed(docid)
        context = extract_context(xml_metadata)
        answer_text = snippet['text']
        answer_start = context.find(answer_text)
        # TODO: Analyze remaining failure modes.
        if answer_start == -1:
            continue
        paragraph = {
            'id': docid,
            'context': context,
            'question': question['body'],
            'qas': [{
                "answer_start": answer_start,
                "text": answer_text
            }]
        }
        squad_instance['paragraphs'].append(paragraph)
    return squad_instance


def convert(source, dest):
    """
    Converts BioASQ-formatted dataset, to a SQuAD-formatted dataset.
    """
    questions = generate_questions(args.input)
    output = {'data': [squadify(x) for x in questions]}
    with open(args.output, 'w') as f:
        json.dump(output, f, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=Path,
        required=True,
        help='input BioASQ file'
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        required=True,
        help='output file'
    )
    args = parser.parse_args()

    convert(args.input, args.output)

