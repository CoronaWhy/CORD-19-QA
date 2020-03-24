CORD-19 Question Answering
===

This repo contains tools for training/running neural multi-document question answering models to help researchers identify relevant articles in the CORD-19 dataset.
Currently very under construction. STAY TUNED FOR UPDATES!

## Overview

Our approach is based off of [DrQA](https://github.com/facebookresearch/DrQA)(Chen et al. 2017).
Our is pipeline comprised of two components:
1. A document retriever. Currently, a BM25 search index.
2. A document reader. Currently, BioBERT fine-tuned on the BioASQ dataset.

## Tasks

Tasks directly related to this project are tracked [here](https://github.com/CoronaWhy/CORD-19-QA/projects/1).

Please note that we have a healthy team of contributors for this project.
If you are looking for a project to contribute to but do not have much experience with QA, then we recommend trying to solve one of the tasks posted in our [requests for other teams](https://github.com/CoronaWhy/CORD-19-QA/projects/2).

## Usage

### BM25 Index `bm25_index.py`
If you are running this file for the first time, you will need to manually build a BM25 index. For subsequent runs, the BM25 index can be loaded from a pickle file. The most important command-line arguments to worry about are:

* `--data-dir`: path to your data folder. This will walk the directory tree starting from the specified directory and process all `.json` files.
* `--index-path`: path to the BM25 index. If building an index, this is where it will be saved; if loading an index, this is where it will be loaded.
* `--query`: the query (a string).
* `--result-path-base`: the base path where the results will be saved as a `.csv`.

Other arguments include:
* `--nresults`: number of results to return. Ordered by descending score.
* `--rebuild-index`: rebuilds the BM25 index from scratch.
* `--paragraphs`: only used if `rebuild-index` is specified; whether to build the index on abstracts or paragraphs.

## Resources

- https://github.com/facebookresearch/DrQA
- http://participants-area.bioasq.org/
- https://github.com/dmis-lab/bioasq-biobert
