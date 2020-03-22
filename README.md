CORD-19 Question Answering
===

This repo contains tools for training/running neural mult-document question answering models to help researchers identify relevant articles in the CORD-19 dataset.
Currently very under construction. STAY TUNED FOR UPDATES!

## Tasks

- Build BM25 index over CORD-19 to retrieve relevant paragraphs (from abstracts and results)
- Convert BioASQ-8b to SQuAD format, with full-abstract contexts
- Fine-tune the pretrained BioBERT on SQuAD-ified BioASQ-8b
- Evaluate on CORD-19 (connect the retriever with the answerer)

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
