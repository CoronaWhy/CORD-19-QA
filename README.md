CORD-19 Question Answering
===

This repo contains tools for training/running neural mult-document question answering models to help researchers identify relevant articles in the CORD-19 dataset.
Currently very under construction. STAY TUNED FOR UPDATES!

## Tasks

- Build BM25 index over CORD-19 to retrieve relevant paragraphs (from abstracts and results)
- Convert BioASQ-8b to SQuAD format, with full-abstract contexts
- Fine-tune the pretrained BioBERT on SQuAD-ified BioASQ-8b
- Evaluate on CORD-19 (connect the retriever with the answerer)

## Resources

- https://github.com/facebookresearch/DrQA
- http://participants-area.bioasq.org/
- https://github.com/dmis-lab/bioasq-biobert