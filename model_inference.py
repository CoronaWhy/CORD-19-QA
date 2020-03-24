from transformers import BertTokenizer, BertForQuestionAnswering
from bm25_index import BM25Index
import torch

class QAInfer(object):
  def __init__(self, vocab_path:str, model_save_path:str, bm25_root):
    self.tokenizer = BertTokenizer.from_pretrained(vocab_path, add_special_token=False)
    self.model = BertForQuestionAnswering.from_pretrained(model_save_path)
    self.bm25 = pickle.load(open(bm25_root, "rb"))

  def run_query(self, query:str, number_to_use:int, abstract_or_text='abstract'):
    useful_results = self.bm25.search(query, number_to_use)[abstract_or_text]
    result_list = []
    # TODO combine into a single batch and have inference only once on the batch
    for result in useful_results:
      result_list.append(self.infer_question_answer(query, result))
    return result_list

  def infer_question_answer(self, question:str, text:str)->str:
    encoded = self.tokenizer.encode_plus(question,
                                         text,
                                         add_special_tokens=False,
                                         return_tensors='pt')    
    start_logits, end_logits = self.model(**encoded)

    # We need to know where to start looking for an answer. This approach only
    # works under the assumption we are predicting on a single instance at a
    # time.
    question_start_idx = torch.sum(encoded['token_type_ids'] == 0).item()
    start_logits = start_logits[:, question_start_idx:]
    end_logits = end_logits[:, question_start_idx:]
    prediction = get_best_span(start_logits, end_logits).squeeze()
    all_tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'].squeeze().tolist())
    answer = ' '.join(all_tokens[prediction[0] + question_start_idx : prediction[1] + question_start_idx + 1])
    return answer.replace(" ##", "")

def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
"""
This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
directly import this function without the class.

We call the inputs "logits" - they could either be unnormalized logits or normalized log
probabilities.  A log_softmax operation is a constant shifting of the entire logit
vector, so taking an argmax over either one gives the same result.
"""
if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
    raise ValueError("Input shapes must be (batch_size, passage_length)")
batch_size, passage_length = span_start_logits.size()
device = span_start_logits.device
# (batch_size, passage_length, passage_length)
span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
# Only the upper triangle of the span matrix is valid; the lower triangle has entries where
# the span ends before it starts.
span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
valid_span_log_probs = span_log_probs + span_log_mask

# Here we take the span matrix and flatten it, then find the best span using argmax.  We
# can recover the start and end indices from this flattened list using simple modular
# arithmetic.
# (batch_size, passage_length * passage_length)
best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
span_start_indices = best_spans // passage_length
span_end_indices = best_spans % passage_length
return torch.stack([span_start_indices, span_end_indices], dim=-1)