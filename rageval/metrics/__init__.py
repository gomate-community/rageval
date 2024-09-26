from .base import Metric, MetricWithLLM, add_attribute

# Metrics about the answer correctness
from .answer_correctness._answer_accuracy import AnswerAccuracy
from .answer_correctness._answer_bleu import AnswerBleuScore
from .answer_correctness._answer_chrf import AnswerCHRFCorrectness
from .answer_correctness._answer_exact_match import AnswerEMCorrectness
from .answer_correctness._answer_f1 import AnswerF1Correctness
from .answer_correctness._answer_rouge_correctness import AnswerRougeCorrectness
from .answer_correctness._answer_bert_score import AnswerBERTScore
from .answer_correctness._answer_edit_distance import AnswerEditDistance
from .answer_correctness._answer_claim_recall import AnswerNLICorrectness
from .answer_correctness._answer_disambig_f1 import AnswerDisambigF1Correctness
from .answer_correctness._answer_lcs_ratio import AnswerLCSRatio
from .answer_correctness._answer_ter import AnswerTERCorrectness
##from .answer_correctness._answer_relevancy import AnswerRelevancy

# Metrics about the answer groundedness
from .answer_groundedness._answer_citation_precision import AnswerCitationPrecision
from .answer_groundedness._answer_citation_recall import AnswerCitationRecall
from .answer_groundedness._context_reject_rate import ContextRejectRate
##from .answer_groundedness._claim_faithfulness import ClaimFaithfulness

# Metrics about the answer informativeness
##from .answer_informative._claim_num import ClaimNum
from .answer_informativeness._text_length import TextLength
##from .answer_informativeness._repetitiveness import Repetitiveness
##from .answer_informativeness._pairwise_accuracy import PairwiseAccuracy
from .answer_informativeness._answer_distinct12 import AnswerDistinct

# Metrics about the context relevancy
from .context_relevancy._context_recall import ContextRecall

# Metrics about the context aduquacy
from .context_adequacy._context_recall import ContextRecall

# Metrics about the context relevance

