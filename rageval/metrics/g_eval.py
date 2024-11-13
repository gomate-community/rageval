from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple, Dict
import datasets
from nltk import ngrams
from rageval.metrics import Metric, add_attribute
from rageval.models.openai import OpenAILLM

_DESCRIPTION = """\
Code for paper "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment". See https://arxiv.org/abs/2303.16634 for more details.
"""

_KWARGS_DESCRIPTION = """\
Args:
    pred_answers (list of str): List of generated texts for which distinct metrics are computed.
    n_grams (int): The n-gram order for which distinct metrics are computed.

Returns:
    dict: Dictionary containing Distinct-1 and Distinct-2 scores.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
        "contexts": [
            "Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with Burnley on Sunday . 'Just been watching the game , did you miss the coach ? # RubberDub # 7minutes , ' Merson put on Twitter . Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in ( the England team ) then it opens it up to anybody . ' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley Andros Townsend scores England 's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake . 'It 's not as though I was watching hoping he would n't score for England , I 'm genuinely pleased for him and fair play to him \u00e2\u20ac\u201c it was a great goal , ' Merson said . 'It 's just a matter of opinion , and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson , so he should n't have been in the squad . 'When I 'm wrong , I hold my hands up . I do n't have a problem with doing that - I 'll always be the first to admit when I 'm wrong . ' Townsend hit back at Merson on Twitter after scoring for England against Italy Sky Sports pundit Merson ( centre ) criticised Townsend 's call-up to the England squad last week Townsend hit back at Merson after netting for England in Turin on Wednesday , saying 'Not bad for a player that should be 'nowhere near the squad ' ay @ PaulMerse ? ' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor ."]
    ...     "answers": [
                "Andros Townsend an 83rd minute sub in Tottenham 's draw with Burnley . He was unable to find a winner as the game ended without a goal . Townsend had clashed with Paul Merson last week over England call-up ."
    ...     ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> metric = rl.metrics.GEval(dimension="coherent")
    >>> metric.mtype
    'Multidimensional'
    >>> score, results = metric.compute(contexts=dataset['contexts'], pred_answers=dataset['answers'])
    >>> score
    1.1
"""

_CITATION = """\
@misc{liu2023gevalnlgevaluationusing,
      title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
      author={Yang Liu and Dan Iter and Yichong Xu and Shuohang Wang and Ruochen Xu and Chenguang Zhu},
      year={2023},
      eprint={2303.16634},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2303.16634},
}
"""

PROMPT_TEMPLATE = {
    "coherent": """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension with the DUC quality question of structure and coherence whereby "the summary should be well-structured and well-organized. The summary should not just be a heap of related information, but should build from sentence to a coherent body of information about a topic."

Evaluation Steps:

1. Read the news article carefully and identify the main topic and key points.
2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Source Text:

{Document}

Summary:

{Summary}


Evaluation Form (scores ONLY):

- Coherence (1-5):""",
    "consistency": """You will be given a news article. You will then be given one summary written for this article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Consistency (1-5) - the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.

Evaluation Steps:

1. Read the news article carefully and identify the main facts and details it presents.
2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
3. Assign a score for consistency based on the Evaluation Criteria.


Source Text:

{Document}

Summary:

{Summary}


Evaluation Form (scores ONLY):

- Consistency (1-5):""",
    "fluency": """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


Evaluation Criteria:

Fluency (1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

- 1: Poor. The summary has many errors that make it hard to understand or sound unnatural.
- 2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.
- 3: Good. The summary has few or no errors and is easy to read and follow.


Summary:

{Summary}


Evaluation Form (scores ONLY):

- Fluency (1-3):""",
    "relevance": """You will be given one summary written for a news article.

Your task is to rate the summary on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Relevance (1-5) - selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess information.

Evaluation Steps:

1. Read the summary and the source document carefully.
2. Compare the summary to the source document and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Assign a relevance score from 1 to 5.


Source Text:

{Document}

Summary:

{Summary}


Evaluation Form (scores ONLY):

- Relevance (1-5):""",
}


def to_prompt(dimension: str, context: str, pred_answer: str) -> str:
    return PROMPT_TEMPLATE[dimension].format(Document=context, Summary=pred_answer)


@dataclass
@add_attribute('mtype', 'Multidimensional')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class GEval(Metric):
    """G-Eval, a framework of using large language models with chain-of-thoughts (CoT) and a form-filling paradigm, to assess the quality of NLG outputs"""

    name = "g_eval"

    ALIAS = ['g_eval']

    def __init__(self,
                 model: str = "gpt-4o",
                 _api_key_env_var: str = "API_KEY",
                 base_url: str = "https://api.openai.com/v1",
                 dimension: str = None,
                 ):
        """
        Explicitly initialize Distinct.

        Ensure all parent classes are initialized.
        """
        super().__init__()
        self.llm = OpenAILLM(model=model,
                             _api_key_env_var=_api_key_env_var,
                             base_url=base_url,
                             temperature=1,
                             max_tokens=2,
                             top_p=1,
                             n=20)
        assert dimension is not None, "Please specify the dimensions to evaluate."
        assert dimension in PROMPT_TEMPLATE, f"Invalid dimensions specified. Dimension must be one of {list(PROMPT_TEMPLATE.keys())}."
        self.dimension = dimension

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "contexts": datasets.Value("string"),
                    "pred_answers": datasets.Value("string"),
                }
            ),
            codebase_urls=["https://github.com/nlpyang/geval"],
            reference_urls=["https://arxiv.org/abs/2303.16634"]
        )

    def _generate_messages(self,
                           pred_answers: List[str],
                           contexts: List[str]) -> List[List[Dict[str, str]]]:
        msgs = []
        for pred_answer, context in zip(pred_answers, contexts):
            msgs.append([{"role": "system", "content": to_prompt(self.dimension, context, pred_answer)}])
        return msgs

    def compute(
        self,
        pred_answers: List[str] = None,
        contexts: List[str] = None,
        batch_size: int = 1,
    ) -> Tuple[float, List[float]]:
        """
        Evaluate the dataset.

        Return average scores of all inputs and a score list for each example.
        """
        self._validate_data(pred_answers, None, contexts)
        msgs = self._generate_messages(pred_answers, contexts)
        results = self.llm.batch_generate(msgs, max_workers=batch_size)
        scores = []
        for r in results:
            instance_scores = []
            for g in r.generations[0]:
                if g.generation_info['stop'] != 'stop':
                    continue
                try:
                    score = int(g.text[:1])
                except ValueError:
                    pass
                instance_scores.append(score)
            avg_score = sum(instance_scores) / len(instance_scores) if len(instance_scores) > 0 else 0
            scores.append(avg_score)
        return sum(scores) / len(scores), scores
