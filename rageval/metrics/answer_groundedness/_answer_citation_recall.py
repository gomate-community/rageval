import re
from dataclasses import dataclass
from typing import List, Callable

import datasets
from tqdm import tqdm

from rageval.metrics import Metric, add_attribute
from rageval.utils import text_to_sents, remove_citations

_DESCRIPTION = """\
Citation recall determines if the result from LLM is entirely supported by cited passages.

In different RAG evaluation tasks, both ‘contexts’ and ‘gt_contexts’ may be used as part of the input of LLM.
This metric doesn't care whether the ‘contexts’ come from real-time retrieval or annotated datasets.
For simplicity, we refer to all contexts collectively as ‘contexts’.

For details, see the paper: https://arxiv.org/abs/2305.14627.
"""

_KWARGS_DESCRIPTION = r"""\
Args:
    name : str
    batch_size : int, Batch size for openai completion.

Optional Args:
    None

Functions:
    _compute_one: compute the citation recall of an answer.

Examples:
    >>> from datasets import Dataset
    >>> import rageval as rl
    >>> sample = {
    ...     "answers": [
    ...         "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an "
    ...         "average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which "
    ...         "reported an annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by "
    ...         "Mawsynram, India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, "
    ...         "India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 "
    ...         "and most rain in a year from August 1860 to July 1861 [1]."
    ...     ],
    ...    "contexts": [
    ...        [
    ...             "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be "
    ...             "spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in "
    ...             "the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal "
    ...             "chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often "
    ...             "been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds "
    ...             "that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar "
    ...             "month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received "
    ...             "in",
    ...             "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji "
    ...             "Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled "
    ...             "Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the "
    ...             "Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal "
    ...             "chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often "
    ...             "been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds "
    ...             "that distinction. Cherrapunji still holds the all-time record for the most rainfall",
    ...             "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in "
    ...             "north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls "
    ...             "in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 "
    ...             "mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of "
    ...             "12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual "
    ...             "12,892 mm per year between 1960 and 2012. According to the \"Guinness Book of World Records\", "
    ...             "Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′"
    ...        ]
    ...    ]
    ... }
    >>> dataset = Dataset.from_dict(sample)
    >>> nli_model = rl.models.NLIModel(
    ...     'text2text-generation',
    ...     'hf-internal-testing/tiny-random-T5ForConditionalGeneration'
    ... )
    >>> metric = rl.metrics.AnswerCitationRecall(nli_model=nli_model)
    >>> metric.mtype
    'AnswerGroundedness'
    >>> score, results = metric.compute(dataset['answers'], dataset['contexts'], 1)
"""

_CITATION = """\
@misc{gao2023enabling,
      title={Enabling Large Language Models to Generate Text with Citations},
      author={Tianyu Gao and Howard Yen and Jiatong Yu and Danqi Chen},
      year={2023},
      eprint={2305.14627},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


@dataclass
@add_attribute('mtype', 'AnswerGroundedness')
@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class AnswerCitationRecall(Metric):
    """Estimates the citation recall of the generated answer based on the NLI model."""

    name = "answer_citation_recall"

    ALIAS = ['answer_citation_recall']

    def __init__(self, nli_model: Callable):
        """
        Explicitly initialize AnswerCitationRecall.

        Ensure all parent classes are initialized.
        Ensure nli_model is initialized.
        """
        super().__init__()
        self.nli_model = nli_model

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            inputs_description=_KWARGS_DESCRIPTION,
            citation=_CITATION,
            homepage="",
            features=datasets.Features(
                {
                    "answers": datasets.Value("string"),
                    "contexts": datasets.Sequence(datasets.Value("string"))
                }
            ),
            codebase_urls=["https://github.com/princeton-nlp/ALCE"],
            reference_urls=["https://arxiv.org/abs/2305.14627"]
        )

    def _compute_one(
        self,
        answer: str,
        context: List[str]
    ) -> float:
        """Evaluate the citation recall of an answer."""
        total_entail = 0

        sents = text_to_sents(answer)
        target_sents = [remove_citations(sent).strip() for sent in sents]

        for idx, sent in enumerate(sents):
            target_sent = target_sents[idx]

            context_ids = []
            for r in re.findall(r"\[\d+", sent):
                context_id = int(r[1:])
                if 1 <= context_id <= len(context):
                    context_ids.append(context_id)
                else:
                    context_ids = []
                    break

            if len(context_ids) > 0:
                # citation id starts from 1 in sents
                premise = " ".join([context[context_id - 1] for context_id in context_ids])
                label = self.nli_model.generate_infer(premise=premise, hypothesis=target_sent)
                total_entail += label

        if len(sents) == 0:
            return 0
        return total_entail / len(sents)

    def _compute_batch(
        self,
        answers: List[str],
        contexts: List[List[str]]
    ) -> List[float]:
        """
        Evaluate the citation recall of a batch of answers.

        Firstly, calculate the citation recall of each statement (0 or 1).
        For each statement, its citation recall is 1 if and only if there is at least one citation and connects all
        paragraphs cited by this statement as premise, statement as hypothesis, and the NLI model outputs 1
        when it determines that premise entails the hypothesis, otherwise it is 0.
        Then, average over all statements in the LLM answer.
        Finally, average over all scores of each answer.
        """
        return super()._compute_batch(pred_answers=answers, ref_answers=contexts)
