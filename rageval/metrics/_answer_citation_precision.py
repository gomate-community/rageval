import copy
import re
from dataclasses import dataclass
from typing import List, Callable, Tuple

import datasets
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from rageval.metrics import Metric, add_attribute
from rageval.utils import text_to_sents, remove_citations

_DESCRIPTION = """\
Citation precision evaluation detects citations that are irrelevant to the claim, but it does not require citing \
a minimal set and it permits citing redundant passages entailing similar claims.

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
    _compute_one: compute the citation precision of an answer.

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
    >>> metric = rl.metrics.AnswerCitationPrecision(nli_model=nli_model)
    >>> metric.mtype
    'AnswerGroundedness'
    >>> s, ds = metric.compute(dataset, batch_size=1)
    >>> assert 0 <= s <= 1
    >>> type(ds)
    <class 'datasets.arrow_dataset.Dataset'>
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
class AnswerCitationPrecision(Metric):
    """Estimates the citation precision of the generated answer based on the NLI model."""

    name = "answer_citation_precision"

    ALIAS = ['answer_citation_precision']

    def __init__(self, nli_model: Callable):
        """
        Explicitly initialize AnswerCitationPrecision.

        Ensure all parent classes are initialized.
        Ensure nli_model is initialized.
        """
        super().__init__()
        self._required_columns = ['answers', 'contexts']
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
    ) -> Tuple[float, float]:
        """Evaluate the citation precision of an answer."""
        citation_correct = 0
        citation_total = 0

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
                label_full = self.nli_model.generate_infer(premise=premise, hypothesis=target_sent)
                if label_full == 1:
                    citation_total += len(context_ids)
                    for context_id in context_ids:
                        label_single = self.nli_model.generate_infer(premise=context[context_id - 1], hypothesis=target_sent)
                        if label_single == 1:
                            citation_correct += 1
                        else:
                            subset_context_id = copy.deepcopy(context_ids)
                            subset_context_id.remove(context_id)
                            subset_premise = " ".join([context[context_id - 1] for context_id in subset_context_id])
                            label_exclude = self.nli_model.generate_infer(premise=subset_premise, hypothesis=target_sent)
                            if label_exclude == 0:
                                citation_correct += 1

        return citation_correct, citation_total

    def _compute_batch(
        self,
        dataset: datasets.Dataset
    ) -> List[Tuple[float, float]]:
        """
        Evaluate the citation precision of a batch of answers.

        Firstly, calculate the citation precision of each statement (0 or 1).
        Precision check: did the model cite any unnecessary documents?
        Then, average over all statements in the LLM answer.
        Finally, average over all scores of each answer.
        """

        answers, contexts = (
            dataset["answers"],
            dataset["contexts"]
        )

        results = []
        for answer, context in tqdm(zip(answers, contexts)):
            citation_correct, citation_total = self._compute_one(answer, context)
            results.append((citation_correct, citation_total))
        return results

    def compute(
        self,
        dataset: Dataset,
        batch_size: int = None,
    ) -> Tuple[float, Dataset]:
        """Evaluate the dataset."""
        self._validate_data(dataset)
        scores = []

        length = len(dataset)
        if batch_size:
            for start in tqdm(range(0, length, batch_size)):
                end = start + batch_size
                end = end if end < length else length
                score = self._compute_batch(dataset.select(range(start, end)))
                scores.extend(score)
        else:
            scores = self._compute_batch(dataset)

        citation_correct = np.sum([correct for correct, total in scores])
        citation_total = np.sum([total for correct, total in scores])

        if citation_total == 0:
            return 0.0, dataset.add_column(f"{self.name}", scores)
        else:
            return citation_correct / citation_total, dataset.add_column(f"{self.name}", scores)
