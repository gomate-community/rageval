# import sys
# sys.path.insert(0, '../src')

import pytest

from rageval.models import NLIModel


@pytest.fixture(scope='module')
def test_case():
    sample = {
            "claim": "In 1980, the oldest justice on the United States Supreme Court was Justice William O. Douglas.",
            "evidence": "August 3, 1994 \u2013 June 30, 2022 (27 years, 10 months, 27 days) photo source: Wikimedia "
                        "Commons After the passing of Ruth Bader Ginsberg in 2020, Stephen Breyer was the oldest "
                        "sitting member of the Supreme Court until his retirement in 2022. Stepping down at the age "
                        "of 83, Breyer is now one of the oldest Supreme Court justices ever. Breyer was nominated by "
                        "Bill Clinton and served on the Court for more than 27 years. During his tenure, Breyer fell "
                        "in line with the liberal wing of the court. Before he was appointed to the Supreme Court, "
                        "Breyer served as a judge on the U.S. Court of Appeals for the First Circuit; he was the "
                        "Chief Judge for the last four years of his appointment.",
            "stance": "irrelevant"
            }
    return sample


@pytest.mark.slow
def test_nli(test_case):

    # model = NLIModel('sentiment-analysis', 'roberta-large-mnli')
    model = NLIModel(
        'text-classification',
        'hf-internal-testing/tiny-random-RobertaPreLayerNormForSequenceClassification'
    )

    # test request
    result = model.infer_prob(test_case['evidence'], test_case['claim'])
    # print(result)
    assert result[0]['label'] in ['LABEL_0', 'LABEL_1']
    assert 'score' in result[0]

# case = test_case()
# test_nli(case)
