import pytest
from datasets import Dataset

from rageval.models import NLIModel
from rageval.metrics import AnswerCitationPrecision


@pytest.fixture(scope='module')
def sample():
    test_case = {
        "answers": [
            "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average "
            "annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an "
            "annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by Mawsynram, India "
            "with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, India, also known as "
            "Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year "
            "from August 1860 to July 1861 [1]."
        ],
        "contexts": [
            [
                "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be "
                "spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in "
                "the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal "
                "chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been "
                "credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that "
                "distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month "
                "for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in",
                "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji "
                "Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled "
                "Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian "
                "state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship "
                "constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as "
                "being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. "
                "Cherrapunji still holds the all-time record for the most rainfall",
                "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in "
                "north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls "
                "in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, "
                "but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm "
                "between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per "
                "year between 1960 and 2012. According to the \"Guinness Book of World Records\", Mawsynram received "
                "of rainfall in 1985. Mawsynram is located at 25° 18′"
            ]
        ]
    }
    return test_case


@pytest.fixture(scope='module')
def testset(sample):
    ds = Dataset.from_dict(sample)
    return ds


@pytest.mark.slow
def test_answer_citation_recall(testset):
    nli_model = NLIModel(
        'text2text-generation',
        'hf-internal-testing/tiny-random-T5ForConditionalGeneration'
    )
    metric = AnswerCitationPrecision(nli_model=nli_model)
    assert metric.name == "answer_citation_precision"
    assert metric.mtype == 'AnswerGroundedness'
    score, results = metric.compute(testset, 1)
    assert 0 <= score <= 1
    assert isinstance(results, Dataset)
