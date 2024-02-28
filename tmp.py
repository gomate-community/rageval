# _DESCRIPTION = """\
# The AnswerClaimRecall is to measure the correctness of long-form answers. In the original paper, the author first use \
# Instruct-GPT(text-davinci-003) to generate three "sub-claims" (based on gold answers) and use a state-of-the-art \
# natural-language inference (NLI) model TRUE(Honovich et al., 2022) to check whether the model output entails the \
# sub-claims (claim recall).
#
# For details, see the paper: http://arxiv.org/abs/2305.14627.
# """
#
# print(_DESCRIPTION)

from datasets import Dataset
import rageval as rl
sample = {"answers": ["test answer"], "gt_answers": ["test context"]}
dataset = Dataset.from_dict(sample)


# def add_prefix(example):
#     example["gt_answers"] = example["gt_answers"].split()
#     return example
#
# print(type(dataset["gt_answers"][0]))
# dataset = dataset.map(add_prefix)
# print(type(dataset["gt_answers"][0]))

model = rl.models.NLIModel('text-classification',
                                'hf-internal-testing/tiny-random-RobertaPreLayerNormForSequenceClassification')
metric = rl.metrics.AnswerClaimRecall()
metric.init_model(model)
s, ds = metric.compute(dataset, batch_size=1)
print(s)