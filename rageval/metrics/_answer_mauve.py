import os

import datasets

import mauve
from rageval.metrics import Metric, add_attribute

_CITATION = """\
@misc{MAUVE2023,
    title={MAUVE: Multi-Model Aggregated User Validation for Evaluation},
    author={Bloomberg Finance L.P.},
    year={2023},
    url={URL to be added}
}
"""


@add_attribute('mtype', 'EvaluationMetric')
class MAUVE(Metric):
    """MAUVE: Multi-Model Aggregated User Validation for Evaluation."""

    name = "mauve"

    ALIAS = ['mauve']

    def __init__(self, config):
        super().__init__()
        self.config = config

    def __repr__(self) -> str:
        """:return: Formatted string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def _info(self):
        return datasets.MetricInfo(
            description="MAUVE: Multi-Model Aggregated User Validation for Evaluation",
            citation=_CITATION,
            features=datasets.Features(
                {
                    "human": datasets.Value("string"),
                    "other_models": datasets.Value("string"),
                }
            ),
            reference_urls=["URL to be added"]
        )

    def compute(
        self,
        data,
        batch_size,
        max_text_length,
        device,
        featurize_model_name="gpt2-large-mauve"
    ):
        if not os.path.isdir(featurize_model_name):
            print(
                f"ERROR: please get {featurize_model_name} first following the instruction in README"
            )
            exit()

        device_id = device.index
        p_features = mauve.get_features_from_input(
            None,
            None,
            data["human"],
            featurize_model_name=featurize_model_name,
            max_len=max_text_length,
            device_id=device_id,
            name="p",
            verbose=True,
            batch_size=batch_size,
            use_float64=False,
        )
        scores = {}
        for model_name in data:
            if model_name != "human":
                out = mauve.compute_mauve(
                    p_features=p_features,
                    q_text=data[model_name],
                    max_text_length=max_text_length,
                    verbose=False,
                    device_id=device_id,
                    batch_size=batch_size,
                    featurize_model_name=featurize_model_name,
                )
                scores[f"{model_name}_mauve"] = out.mauve
        return scores
