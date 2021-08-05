# Loading script for the ViquiQuAD dataset.
import json
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
            """

_DESCRIPTION = """
               """

_HOMEPAGE = """https://zenodo.org/record/4562345#.YK41aqGxWUk"""

_URL = "../data/qa/"
_TRAINING_FILE = "train.json"
_DEV_FILE = "dev.json"
_TEST_FILE = "xquad.es.json"


class ViquiQuADConfig(datasets.BuilderConfig):
    """ Builder config for the XQUAD-es dataset """

    def __init__(self, **kwargs):
        """BuilderConfig for XQUAD-es.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ViquiQuADConfig, self).__init__(**kwargs)


class ViquiQuAD(datasets.GeneratorBasedBuilder):
    """XQUAD-es Dataset."""

    BUILDER_CONFIGS = [
        ViquiQuADConfig(
            name="XQUAD-es",
            #version=datasets.Version("1.0.1"),
            description="XQUAD-es dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "dev": f"{_URL}{_TEST_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            viquiquad = json.load(f, encoding="utf-8")
            for article in viquiquad["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
