# Loading script for the STS dataset.
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """ """

_DESCRIPTION = """
               """

_HOMEPAGE = """None"""

_URL = "../data/sts/"
_TRAINING_FILE = "train.tsv"
_DEV_FILE = "valid.tsv"
_TEST_FILE = "test.tsv"


class STSConfig(datasets.BuilderConfig):
    """ Builder config for the STS dataset """

    def __init__(self, **kwargs):
        """BuilderConfig for STS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(STSConfig, self).__init__(**kwargs)


class STS(datasets.GeneratorBasedBuilder):
    """ STS dataset."""

    BUILDER_CONFIGS = [
        STSConfig(
            name="STS",
            version=datasets.Version("1.0.0"),
            description="STS dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence1": datasets.Value("string"),
                    "sentence2": datasets.Value("string"),
                    "label": datasets.features.Value('float')
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            for line in f:
                line = line.split('\t')
                sentence1 = line[0]
                sentence2 = line[1]
                label = line[2]
                yield guid, {
                    "id": str(guid),
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "label": label
                }
                guid += 1
