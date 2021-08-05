# Loading script for the Ancora NER dataset. 
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """ """

_DESCRIPTION = """
               """

_HOMEPAGE = """None"""

_URL = "../data/mldocs/"
_TRAINING_FILE = "train.tsv"
_DEV_FILE = "valid.tsv"
_TEST_FILE = "test.tsv"


class MLDocConfig(datasets.BuilderConfig):
    """ Builder config for the MLDocs dataset """

    def __init__(self, **kwargs):
        """BuilderConfig for MLDocs.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MLDocConfig, self).__init__(**kwargs)


class MLDoc(datasets.GeneratorBasedBuilder):
    """ MLDoc dataset."""

    BUILDER_CONFIGS = [
        MLDocConfig(
            name="MLDoc",
            version=datasets.Version("1.0.0"),
            description="MLDoc dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                            names=[
                                'MCAT', 'GCAT', 'ECAT', 'CCAT'
                            ])
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
                sentence = '\t'.join(line[1:])
                label = line[0]
                yield guid, {
                    "id": str(guid),
                    "sentence": sentence,
                    "label": label
                }
                guid += 1
