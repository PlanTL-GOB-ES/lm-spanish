# Loading script for the POS dataset.
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """ """

_DESCRIPTION = """
               """

_HOMEPAGE = """None"""

_URL = "../data/pos_capitel/"
_TRAINING_FILE = "train.conllu"
_DEV_FILE = "val.conllu"
_TEST_FILE = "gold-standard.conllu"


class NERConfig(datasets.BuilderConfig):
    """ Builder config for the POS dataset """

    def __init__(self, **kwargs):
        """BuilderConfig for POS.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NERConfig, self).__init__(**kwargs)


class NER(datasets.GeneratorBasedBuilder):
    """ POS dataset."""

    BUILDER_CONFIGS = [
        NERConfig(
            name="POS",
            version=datasets.Version("2.0.0"),
            description="POS dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'NUM', 'VERB', 'X', 'ADV', 'PRON', 'INTJ', 'ADJ', 'PUNCT', 'ADP', 'SCONJ', 'NOUN',
                                'PROPN', 'CCONJ', 'DET'
                            ]
                        )
                    ),
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
            tokens = []
            pos_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n" or line.startswith('#'):
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "pos_tags": pos_tags,
                        }
                        guid += 1
                        tokens = []
                        pos_tags = []
                else:
                    # tokens are space separated
                    splits = line.split('\t')
                    tokens.append(splits[1])
                    pos_tags.append(splits[3].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "pos_tags": pos_tags,
            }
