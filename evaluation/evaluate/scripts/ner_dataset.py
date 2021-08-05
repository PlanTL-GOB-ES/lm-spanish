# Loading script for the Ancora NER dataset. 
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """ """

_DESCRIPTION = """
               """

_HOMEPAGE = """None"""

_URL = "../data/ner_capitel/"
_TRAINING_FILE = "train.tsv"
_DEV_FILE = "val.tsv"
_TEST_FILE = "gold-standard.tsv"


class NERConfig(datasets.BuilderConfig):
    """ Builder config for the Ancora Ca NER dataset """

    def __init__(self, **kwargs):
        """BuilderConfig for AncoraCaNer.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NERConfig, self).__init__(**kwargs)


class NER(datasets.GeneratorBasedBuilder):
    """ AncoraCaNer dataset."""

    BUILDER_CONFIGS = [
        NERConfig(
            name="AncoraCaNer",
            version=datasets.Version("2.0.0"),
            description="AncoraCaNer dataset"
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'B-OTH',
                                'I-OTH',
                                'E-OTH',
                                'S-ORG',
                                'S-OTH',
                                'B-PER',
                                'E-PER',
                                'B-ORG',
                                'E-ORG',
                                'S-LOC',
                                'S-PER',
                                'B-LOC',
                                'E-LOC',
                                'I-PER',
                                'I-ORG',
                                'I-LOC',
                                'O'
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
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # AncoraCaNer tokens are space separated
                    splits = line.split('\t')
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
