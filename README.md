# Spanish Language Models 💃🏻

A repository part of the MarIA project.


## Corpora 📃

| Corpora | Number of documents | Number of tokens | Size (GB) |
|---------|---------------------|------------------|-----------|
| BNE     |         201,080,084 |  135,733,450,668 |     570GB |


## Models 🤖
- ✨ <b>new</b> ✨ Ǎguila-7B: https://huggingface.co/projecte-aina/aguila-7b

  **Ǎguila-7B** Ǎguila is a 7B parameters LLM that has been trained on a mixture of Spanish, Catalan and English data, adding up to a total of 26B tokens. It uses the [Falcon-7b](https://huggingface.co/tiiuae/falcon-7b) model as a starting point, a state-of-the-art English language model that was openly released just a few months ago by the Technology Innovation Institute. Read more [here](https://medium.com/@mpamies247/introducing-a%CC%8Cguila-a-new-open-source-llm-for-spanish-and-catalan-ee1ebc70bc79)

- RoBERTa-base BNE: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne
- RoBERTa-large BNE: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne
- longformer-base-4096-bne-es: https://huggingface.co/PlanTL-GOB-ES/longformer-base-4096-bne-es
- GPT2-base BNE: https://huggingface.co/PlanTL-GOB-ES/gpt2-base-bne
- GPT2-large BNE: https://huggingface.co/PlanTL-GOB-ES/gpt2-large-bne


## Fine-tunned models 🧗🏼‍♀️🏇🏼🤽🏼‍♀️🏌🏼‍♂️🏄🏼‍♀️

- RoBERTa-base-BNE for Capitel-POS: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-pos
- RoBERTa-large-BNE for Capitel-POS: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-pos
- RoBERTa-base-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner
- RoBERTa-base-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus (**very robust**)
- RoBERTa-large-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-ner
- RoBERTa-base-BNE for SQAC: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-sqac
- RoBERTa-large-BNE for SQAC: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-sqac

For a complete list, refer to https://huggingface.co/PlanTL-GOB-ES

## Other Spanish Language Models 👩‍👧‍👦

Domain-specific language models:

- ⚖️ Legal Language Model: https://github.com/PlanTL-GOB-ES/lm-legal-es
- ⚕️ Biomedical and Clinical Language Models: https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es


## Word embeddings 🔤

- **Spanish CBOW Word Embeddings in Floret**, trained with the corpus from the National Library of Spain (Biblioteca Nacional de España or BNE) using floret: https://zenodo.org/record/7314098
- **Biomedical Spanish CBOW Word Embeddings in Floret**, trained with a biomedical Spanish corpus using floret: https://zenodo.org/record/7314041
- **Spanish Skip-Gram Word Embeddings in FastText**, trained with the corpus from the BNE: https://zenodo.org/record/5046525
- **Spanish Legal Domain Word & Sub-Word Embeddings**, trained with a Spanish Legal resources: https://zenodo.org/record/5036147


## Datasets 🗂️

- Spanish Question Answering Corpus (SQAC)🦆: https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC
- Spanish Semantic Text Similarity (STS-es): https://huggingface.co/datasets/PlanTL-GOB-ES/sts-es
- Professional translation into Spanish of Winograd NLI dataset (WNLI-es): https://huggingface.co/datasets/PlanTL-GOB-ES/wnli-es
- Spanish dataset of the CoNLL-2002 Shared Task (CoNLL-NERC): https://huggingface.co/datasets/PlanTL-GOB-ES/CoNLL-NERC-es
- Spanish corpus for thematic Text Classification tasks (WikiCAT_es): https://huggingface.co/datasets/PlanTL-GOB-ES/WikiCAT_esv2
- English corpus for thematic Text Classification tasks (WikiCAT_en): https://huggingface.co/datasets/PlanTL-GOB-ES/WikiCAT_en

For a complete list, refer to https://huggingface.co/PlanTL-GOB-ES


## EvalES: The Spanish Evaluation Benchmark

The EvalES benchmark consists of 8 tasks: Named Entity Recognition and Classification (CoNLL-NERC), Part-of-Speech Tagging (UD-POS), Text Classification (MLDoc), Paraphrase Identification (PAWS-X), Semantic Textual Similarity (STS), Question Answering (SQAC), Textual Entailment (XNLI) and Massive.


### Results ✅

| Dataset      | Metric   | [**RoBERTa-b**](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne)   | [RoBERTa-l](https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne) | [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)*   | [mBERT](https://huggingface.co/bert-base-multilingual-cased)  | [BERTIN](https://huggingface.co/bertin-project/bertin-roberta-base-spanish/tree/v1-512)** | [Electricidad](https://huggingface.co/mrm8488/electricidad-base-generator)*** |
|--------------|----------|------------|------------|------------|--------|--------|---------|
| MLDoc        | F1       |     0.9664 |     0.9702 | **0.9714**🔥 | 0.9617 | 0.9668 |  0.9565 |
| CoNLL-NERC   | F1       | **0.8851**🔥 |     0.8823 |     0.8759 | 0.8691 | 0.8835 |  0.7954 |
| CAPITEL-NERC | F1       |     0.8960 | **0.9051**🔥 |     0.8772 | 0.8810 | 0.8856 |  0.8035 |
| PAWS-X       | F1       |     0.9020 | **0.9150**🔥 |     0.8930 | 0.9000 | 0.8965 |  0.9045 |
| UD-POS       | F1       | **0.9907**🔥 |     0.9904 |     0.9900 | 0.9886 | 0.9898 |  0.9818 |
| CAPITEL-POS  | F1       |     0.9846 | **0.9856**🔥 |     0.9836 | 0.9839 | 0.9847 |  0.9816 |
| SQAC         | F1       |     0.7923 | **0.8202**🔥 |     0.7923 | 0.7562 | 0.7678 |  0.7383 |
| STS          | Combined |     **0.8533**🔥 |     0.8411 |     0.8159 | 0.8164 | 0.7945 |  0.8063 |
| XNLI         | Accuracy |     0.8016 | **0.8263**🔥 |     0.8130 | 0.7876 | 0.7890 |  0.7878 |
| Massive      | Accuracy |     0.8605 | 0.8722 |     **0.8732**🔥 | 0.8504 | 0.8500 |  0.8517 |

_* A model based on BERT architecture._

_** A model based on RoBERTa architecture._

_*** A model based on Electra architecture._

For more information, refer to https://benchmark.plantl.bsc.es/


## Usage example ⚗️

For the RoBERTa-base
```python
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)
text = f"¡Hola <mask>!"
res_hf = pipeline(text)
pprint([r['token_str'] for r in res_hf])
```

For the RoBERTa-large
```python
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)
text = f"¡Hola <mask>!"
res_hf = pipeline(text)
pprint([r['token_str'] for r in res_hf])
```


## Demos

- Anonimizador: Anonymizer for Spanish and Catalan user generated content in conversational systems: https://anonimizador.plantl.bsc.es/
- spaCy: Demo of the different spacy models for Spanish: https://spacy.plantl.bsc.es/
- QA: Question/Answer system in the Spanish Wikipedia based on models and datasets generated under PlanTL: https://qa.plantl.bsc.es/
- Traductor: Automatic translators between Spanish and Catalan and between Spanish and Galician: https://traductor.plantl.bsc.es/
- EvalES: Collection of resources for assessing natural language comprehension systems: https://benchmark.plantl.bsc.es/


## Cite 📣
```
@article{gutierrezfandino2022,
	author = {Asier Gutiérrez-Fandiño and Jordi Armengol-Estapé and Marc Pàmies and Joan Llop-Palao and Joaquin Silveira-Ocampo and Casimiro Pio Carrino and Carme Armentano-Oller and Carlos Rodriguez-Penagos and Aitor Gonzalez-Agirre and Marta Villegas},
	title = {MarIA: Spanish Language Models},
	journal = {Procesamiento del Lenguaje Natural},
	volume = {68},
	number = {0},
	year = {2022},
	issn = {1989-7553},
	url = {http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6405},
	pages = {39--60}
}
```


## Contact 📧
📋 We are interested in (1) extending our corpora to make larger models (2) train/evaluate the model in other tasks.

For questions regarding this work, contact <plantl-gob-es@bsc.es>


## Disclaimer

The models published in this repository are intended for a generalist purpose and are available to third parties. These models may have bias and/or any other undesirable distortions.

When third parties, deploy or provide systems and/or services to other parties using any of these models (or using systems based on these models) or become users of the models, they should note that it is their responsibility to mitigate the risks arising from their use and, in any event, to comply with applicable regulations, including regulations regarding the use of artificial intelligence.

In no event shall the owner of the models (SEDIA – State Secretariat for digitalization and artificial intelligence) nor the creator (BSC – Barcelona Supercomputing Center) be liable for any results arising from the use made by third parties of these models.


Los modelos publicados en este repositorio tienen una finalidad generalista y están a disposición de terceros. Estos modelos pueden tener sesgos y/u otro tipo de distorsiones indeseables.

Cuando terceros desplieguen o proporcionen sistemas y/o servicios a otras partes usando alguno de estos modelos (o utilizando sistemas basados en estos modelos) o se conviertan en usuarios de los modelos, deben tener en cuenta que es su responsabilidad mitigar los riesgos derivados de su uso y, en todo caso, cumplir con la normativa aplicable, incluyendo la normativa en materia de uso de inteligencia artificial.

En ningún caso el propietario de los modelos (SEDIA – Secretaría de Estado de Digitalización e Inteligencia Artificial) ni el creador (BSC – Barcelona Supercomputing Center) serán responsables de los resultados derivados del uso que hagan terceros de estos modelos.

