# Spanish Language Models ππ»

A repository part of the MarIA project.

## Corpora π

| Corpora | Number of documents | Number of tokens | Size (GB) |
|---------|---------------------|------------------|-----------|
| BNE     |         201,080,084 |  135,733,450,668 |     570GB |

## Models π€
- RoBERTa-base BNE: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne
- RoBERTa-large BNE: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne 
- GPT2-base BNE: https://huggingface.co/PlanTL-GOB-ES/gpt2-base-bne
- GPT2-large BNE: https://huggingface.co/PlanTL-GOB-ES/gpt2-large-bne 
- Other models: _(WIP)_

## Fine-tunned models π§πΌββοΈππΌπ€½πΌββοΈππΌββοΈππΌββοΈ

- RoBERTa-base-BNE for Capitel-POS: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-pos
- RoBERTa-large-BNE for Capitel-POS: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-pos
- RoBERTa-base-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner
- RoBERTa-base-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus (**very robust**)
- RoBERTa-large-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-ner
- RoBERTa-base-BNE for SQAC: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-sqac
- RoBERTa-large-BNE for SQAC: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-sqac

## Word embeddings π€
Word embeddings trained with FastText for 300d:
- CBOW Word embeddings: https://zenodo.org/record/5044988
- Skip-gram Word embeddings: https://zenodo.org/record/5046525

## Datasets ποΈ

- Spanish Question Answering Corpus (SQAC)π¦: https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC

## Evaluation β
| Dataset      | Metric   | [**RoBERTa-b**](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne)   | [RoBERTa-l](https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne) | [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)*   | [mBERT](https://huggingface.co/bert-base-multilingual-cased)  | [BERTIN](https://huggingface.co/bertin-project/bertin-roberta-base-spanish/tree/v1-512)** | [Electricidad](https://huggingface.co/mrm8488/electricidad-base-generator)*** |
|--------------|----------|------------|------------|------------|--------|--------|---------|
| MLDoc        | F1       |     0.9664 |     0.9702 | **0.9714**π₯ | 0.9617 | 0.9668 |  0.9565 |
| CoNLL-NERC   | F1       | **0.8851**π₯ |     0.8823 |     0.8759 | 0.8691 | 0.8835 |  0.7954 |
| CAPITEL-NERC | F1       |     0.8960 | **0.9051**π₯ |     0.8772 | 0.8810 | 0.8856 |  0.8035 |
| PAWS-X       | F1       |     0.9020 | **0.9150**π₯ |     0.8930 | 0.9000 | 0.8965 |  0.9045 |
| UD-POS       | F1       | **0.9907**π₯ |     0.9904 |     0.9900 | 0.9886 | 0.9898 |  0.9818 |
| CAPITEL-POS  | F1       |     0.9846 | **0.9856**π₯ |     0.9836 | 0.9839 | 0.9847 |  0.9816 |
| SQAC         | F1       |     0.7923 | **0.8202**π₯ |     0.7923 | 0.7562 | 0.7678 |  0.7383 |
| STS          | Combined |     **0.8533**π₯ |     0.8411 |     0.8159 | 0.8164 | 0.7945 |  0.8063 |
| XNLI         | Accuracy |     0.8016 | **0.8263**π₯ |     0.8130 | 0.7876 | 0.7890 |  0.7878 |

_* A model based on BERT architecture._

_** A model based on RoBERTa architecture._

_*** A model based on Electra architecture._


## Usage example βοΈ
For the RoBERTa-base
```python
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)
text = f"Β‘Hola <mask>!"
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
text = f"Β‘Hola <mask>!"
res_hf = pipeline(text)
pprint([r['token_str'] for r in res_hf])
```

## Other Spanish Language Models π©βπ§βπ¦
We are developing domain-specific language models:

- βοΈ [Legal Language Model](https://github.com/PlanTL-GOB-ES/lm-legal-es)
- βοΈ [Biomedical and Clinical Language Models](https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es) 

## Cite π£
```
@article{gutierrezfandino2022,
	author = {Asier GutiΓ©rrez-FandiΓ±o and Jordi Armengol-EstapΓ© and Marc PΓ mies and Joan Llop-Palao and Joaquin Silveira-Ocampo and Casimiro Pio Carrino and Carme Armentano-Oller and Carlos Rodriguez-Penagos and Aitor Gonzalez-Agirre and Marta Villegas},
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

## Contact π§
π We are interested in (1) extending our corpora to make larger models (2) train/evaluate the model in other tasks.

For questions regarding this work, contact <plantl-gob-es@bsc.es>
