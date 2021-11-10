# Spanish Language Models 💃🏻

A repository part of the MarIA project.

## Corpora 📃

| Corpora | Number of documents | Number of tokens | Size (GB) |
|---------|---------------------|------------------|-----------|
| BNE     |         201,080,084 |  135,733,450,668 |     570GB |

## Models 🤖
- RoBERTa-base BNE: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne
- RoBERTa-large BNE: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne 
- GPT2-base BNE: https://huggingface.co/PlanTL-GOB-ES/gpt2-base-bne
- GPT2-large BNE: https://huggingface.co/PlanTL-GOB-ES/gpt2-large-bne 
- Other models: _(WIP)_

## Fine-tunned models 🧗🏼‍♀️🏇🏼🤽🏼‍♀️🏌🏼‍♂️🏄🏼‍♀️

- RoBERTa-base-BNE for Capitel-POS: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-pos
- RoBERTa-large-BNE for Capitel-POS: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-pos
- RoBERTa-base-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner
- RoBERTa-base-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus (**very robust**)
- RoBERTa-large-BNE for Capitel-NER: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-ner
- RoBERTa-base-BNE for SQAC: https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne-sqac
- RoBERTa-large-BNE for SQAC: https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-sqac

## Word embeddings 🔤
Word embeddings trained with FastText for 300d:
- CBOW Word embeddings: https://zenodo.org/record/5044988
- Skip-gram Word embeddings: https://zenodo.org/record/5046525

## Datasets 🗂️

- Spanish Question Answering Corpus (SQAC)🦆: https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC

## Evaluation ✅
| Dataset     | Metric   | [RoBERTa-b](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne) | [RoBERTa-l](https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne) | [BETO](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)*   | [mBERT](https://huggingface.co/bert-base-multilingual-cased)  | [BERTIN](https://huggingface.co/bertin-project/bertin-roberta-base-spanish/tree/v1-512)** | [Electricidad](https://huggingface.co/mrm8488/electricidad-base-generator)*** |
|-------------|----------|-----------|-----------|--------|--------|--------|---------|
| UD-POS      | F1       | 0.9907    | 0.9898    | 0.9900 | 0.9886 | 0.9898 | 0.9818  |
| Conll-NER   | F1       | 0.8851    | 0.8772    | 0.8759 | 0.8691 | 0.8835 | 0.7954  |
| Capitel-POS | F1       | 0.9846    | 0.9851    | 0.9836 | 0.9839 | 0.9847 | 0.9816  |
| Capitel-NER | F1       | 0.8960    | 0.8998    | 0.8772 | 0.8810 | 0.8856 | 0.8035  |
| STS         | Combined | 0.8533    | 0.8353    | 0.8159 | 0.8164 | 0.7945 | 0.8063  |
| MLDoc       | Accuracy | 0.9623    | 0.9675    | 0.9663 | 0.9550 | 0.9673 | 0.9493  |
| PAWS-X      | F1       | 0.9000    | 0.9060    | 0.9000 | 0.8955 | 0.8990 | 0.9025  |
| XNLI        | Accuracy | 0.8016    | 0.7958    | 0.8130 | 0.7876 | 0.7890 | 0.7878  |
| SQAC        | F1       | 0.7923    | 0.7993    | 0.7923 | 0.7562 | 0.7678 | 0.7383  |

_* A model based on BERT architecture._

_** A model based on RoBERTa architecture._

_*** A model based on Electra architecture._


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

## Other Spanish Language Models 👩‍👧‍👦
We are developing domain-specific language models:

- ⚖️ [Legal Language Model](https://github.com/PlanTL-GOB-ES/lm-legal-es)
- ⚕️ [Biomedical and Clinical Language Models](https://github.com/PlanTL-GOB-ES/lm-biomedical-clinical-es) 

## Cite 📣
```
@misc{gutierrezfandino2021spanish,
      title={Spanish Language Models}, 
      author={Asier Gutiérrez-Fandiño and Jordi Armengol-Estapé and Marc Pàmies and Joan Llop-Palao and Joaquín Silveira-Ocampo and Casimiro Pio Carrino and Aitor Gonzalez-Agirre and Carme Armentano-Oller and Carlos Rodriguez-Penagos and Marta Villegas},
      year={2021},
      eprint={2107.07253},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact 📧
📋 We are interested in (1) extending our corpora to make larger models (2) train/evaluate the model in other tasks.

For questions regarding this work, contact Asier Gutiérrez-Fandiño (asier.gutierrez@bsc.es)
