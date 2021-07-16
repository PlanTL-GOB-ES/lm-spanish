# Spanish Language Models 💃🏻

## Corpora 📃

| Corpora | Number of documents | Size (GB) |
|---------|---------------------|-----------|
| BNE     |         201,080,084 |     570GB |

## Models 🤖
- RoBERTa-base BNE: https://huggingface.co/BSC-TeMU/roberta-base-bne
- RoBERTa-large BNE: https://huggingface.co/BSC-TeMU/roberta-large-bne 
- Other models: _(WIP)_

## Word embeddings 🔤
Word embeddings trained with FastText for 300d:
- CBOW Word embeddings: https://zenodo.org/record/5044988
- Skip-gram Word embeddings: https://zenodo.org/record/5046525

## Evaluation ✅
| Dataset     | Metric   | RoBERTa-b | RoBERTa-l | BETO   | mBERT  | BERTIN |
|-------------|----------|-----------|-----------|--------|--------|--------|
| UD-POS      | F1       |    0.9907 |    0.9901 | 0.9900 | 0.9886 | 0.9904 |
| Conll-NER   | F1       |    0.8851 |    0.8772 | 0.8759 | 0.8691 | 0.8627 |
| Capitel-POS | F1       |    0.9846 |    0.9851 | 0.9836 | 0.9839 | 0.9826 |
| Capitel-NER | F1       |    0.8959 |    0.8998 | 0.8771 | 0.8810 | 0.8741 |
| STS         | Combined |    0.8423 |    0.8420 | 0.8216 | 0.8249 | 0.7822 |
| MLDoc       | Accuracy |    0.9595 |    0.9600 | 0.9650 | 0.9560 | 0.9673 |
| PAWS-X      | F1       |    0.9035 |    0.9000 | 0.8915 | 0.9020 | 0.8820 |
| XNLI        | Accuracy |    0.8016 |       WiP | 0.8130 | 0.7876 |    WiP |

## Usage example ⚗️
For the RoBERTa-base
```python
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
from pprint import pprint
tokenizer_hf = AutoTokenizer.from_pretrained('BSC-TeMU/roberta-base-bne')
model = AutoModelForMaskedLM.from_pretrained('BSC-TeMU/roberta-base-bne')
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
tokenizer_hf = AutoTokenizer.from_pretrained('BSC-TeMU/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('BSC-TeMU/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)
text = f"¡Hola <mask>!"
res_hf = pipeline(text)
pprint([r['token_str'] for r in res_hf])
```

## Other Spanish Language Models 👩‍👧‍👦
We are developing domain-specific language models:

- [Legal Language Model](https://github.com/PlanTL-SANIDAD/lm-legal-es)

## Cite 📣
```
@misc{gutiérrezfandiño2021spanish,
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
