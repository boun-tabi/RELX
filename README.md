# RELX
The RELX Dataset and Matching the Multilingual Blanks for Cross-lingual Relation Classification, EMNLP-Findings 2020.

Paper: 


## RELX & RELX-Distant Datasets

KBP-37 (English): [Download](https://github.com/zhangdongxu/kbp37)

|            |      **RELX**       | **RELX-Distant Sample** | **RELX-Distant** |
|------------|---------------------|------------------------|-------------------|
| English    | [Download](Datasets/RELX/RELX_en.txt) | [Download](Datasets/RELX-Distant/RELX_distant_en.json)  | [Download](https://drive.google.com/file/d/1dWHW3D67V48b7HKYI7sUw67DKHPU4zTB/view?usp=sharing)           |
| French     | [Download](Datasets/RELX/RELX_fr.txt) | [Download](Datasets/RELX-Distant/RELX_distant_fr.json)  | [Download](https://drive.google.com/file/d/1rXnT0KHKI89extkEszP7UsDgbTKC7kvU/view?usp=sharing)            |
| German     | [Download](Datasets/RELX/RELX_de.txt) | [Download](Datasets/RELX-Distant/RELX_distant_de.json)  | [Download](https://drive.google.com/file/d/1vAZ4VqcpnSSyk7nZMrYUew70r_9azZ8x/view?usp=sharing)      |
| Spanish    | [Download](Datasets/RELX/RELX_es.txt) | [Download](Datasets/RELX-Distant/RELX_distant_es.json)  | [Download](https://drive.google.com/file/d/1MOzl7_h-7jVoryu9WEMRpzMcfDL8aPxe/view?usp=sharing)      |
| Turkish    | [Download](Datasets/RELX/RELX_tr.txt) | [Download](Datasets/RELX-Distant/RELX_distant_tr.json)  | [Download](https://drive.google.com/file/d/1op7xrrK7A5P4naf9TaS0Y09Sg6p8wgMh/view?usp=sharing)      |


## MTMB: Pretraining on RELX-Distant
We pretrained public checkpoint of Multilingual BERT with 20 million pairs of sentences from RELX-Distant (including English, French, German, Spanish, and, Turkish) with Masked Language Model (MLM) and Matching the Multilingual Blanks (MTMB) objectives.  

You can use pretrained MTMB model over MBERT from HuggingFace Model Hub:
```
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("akoksal/MTMB")
model = AutoModel.from_pretrained("akoksal/MTMB")
```
## Training KBP-37 / RELX
Check out finetune.py for more details of finetuning on KBP-37 and evaluating on RELX & test set of KBP-37.
## Results
|            |      **KBP-37 Dev**       | **KBP-37 Test** | **RELX-EN** | **RELX-FR** | **RELX-DE** | **RELX-ES** | **RELX-TR** |
|------------|---------------------|------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| MBERT          | 65.5 | 64.9 | 61.8 | 58.3 | 57.5 | 57.9 | 55.8 | 
| MBERT+MTMB     | **66.8** | **66.5** | **63.6** | **59.9** | **59.9** | **62.4** | **56.2** |

F1 scores of 10 runs. See paper for more details.

## Citation
* Please cite the following paper if you use any part of this work:

```
Paper details
```
