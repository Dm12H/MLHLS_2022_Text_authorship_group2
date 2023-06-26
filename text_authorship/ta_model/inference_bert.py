from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
import pandas as pd


class InferenceBert:
    def __init__(self, bert_dir: str, text_col='text_w_no_ref'):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=False)
        self.bert: BertForSequenceClassification = (BertForSequenceClassification.from_pretrained(bert_dir)
                                                    .to(self.device))
        self.bert.eval()
        with open(f'{bert_dir}/labels.txt', 'r', encoding='utf-8') as f:
            self.labels = f.readlines()
        self.text_col = text_col
        
    @torch.inference_mode()    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        tokenized = self.tokenizer(df[self.text_col].tolist(),
                                   padding=True,
                                   truncation=True,
                                   max_length=512,
                                   return_tensors='pt')
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        output = self.bert(**tokenized)
        logits = output.get('logits')
        probs = softmax(logits, dim=1).detach().cpu().numpy()
        probs = pd.DataFrame(probs, columns=self.labels)
        return probs
    