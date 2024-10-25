import torch
from transformers import BertTokenizer, BertModel

class FakeNewsBERTModel(torch.nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased'):
        super(FakeNewsBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        return self.fc(pooled_output) 

def predict(article):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = FakeNewsBERTModel()
    model.load_state_dict(torch.load('ml_model/best_model_bert.pt'))
    model.eval()

    inputs = tokenizer(article, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.squeeze()
        prediction = torch.sigmoid(logits).item()
    
    return {'label': 1 if prediction >= 0.5 else 0, 'confidence': prediction}

"""Explanation: This file defines the model architecture using BERT and the prediction function that takes an article as input, processes it, and returns the prediction label and confidence."""