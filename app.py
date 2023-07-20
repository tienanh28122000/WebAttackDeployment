import io
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.autograd import Function
import time

from flask import Flask, jsonify, request


app = Flask(__name__)

config = {
    "num_labels": 7,
    "hidden_dropout_prob": 0.15,
    "hidden_size": 768,
    "max_length": 512,
}

training_parameters = {
    "batch_size": 2,
    "epochs": 1,
    "output_folder": "model_weight",
    "output_file": "model.pt",
    "learning_rate": 2e-5,
    "print_after_steps": 100,
    "save_steps": 500,

}

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        
        num_labels = config["num_labels"]
        self.bert = AutoModel.from_pretrained('jackaduma/SecBERT')
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], num_labels),
            nn.LogSoftmax(dim=1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], 2),
            nn.LogSoftmax(dim=1),
        )


    def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          labels=None,
          grl_lambda = 1.0, 
          ):

        outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

#         pooled_output = outputs[1] # For bert-base-uncase
        pooled_output = outputs.pooler_output 
        pooled_output = self.dropout(pooled_output)


        reversed_pooled_output = GradientReversalFn.apply(pooled_output, grl_lambda)

        sentiment_pred = self.sentiment_classifier(pooled_output)
        domain_pred = self.domain_classifier(reversed_pooled_output)

        return sentiment_pred.to(device), domain_pred.to(device)

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')

    def __getitem__(self, index):
        review = self.df.iloc[index]["text"]
        sentiment = self.df.iloc[index]["label"]
        sentiment_dict = {'000 - Normal': 0,
          '126 - Path Traversal': 1,
          '242 - Code Injection': 2,
          '153 - Input Data Manipulation': 3,
          '310 - Scanning for Vulnerable Software': 4,
          '194 - Fake the Source of Data': 5,
          '34 - HTTP Response Splitting': 6}
        label = sentiment_dict[sentiment]
        encoded_input = self.tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length= config["max_length"],
                pad_to_max_length=True,
                return_overflowing_tokens=True,
            )
        if "num_truncated_tokens" in encoded_input and encoded_input["num_truncated_tokens"] > 0:
            # print("Attention! you are cropping tokens")
            pass

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        token_type_ids = encoded_input["token_type_ids"] if "token_type_ids" in encoded_input else None



        data_input = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "label": torch.tensor(label),
        }

        return data_input["input_ids"], data_input["attention_mask"], data_input["token_type_ids"], data_input["label"]



    def __len__(self):
        return self.df.shape[0]

def pred(model, dataset = "transfer", percentage = 0):
    all_pred = []
    with torch.no_grad():
        
        dev_df = pd.read_csv("dataset/dataset_capec_" + dataset + ".csv")
        data_size = dev_df.shape[0]
        if percentage == 0:
            dev_df = dev_df.head(1)
        else:
            selected_for_evaluation = int(data_size*percentage/100)
            dev_df = dev_df.head(selected_for_evaluation)
        dataset = ReviewDataset(dev_df)

        dataloader = DataLoader(dataset = dataset, batch_size = 2, shuffle = True, num_workers = 2)
        
        for input_ids, attention_mask, token_type_ids, labels in dataloader:
            inputs = {
                "input_ids": input_ids.squeeze(axis=1),
                "attention_mask": attention_mask.squeeze(axis=1),
                "token_type_ids" : token_type_ids.squeeze(axis=1),
                "labels": labels,
            }
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            sentiment_pred, _ = model(**inputs)
            pred_label = sentiment_pred.max(dim = 1)[1]
            for pred in pred_label:
                all_pred.append(pred.item())
    return all_pred



weight_path = "training/model_weight/e0model.pt"
model = torch.load(weight_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        start = time.time()
        request_uri = request.json.get('request_uri')
        pred_list = pred(model, dataset = "transfer", percentage = 0)
        end = time.time()
        print(pred_list)
        print((end - start) / len(pred_list))
        return jsonify({'class_id': '1', 'class_name': 'test'})


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5000)