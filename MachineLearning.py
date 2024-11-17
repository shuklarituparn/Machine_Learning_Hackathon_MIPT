import torch
from torch import nn
import pandas as pd
from transformers import Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertConfig
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import os

class RelevantDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Переносим данные на GPU
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# кастомный Trainer для реализации взвешивания несбалансированных классов
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(logits.device)) # 1.0 и 3.0 - веса нерелевантных и релевантных ответов соответственно
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1]
    preds = (predictions > 0.5).int()

    # Рассчитываем метрику ROC AUC
    roc_auc = roc_auc_score(labels, predictions)

    # Другие метрики для удобства
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    return {
        'accuracy': acc,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def TrainingBertModel(train_df, val_df):
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    train_texts = train_df.copy()
    train_texts = train_df.sample(frac=1).reset_index(drop=True)
    valid_texts = val_df.copy()
    train_texts['text'] = train_df['question'] + " [SEP] " + train_df['answer']
    valid_texts['text'] = val_df['question'] + " [SEP] " + val_df['answer']
    train_texts = train_texts.drop(["question", "answer", "isRelevant"], axis=1)
    valid_texts = valid_texts.drop(["question", "answer", "isRelevant"], axis=1)

    train_labels = train_df["isRelevant"]
    valid_labels = val_df["isRelevant"]

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    device = torch.device("cuda")

    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    config = BertConfig.from_pretrained('cointegrated/rubert-tiny2', hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2, num_labels=2)
    
    model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2', config=config)

    train_encodings = tokenizer(train_texts["text"].tolist(), truncation=True, padding=True)
    valid_encodings = tokenizer(valid_texts["text"].tolist(), truncation=True, padding=True)
    print("Encodings were successfully completed")

    if torch.cuda.is_available(): model.to("cuda")

    train_dataset = RelevantDataset(train_encodings, train_labels.tolist())
    valid_dataset = RelevantDataset(valid_encodings, valid_labels.tolist())

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # увеличено число эпох (а потом снова уменьшено)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,  # зафиксированная скорость обучения
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="steps",  # оценка на каждом шаге
        eval_steps=500,
        lr_scheduler_type='cosine',
        load_best_model_at_end=True,
        metric_for_best_model="recall",  # используем метрику recall для выбора лучшей модели
        greater_is_better=True,
        save_strategy="steps",  # сохранение на каждом шаге
        save_steps=500,  # save_steps кратно eval_steps
        # report_to=None  # Disable wandb integration
    )

    # trainer = Trainer(
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics  # Передаем функцию метрик
    )

    trainer.train()

    return ((train_texts, train_labels, valid_texts, valid_labels), model)

def SaveModel(model: BertForSequenceClassification, ref):
    os.makedirs('Models', exist_ok=True)
    torch.save(model.state_dict(), f"Models/{ref}")

class Model():
    def __init__(self, ref):
        self.model2 = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2', num_labels=2)
        self.state_dict2 = torch.load(f'Models/{ref}')
        self.model2.load_state_dict(self.state_dict2)

    def predict2(self, question, answer):
        self.tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        inputs = self.tokenizer(f"{question}[SEP]{answer}", return_tensors='pt', truncation=True, padding=True)
        outputs = self.model2(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return 1 if predictions.item() == 1 else 0

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         # forward pass
#         outputs = model(**inputs)
#         logits = outputs.get('logits')
#         loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(logits.device)) # 1.0 - вес нуля, 4.0 - вес единички
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss

def Test1(valid_texts, valid_labels, M: Model):
    predictions=[]
    for text in tqdm(valid_texts["text"]):
        qu = text.split("[SEP]")[0]
        an = text.split("[SEP]")[1]
        predictions.append(M.predict2(qu, an))
    return (predictions, classification_report(valid_labels.tolist(), predictions),
            roc_auc_score(valid_labels.tolist(), predictions))

def Test2(test_df, M: Model):
    test_texts = test_df.copy()
    test_texts['text'] = test_df['question'] + " [SEP] " + test_df['answer']
    test_labels = test_df["isRelevant"]

    test_predictions = []

    for text in tqdm(test_texts["text"]):
        qu = text.split("[SEP]")[0]
        an = text.split("[SEP]")[1]
        test_predictions.append(M.predict2(qu, an))

    return (test_predictions, classification_report(test_labels.tolist(), test_predictions),
            roc_auc_score(test_labels.tolist(), test_predictions))
