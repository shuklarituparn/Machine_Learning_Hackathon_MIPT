import pandas as pd
import numpy as np
from datasets import load_dataset
import json
from sklearn.model_selection import train_test_split
import os
from sklearn.utils import shuffle

def CreateNecessaryDataset():
    ds = load_dataset("Cohere/miracl-ru-queries-22-12")
    train = ds["train"]
    dev = ds["dev"]
    train_queries = train["query"]
    n = len(train_queries)
    train_positive_passages = train["positive_passages"]
    train_negative_passages = train["negative_passages"]
    train_df = pd.DataFrame(columns=['question', 'answer', 'isRelevant'])
    for i in range(n):
        positive_answers = [x["text"] for x in train_positive_passages[i]]
        negative_answers = [x["text"] for x in train_negative_passages[i]]

        positive_data = pd.DataFrame({
            'question': [train_queries[i]] * len(positive_answers),
            'answer': positive_answers,
            'isRelevant': [1] * len(positive_answers)
        })

        negative_data = pd.DataFrame({
            'question': [train_queries[i]] * len(negative_answers),
            'answer': negative_answers,
            'isRelevant': [0] * len(negative_answers)
        })

        train_df = pd.concat([train_df, positive_data, negative_data], ignore_index=True)
    val_positive_passages = dev["positive_passages"]
    val_negative_passages = dev["negative_passages"]
    val_queries = dev["query"]
    m = len(val_queries)
    val_df = pd.DataFrame(columns=['question', 'answer', 'isRelevant'])
    for i in range(m):
        positive_answers = [x["text"] for x in val_positive_passages[i]]
        negative_answers = [x["text"] for x in val_negative_passages[i]]

        positive_data = pd.DataFrame({
            'question': [val_queries[i]] * len(positive_answers),
            'answer': positive_answers,
            'isRelevant': [1] * len(positive_answers)
        })

        negative_data = pd.DataFrame({
            'question': [val_queries[i]] * len(negative_answers),
            'answer': negative_answers,
            'isRelevant': [0] * len(negative_answers)
        })

        val_df = pd.concat([val_df, positive_data, negative_data], ignore_index=True)
    return (train_df, val_df)

def AddMiraclEN(train_df):
    # Загружаем английский MIRACL датасет
    ds_en = load_dataset("Cohere/miracl-en-queries-22-12")
    train_en = ds_en["train"]
    train_en_queries = train_en["query"]
    
    train_en_positive_passages = train_en["positive_passages"]
    train_en_negative_passages = train_en["negative_passages"]
    
    # Создаем DataFrame для английских данных
    train_en_df = pd.DataFrame(columns=['question', 'answer', 'isRelevant'])
    n_en = len(train_en_queries)
    
    # Заполняем DataFrame английскими данными
    for i in range(n_en):
        positive_answers_en = [x["text"] for x in train_en_positive_passages[i]]
        negative_answers_en = [x["text"] for x in train_en_negative_passages[i]]
    
        positive_data_en = pd.DataFrame({
            'question': [train_en_queries[i]] * len(positive_answers_en),
            'answer': positive_answers_en,
            'isRelevant': [1] * len(positive_answers_en)
        })

        negative_data_en = pd.DataFrame({
            'question': [train_en_queries[i]] * len(negative_answers_en),
            'answer': negative_answers_en,
            'isRelevant': [0] * len(negative_answers_en)
        })

        train_en_df = pd.concat([train_en_df, positive_data_en, negative_data_en], ignore_index=True)

    # Применяем undersampling для нерелевантных ответов в английском датасете
    num_relevant_en = len(train_en_df[train_en_df["isRelevant"] == 1])
    train_en_irrelevant_df = train_en_df[train_en_df["isRelevant"] == 0].sample(num_relevant_en, random_state=42)
    train_en_relevant_df = train_en_df[train_en_df["isRelevant"] == 1]

    # Объединяем релевантные и undersampled нерелевантные данные английского датасета
    train_en_balanced_df = pd.concat([train_en_relevant_df, train_en_irrelevant_df], ignore_index=True)

    # Объединяем англоязычный и русскоязычный тренировочные наборы данных
    return shuffle(pd.concat([train_df, train_en_balanced_df], ignore_index=True))

def SaveDF(train_df=pd.DataFrame(), val_df=pd.DataFrame(), test_df=pd.DataFrame()):
    os.makedirs('UsedDatasets', exist_ok=True)
    if not(train_df.empty):
        train_df.to_csv('UsedDatasets/train_df.csv')
    if not(val_df.empty):
        val_df.to_csv('UsedDatasets/val_df.csv')
    if not(test_df.empty):
        test_df.to_csv('UsedDatasets/test_df.csv')

def LoadDF(train='', val='', test=''):
    dfs=[]
    if train:
        train_df=pd.read_csv(f'UsedDatasets/{train}')
        dfs.append(train_df)
    if val:
        val_df=pd.read_csv(f'UsedDatasets/{val}')
        dfs.append(val_df)
    if test:
        test_df=pd.read_csv(f'UsedDatasets/{test}')
        dfs.append(test_df)
    dfs=(i for i in dfs)
    return dfs

def SplitIntoTest(train_df, val_df, test_ratio = 0.1):
    # Разделяем train на train и test
    train_df, test_1_df = train_test_split(train_df, test_size=test_ratio, random_state=42)
    # Dev остается без изменений
    val_df, test_2_df = train_test_split(val_df, test_size=test_ratio, random_state=42)
    # Объединяем тестовые выборки
    test_df = pd.concat([test_1_df, test_2_df])
    return (train_df, val_df, test_df)
