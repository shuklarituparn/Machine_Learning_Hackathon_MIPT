import nltk
import random
from transformers import MarianTokenizer, MarianMTModel
from nltk.corpus import wordnet
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from tqdm import tqdm
import pandas as pd

nltk.download('wordnet')
nltk.download('omw-1.4')

# Инициализация модели перевода один раз
model_name_ru_en = 'Helsinki-NLP/opus-mt-ru-en'
model_name_en_ru = 'Helsinki-NLP/opus-mt-en-ru'

tokenizer_ru_en = MarianTokenizer.from_pretrained(model_name_ru_en)
model_ru_en = MarianMTModel.from_pretrained(model_name_ru_en).half().to('cuda') 

tokenizer_en_ru = MarianTokenizer.from_pretrained(model_name_en_ru)
model_en_ru = MarianMTModel.from_pretrained(model_name_en_ru).half().to('cuda')

# Обрабатываем перевод для батчей с прогресс-баром tqdm
def batch_translate(texts, model, tokenizer, batch_size=16):
    translated_texts = []
    loader = DataLoader(texts, batch_size=batch_size, shuffle=False)

    for batch in tqdm(loader, desc="Перевод батчей"):
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
        translated = model.generate(**inputs)
        translated_texts.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])

    return translated_texts

# Обрабатываем перевод для батчей
def batch_translate_to_english(texts, batch_size=16):
    return batch_translate(texts, model_ru_en, tokenizer_ru_en, batch_size)

def batch_translate_to_russian(texts, batch_size=16):
    return batch_translate(texts, model_en_ru, tokenizer_en_ru, batch_size)

# Функция для замены синонимов
def synonym_replacement(sentence, n=3):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)

    num_replacements = min(n, len(random_word_list))

    for random_word in random_word_list[:num_replacements]:
        synonyms = wordnet.synsets(random_word)
        synonym_words = [syn.lemmas()[0].name() for syn in synonyms if syn.lemmas()]
        if synonym_words:
            synonym = random.choice(synonym_words)
            new_words = [synonym if word == random_word else word for word in new_words]

    return ' '.join(new_words)
  
# Функция случайного удаления слов
def random_deletion(sentence, p=0.2):
    words = sentence.split()
    if len(words) == 1:
        return sentence
    new_words = [word for word in words if random.random() > p]
    if not new_words:
        return random.choice(words)
    return ' '.join(new_words)

# "Пустая" функция для обратного перевода (без других изменений)
def identity_augmentation(sentence):
    return sentence

# # Функция для аугментации одного примера
# def augment_example(question, answer):
#     # Переводим на английский для аугментации
#     translated_answer = translate_to_english(answer)

#     # Применяем одну из аугментаций (включая обратный перевод без изменений)
#     aug_methods = [synonym_replacement, random_deletion, identity_augmentation]
#     method = random.choice(aug_methods)
#     augmented_answer = method(translated_answer)

#     # Переводим обратно на русский
#     back_to_russian = translate_to_russian(augmented_answer)

#     return back_to_russian

# Аугментация релевантных ответов с использованием batch processing
def augment_relevant_answers_batch(df, batch_size=16):
    relevant_rows = df[df['isRelevant'] == 1]
    augmented_rows = []

    answers = relevant_rows['answer'].tolist()
    questions = relevant_rows['question'].tolist()

    print("Начинаем перевод на английский...")
    translated_answers = batch_translate_to_english(answers, batch_size)
    print("Перевод на английский завершен.")

    print("Начинаем аугментацию ответов...") # все манипуляции проводим с предложениями, переведенными на английский
    aug_methods = [synonym_replacement, random_deletion, identity_augmentation]
    augmented_answers = []

    for answer in tqdm(translated_answers, desc="Аугментация ответов"):
        augmented_answer = random.choice(aug_methods)(answer)
        augmented_answers.append(augmented_answer)

    print("Аугментация завершена. Начинаем перевод обратно на русский...")
    augmented_answers_russian = batch_translate_to_russian(augmented_answers, batch_size)

    for question, augmented_answer in zip(questions, augmented_answers_russian):
        new_row = {'question': question, 'answer': augmented_answer, 'isRelevant': 1}
        augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df_full = pd.concat([df, augmented_df], ignore_index=True)

    # Перемешиваем строки
    return shuffle(augmented_df_full)

# самая первая версия аугментаций (неактуальна)
# def apply_augmentations(df,
#                         symmetrize=True,
#                         speech_garbage=0,
#                         drop_symbol=0,
#                         drop_token=0,
#                         double_token=0,
#                         insert_random_symbol=0,
#                         swap_tokens=0,
#                         siblings=0):

#     cyrillic_letters = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
#     sibling_letters = { 'а': 'a','В': 'B', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'Т': 'T', 'у': 'y', 'х': 'x'}


#     augmentation_probas = [speech_garbage, drop_symbol, drop_token, double_token, insert_random_symbol, swap_tokens, siblings]

#     if sum(augmentation_probas) <= 0:
#         if symmetrize:
#             raise RuntimeError("To symmetrize the classes at least one augmentation must be applied")
#         else:
#             return df


#     df_0 = df[df["isRelevant"] == 0]
#     df_1 = df[df["isRelevant"] == 1]
#     new_rows = []
#     if symmetrize:
#         while df_0.shape[0] + len(new_rows) < df_1.shape[0]:
#             df_0 = df_0.drop_duplicates()
#             if speech_garbage > 0:
#                 if np.random.rand() < speech_garbage:
#                     new_row = df_0.sample().copy()
#                     text = new_row['question']
#                     text_split = text.iloc[0].split()
#                     if text_split:
#                         insert_index = np.random.randint(0, len(text_split) + 1)
#                         random_word = np.random.choice(["ээ", "мм", "ну", "кхм-кхм"])
#                         text_split.insert(insert_index, random_word)
#                     text = " ".join(text_split)
#                     new_row["question"] = text
#                     if text_split:
#                       if type(new_row["question"].iloc[0]) != type("aboba"):
#                           print(type(new_row["question"].iloc[0]), 1)
#                       new_rows.append(new_row)

#             if drop_symbol > 0:
#                 if np.random.rand() < drop_symbol:
#                     new_row = df_0.sample().copy()
#                     for _ in range (10):
#                         symbol_to_drop = np.random.choice(list(cyrillic_letters))
#                         new_row['answer'] = new_row['answer'].str.replace(symbol_to_drop, '', regex=False)
#                     if type(new_row["answer"].iloc[0]) != type("aboba"):
#                         print(type(new_row["answer"].iloc[0]), 2)
#                     new_rows.append(new_row)

#             if drop_token > 0:
#                 if np.random.rand() < drop_token:
#                     new_row = df_0.sample().copy()
#                     tokens = new_row['answer'].iloc[0].split()
#                     drop_index = np.random.randint(0, len(tokens) + 1)
#                     if len(tokens) > 0:
#                         drop_index = np.random.randint(0, len(tokens))
#                         tokens.pop(drop_index)
#                     new_row['answer'] = ' '.join(tokens)
#                     if tokens:
#                       if type(new_row["answer"].iloc[0]) != type("aboba"):
#                           print(type(new_row["answer"].iloc[0]), 3)
#                       new_rows.append(new_row)

#             if double_token > 0:
#                 if np.random.rand() < double_token:
#                     new_row = df_0.sample().copy()
#                     tokens = new_row['answer'].iloc[0].split()
#                     for _ in range(5):
#                         if tokens:
#                             duplicate_index = np.random.randint(0, len(tokens))
#                             tokens.insert(duplicate_index + 1, tokens[duplicate_index])
#                     new_row['answer'] = ' '.join(tokens)
#                     if tokens:
#                         if type(new_row["answer"].iloc[0]) != type("aboba"):
#                             print(type(new_row["answer"].iloc[0]), 4)
#                         new_rows.append(new_row)

#             if insert_random_symbol > 0:
#                 if np.random.rand() < insert_random_symbol:
#                     new_row = df_0.sample().copy()
#                     for _ in range(10):
#                         random_symbol = np.random.choice(list(cyrillic_letters))
#                         insert_index = np.random.randint(0, len(new_row['answer']))
#                         new_row['answer'] = new_row['answer'][:insert_index] + random_symbol + new_row['answer'][insert_index:]

#                     if type(new_row["answer"].iloc[0]) != type("aboba"):
#                         print(type(new_row["answer"].iloc[0]), 5)
#                     new_rows.append(new_row)

#             if swap_tokens > 0:
#                 if np.random.rand() < swap_tokens:
#                     new_row = df_0.sample().copy()
#                     tokens = new_row['answer'].iloc[0].split()
#                     for _ in range(3):
#                         if len(tokens) > 1:
#                             swap_index = np.random.randint(0, len(tokens) - 1)
#                             tokens[swap_index], tokens[swap_index + 1] = tokens[swap_index + 1], tokens[swap_index]
#                     new_row['answer'] = ' '.join(tokens)
#                     if tokens:
#                         if type(new_row["answer"].iloc[0]) != type("aboba"):
#                             print(type(new_row["answer"].iloc[0]), 6)
#                         new_rows.append(new_row)


#             if siblings > 0:
#                 if np.random.rand() < siblings:
#                     new_row = df_0.sample().copy()
#                     answer = new_row['answer'].iloc[0]
#                     new_answer = []

#                     for char in answer:
#                         if char in sibling_letters and np.random.rand() < siblings:
#                             new_answer.append(sibling_letters[char])
#                         else:
#                             new_answer.append(char)

#                     new_row['answer'] = ''.join(new_answer)
#                     if new_answer:
#                         if type(new_row["answer"].iloc[0]) != type("aboba"):
#                             print(type(new_row["answer"].iloc[0]), 7)
#                         new_rows.append(new_row)


#         while df_1.shape[0] + len(new_rows) < df_0.shape[0]:
#             if speech_garbage > 0:
#                 if np.random.rand() < speech_garbage:
#                     new_row = df_1.sample().copy()
#                     text = new_row['question']
#                     text_split = text.iloc[0].split()
#                     if text_split:
#                         insert_index = np.random.randint(0, len(text_split) + 1)
#                         random_word = np.random.choice(["ээ", "мм", "ну", "кхм-кхм"])
#                         text_split.insert(insert_index, random_word)
#                     text = " ".join(text_split)
#                     new_row["question"] = text
#                     if text_split:
#                         if type(new_row["question"].iloc[0]) != type("aboba"):
#                             print(type(new_row["question"].iloc[0]), 8)
#                         new_rows.append(new_row)

#             if drop_symbol > 0:
#                 if np.random.rand() < drop_symbol:
#                     new_row = df_1.sample().copy()
#                     for _ in range (10):
#                         symbol_to_drop = np.random.choice(list(cyrillic_letters))
#                         new_row['answer'] = new_row['answer'].str.replace(symbol_to_drop, '', regex=False)
#                     if type(new_row["answer"].iloc[0]) != type("aboba"):
#                         print(type(new_row["answer"].iloc[0]), 9)
#                     new_rows.append(new_row)

#             if drop_token > 0:
#                 if np.random.rand() < drop_token:
#                     new_row = df_1.sample().copy()
#                     tokens = new_row['answer'].iloc[0].split()
#                     drop_index = np.random.randint(0, len(tokens) + 1)
#                     if len(tokens) > 0:
#                         drop_index = np.random.randint(0, len(tokens))
#                         tokens.pop(drop_index)
#                     new_row['answer'] = ' '.join(tokens)
#                     if tokens:
#                         if type(new_row["answer"].iloc[0]) != type("aboba"):
#                             print(type(new_row["answer"].iloc[0]), 10)
#                         new_rows.append(new_row)

#             if double_token > 0:
#                 if np.random.rand() < double_token:
#                     new_row = df_1.sample().copy()
#                     tokens = new_row['answer'].iloc[0].split()
#                     for _ in range(5):
#                         if tokens:
#                             duplicate_index = np.random.randint(0, len(tokens))
#                             tokens.insert(duplicate_index + 1, tokens[duplicate_index])
#                     new_row['answer'] = ' '.join(tokens)
#                     if tokens:
#                         if type(new_row["answer"].iloc[0]) != type("aboba"):
#                             print(type(new_row["answer"].iloc[0]), 11)
#                         new_rows.append(new_row)

#             if insert_random_symbol > 0:
#                 if np.random.rand() < insert_random_symbol:
#                     new_row = df_1.sample().copy()
#                     for _ in range(10):
#                         random_symbol = np.random.choice(list(cyrillic_letters))
#                         insert_index = np.random.randint(0, len(new_row['answer']))
#                         new_row['answer'] = new_row['answer'].str[:insert_index] + random_symbol + new_row['answer'].str[insert_index:]
#                     if type(new_row["answer"].iloc[0]) != type("aboba"):
#                         print(type(new_row["answer"].iloc[0]), 12)
#                     new_rows.append(new_row)

#             if swap_tokens > 0:
#                 if np.random.rand() < swap_tokens:
#                     new_row = df_1.sample().copy()
#                     tokens = new_row['answer'].iloc[0].split()
#                     for _ in range(3):
#                         if len(tokens) > 1:
#                             swap_index = np.random.randint(0, len(tokens) - 1)
#                             tokens[swap_index], tokens[swap_index + 1] = tokens[swap_index + 1], tokens[swap_index]
#                     new_row['answer'] = ' '.join(tokens)
#                     if tokens:
#                         if type(new_row["answer"].iloc[0]) != type("aboba"):
#                             print(type(new_row["answer"].iloc[0]), 13)
#                         new_rows.append(new_row)

#             if siblings > 0:
#                 if np.random.rand() < siblings:
#                     new_row = df_1.sample().copy()
#                     answer = new_row['answer'].iloc[0]
#                     new_answer = []

#                     for char in answer:
#                         if char in sibling_letters and np.random.rand() < siblings:
#                             new_answer.append(sibling_letters[char])
#                         else:
#                             new_answer.append(char)

#                     new_row['answer'] = ''.join(new_answer)
#                     if new_answer:
#                         if type(new_row["answer"].iloc[0]) != type("aboba"):
#                             print(type(new_row["answer"].iloc[0]), 14)
#                         new_rows.append(new_row)

#     if new_rows:
#         new_df = pd.concat(new_rows).reset_index(drop=True)
#         df = pd.concat([df, new_df], ignore_index=True)
#     df = df.drop_duplicates()

#     return df
