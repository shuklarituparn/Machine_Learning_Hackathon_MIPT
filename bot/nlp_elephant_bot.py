import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.longpoll import VkLongPoll, VkEventType
import random

from io import BytesIO
import json
import enum
import os

import datetime
from multiprocessing import Process
from threading import Thread


import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification


my_club_id = 227850389
my_prev_message = dict()
users_registered_filename = "users_registered.json"
ya = [399014949]

hello_words = ['привет', 'здрасте', 'здрасьте', 'здравствуй', 'хеллоу', 'hello', 
               'здарова', 'вечер в хату', 'здарово', 'доброе утро', 'добрый день', 'добрый вечер']

my_dela = ['как обычно', 'нормально', 'критически стабильно', 'отлично',
           'балдёжно', 'суперски', 'хорошо']

ask_for_help_words = ['', '', '']

nepon_words = ['Не понял, что вы сказали', 'Неизвестная команда',
               'Я не понял, повторите, пожалуйста, так, чтобы мне стало понятно', 'Не понял',
               'Вы говорите на непонятном мне диалекте, повторите, я постараюсь понять',
               'Не очень понял, что вы имеете в виду']
laughing_smiles = ['&#128518;', '&#128516;', '&#128514;']

intro_text = 'Я - бот для проверки достоверности информации. ' \
            'Я умеею выбирать наиболее достоверный ответ (из предложенных) на заданный вопрос. \n' \
            'Я воспринимаю как команды только слова, которые написаны с восклицательным знаком в начале. \n' \
            '\nДля вывода системы команд напишите \"!Помощь\" (без кавычек). \n' \
            '\nПо всем техническим вопросам обращайтесь в техподдержку: https://vk.com/the_lord_of_praks'

help_text = 'Я - бот для проверки достоверности информации. ' \
            'Я умеею выбирать наиболее достоверный ответ (из 3-x предложенных) на заданный вопрос. \n' \
            'Я воспринимаю как команды только слова, которые написаны с восклицательным знаком в начале. Например: \"!Помощь\". \n' \
            'Система команд: \n' \
            '\"!Помощь\" - вывести систему команд и инструкцию по использованию \n' \
            '\"!Новый вопрос\" - задать новый вопрос \n' \
            '\"!Удалить вопрос\" - удалить текущий вопрос и ответы на него (завершить сеанс) \n' \
            '\nПо всем техническим вопросам обращайтесь в техподдержку: https://vk.com/the_lord_of_praks'

dict_request = dict({"question": "", "answer1": "", "answer2": "", "answer3": ""})

class BotMode(enum.Enum):
    DEFAULT = 0
    WAIT_FOR_QUESTION = 1
    WAIT_FOR_ANS1 = 2
    WAIT_FOR_ANS2 = 3
    WAIT_FOR_ANS3 = 4
    WAIT_FOR_ONE_ANS = 5


bot_mode = BotMode.DEFAULT.value


model = BertForSequenceClassification.from_pretrained('cointegrated/rubert-tiny2', num_labels=2)
state_dict2 = torch.load(os.path.join("..", "best_model_v5.pt"), map_location=torch.device('cpu'))
model.load_state_dict(state_dict2)
tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
    

def predict_valid_ans(question, answer1, answer2, answer3):
    inputs1 = tokenizer(f"{question}[SEP]{answer1}", return_tensors='pt', truncation=True, padding=True)
    inputs2 = tokenizer(f"{question}[SEP]{answer2}", return_tensors='pt', truncation=True, padding=True)
    inputs3 = tokenizer(f"{question}[SEP]{answer3}", return_tensors='pt', truncation=True, padding=True)
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)
    outputs3 = model(**inputs3)
    predictions1, predictions2, predictions3 = torch.argmax(outputs1.logits, dim=-1), torch.argmax(outputs2.logits, dim=-1), torch.argmax(outputs3.logits, dim=-1)
    prob_relevant1 = torch.reshape(outputs1.logits, (-1,))[1].item()
    prob_relevant2 = torch.reshape(outputs2.logits, (-1,))[1].item()
    prob_relevant3 = torch.reshape(outputs3.logits, (-1,))[1].item()
    return np.array([prob_relevant1, prob_relevant2, prob_relevant3])
    #return [torch.reshape(outputs1.logits.numpy, (-1,)), torch.reshape(outputs2.logits, (-1,)), torch.reshape(outputs3.logits, (-1,))]
    
    # predictions = torch.argmax(outputs.logits, dim=-1)
    # return 1 if predictions.item() == 1 else 0

def reply_message(vk_session, event):
    vk = vk_session.get_api()

    message_to_send, deletion = generate_message(event.obj.message['text'], event.obj.message['from_id'], vk_session,
                                                 event.obj.message['peer_id'])
    #print("message generated")
    vk.messages.send(peer_id=event.obj.message['peer_id'],
                     message=message_to_send,
                     random_id=random.randint(0, 2 ** 64))
    #print(message_to_send, deletion)
    if deletion:
        message_id = vk.messages.getHistory(count=1, user_id=event.obj.message['peer_id'])['items'][0]['id']
        vk.messages.delete(delete_for_all=0, message_ids=[message_id])


def jsonify_file(filename):
    with open(filename) as json_file:
        dict_read_data = json.load(json_file)
        #print('drd', dict_read_data)
        return dict_read_data


def generate_message(text, id, vk_session, peer_id):
    global bot_mode

    vk = vk_session.get_api()
    user_info = vk.users.get(user_id=id)
    #name, surname = user_info[0]['first_name'], user_info[0]['last_name']

    text_lower = text.lower()

    if "!Помощь" in text:
        # q = "Почему небо голубое?"
        # a1 = "Потому что ты голубой"
        # a2 = "Из-за рассеяния Рэлея"
        # a3 = "Потому что в нем плавают три кита"
        # print(predict_valid_ans(q, a1, a2, a3))
        return help_text, 0

    for word in hello_words:
        if word in text_lower:
            return "Доброго времени суток! " + intro_text, 0

    if "!Новый вопрос" in text:
        bot_mode = BotMode.WAIT_FOR_QUESTION.value
        return "С нетерпением жду Вашего вопроса", 0
    
    if "!Удалить вопрос" in text:
        for key in dict_request.keys():
            dict_request[key] = ""
        bot_mode = BotMode.WAIT_FOR_QUESTION.value
        return "Вопрос и ответы на него успешно удалены", 0

    if bot_mode == BotMode.WAIT_FOR_QUESTION.value:
        if '?' in text:
            dict_request["question"] = text
            bot_mode = BotMode.WAIT_FOR_ANS1.value
            return "Вопрос принят, теперь введите первый из 3-х предполагаемых ответов на него. Ответ 1 из 3:", 0
        else:
            return "Странные у Вас вопросы - в них даже отсутствует вопросительный знак. Попробуйте ввести вопрос еще раз", 0

    if bot_mode == BotMode.WAIT_FOR_ANS1.value:
        dict_request["answer1"] = text
        bot_mode = BotMode.WAIT_FOR_ANS2.value
        return "Ваш первый вариант ответа принят, теперь введите второй из 3-х предполагаемых ответов. Ответ 2 из 3:", 0

    if bot_mode == BotMode.WAIT_FOR_ANS2.value:
        dict_request["answer2"] = text
        bot_mode = BotMode.WAIT_FOR_ANS3.value
        return "Ваш второй вариант ответа принят, теперь введите третий из 3-х предполагаемых ответов. Ответ 3 из 3:", 0

    if bot_mode == BotMode.WAIT_FOR_ANS3.value:
        dict_request["answer3"] = text
        bot_mode = BotMode.DEFAULT.value
        probs_relevant = predict_valid_ans(dict_request["question"], dict_request["answer1"], dict_request["answer2"], dict_request["answer3"])
        #print(probs_relevant)
        # return "Итак, все 3 из 3 вариантов ответа приняты! \n" + \
        #         f"Я думаю, что наиболее правильным является {np.argmax(probs_relevant) + 1} вариант ответа.\n\n" + \
        #         f"Вопрос: {dict_request['question']}\n" + \
        #         "Вероятности вариантов ответов: \n" +\
        #         f"Первый вариант (вероятность корректности {probs_relevant[0].round(2)}): \"{dict_request['answer1']}\"\n" + \
        #         f"Второй вариант (вероятность корректности {probs_relevant[1].round(2)}): \"{dict_request['answer2']}\"\n" + \
        #         f"Третий вариант (вероятность корректности {probs_relevant[2].round(2)}): \"{dict_request['answer3']}\"\n" + \
        #         "\nЗадавайте вопросы, буду рад на них ответить!", 0
        list_answers = [dict_request['answer1'], dict_request['answer2'], dict_request['answer3']]
        return "Итак, все 3 из 3 вариантов ответа приняты! \n\n" + \
                f"Вопрос: {dict_request['question']}\n" + \
                "Варианты ответов: \n" +\
                f"Первый вариант: \"{dict_request['answer1']}\"\n" + \
                f"Второй вариант: \"{dict_request['answer2']}\"\n" + \
                f"Третий вариант: \"{dict_request['answer3']}\"\n\n" + \
                f"Я думаю, что наиболее правильным является {np.argmax(probs_relevant) + 1} вариант ответа: \"{list_answers[np.argmax(probs_relevant)]}\"\n\n" + \
                "\nЗадавайте вопросы, буду рад на них ответить!", 0
    return nepon_words[random.randint(0, len(nepon_words))] + "." + " \nДля получения справки о списке команд напишите \"!Помощь\"", 0

def listening(longpoll):
    print("start listening")
    for event in longpoll.listen():
        #print(event)
        #try:
        if event.type == VkBotEventType.MESSAGE_NEW:
            #print('Текст сообщения:', event.obj.message['text'])
            reply_message(vk_session, event)
        #except Exception as e:
        #    print("Exception in messager_main")
        #    print(e)


def messager_main():
    global vk_session
    print('start')
    vk_session = vk_api.VkApi(
        token='vk1.a.RT2OHXVqO2oOXGna9StM27-S7MPaU1mmQnAA5R719IQK1LbLGYWnI00VaE7Yb1PSO9QKExZuOziGtm8UAN_Jv2WDNRaxkSBg9sjnM-LY3QXyi27A-igbqb1WK3dGVPDP7Usrd_EeZkswCOFNZa4I7DWdvnH2ESGdSOiZ0L_8U6mBPThRv0OhSLbStU7N2U_ZzgM6V_p2Oig8-Dd2Xw1qlw')
    longpoll = VkBotLongPoll(vk_session, my_club_id)
    listening(longpoll)

while(True):
    try:
        messager_main()
    except Exception as e:
        print(e)
        continue