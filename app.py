from fastapi import FastAPI
import requests
import openai
import gradio as gr
import random
import time
from zipfile import ZipFile
from utils.DataReader import LocalReader
from utils.DataProcessor import DataProcessor
from utils.DataQuery import DataQuery
from utils.TempRetriever import TempRetriever
import os
import glob
import whisper
from pathlib import Path
import soundfile as sf
from utils import PromptLoader
import json

with open("config.json", "r", encoding="utf-8") as config:
    configs = json.load(config)
    json_save_path = configs["save"]
    mp = configs['embedding_model']
    dbp = configs['chroma_path']
    sql_path = configs['sql_path']
    temp_sql_path = configs['temp_sql_path']

processor = DataProcessor(mp, dbp, sql_path)
if os.path.exists(json_save_path):
    processor.load_list(json_save_path)
queryer = DataQuery(mp, dbp)
temp_retriever = TempRetriever(mp, dbp, temp_sql_path)
FileLock = False
TempLock = False
retriever_config = {"启用本地知识库": True, "启用网络搜索": False, "启用数据表": True, "启用多媒体": True,
                    "图片搜索": False,
                    "只根据被选择的文档作答": False, "启用tools": False}

test_file_list = []


def get_name_list(file_list):
    name_list = []
    for doc_data in file_list:
        if doc_data['Name'] not in name_list:
            name_list.append(doc_data['Name'])
        else:
            i = 2
            name = doc_data['Name'] + "_" + str(i)
            while name in name_list:
                i += 1
                name = doc_data['Name'] + "_" + str(i)
            name_list.append(name)
    return name_list


def get_path_list(file_list):
    path_list = []
    for doc_data in file_list:
        path_list.append(doc_data['Path'])
    return path_list


def get_text_from_audio(audio):
    sr, data = audio
    audio_path = 'temp_folder/output.wav'
    sf.write(audio_path, data, sr)
    model = whisper.load_model("small")


    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    os.remove(audio_path)
    # print the recognized text
    return result.text


def upload_files(file_obj):
    global FileLock
    FileLock = True
    file_path = file_obj.name
    processor.add_file([file_path])
    present_persist_list = get_path_list(processor.get_list())
    FileLock = False
    global json_save_path
    processor.save_list(json_save_path)
    return str(present_persist_list), gr.Dropdown(choices=present_persist_list, interactive=True)


def set_parameters(persist_retriever, temp_retriever, tools):
    global retriever_config
    for i in retriever_config.keys():
        if i in persist_retriever:
            retriever_config[i] = True
        else:
            retriever_config[i] = False
    if temp_retriever is True:
        retriever_config['只根据被选择的文档作答'] = True
    if tools is True:
        retriever_config['启用tools'] = True

    return str(retriever_config)


def reference_file(file_list):
    global TempLock
    TempLock = True
    path_group = []
    back_up = temp_retriever.current_holder_list
    for i in back_up:
        if i['Path'] not in file_list:
            temp_retriever.delete_file(i['Path'])
    for path in file_list:
        if not temp_retriever.holder_exists(path):
            path_group.append(path)
    temp_retriever.add_file(path_group)
    TempLock = False
    return str(temp_retriever.current_holder_list)+"文件载入成功"
    # temp_retriever.add_file(path_group)


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    text_msg = gr.Textbox()
    with gr.Accordion("语音输入"):
        msg = gr.Textbox()
        gr.Interface(get_text_from_audio, "mic", msg, title="语音输入")
        submit = gr.Button("输入语音文本")
    with gr.Row():
        clear = gr.ClearButton([msg, chatbot])


    def user(user_message, history):
        return "", history + [[user_message, None]]


    def bot(chat_history):
        chat_history[-1][1] = ""
        message = chat_history[-1][0]
        templates = {
            "frequency_penalty": 1,
            "max_tokens": 10000,
            "messages": [
                {
                    "content": "",
                    "raw": False,
                    "role": "user"
                }
            ],
            "model": "rwkv",
            "presence_penalty": 0,
            "presystem": True,
            "stream": True,
            "temperature": 1,
            "top_p": 0.3
        }
        global queryer, processor, temp_retriever
        knowledge = ""
        format = ""
        if retriever_config['只根据被选择的文档作答'] is True and TempLock is not True:
            knowledge = temp_retriever.get_retriever(message)
            # retriever_config = {"启用本地知识库": True, "启用网络搜索": False, "启用数据表": True, "启用多媒体": True,
            #                     "图片搜索": False,
            #                     "只根据被选择的文档作答": False, "启用tools": False}
        elif FileLock is False:
            if retriever_config['启用本地知识库'] is True:
                temp_kw, temp_ft = queryer.relate_search(message)
                knowledge += temp_kw
                format += temp_ft
            if retriever_config['启用网络搜索'] is True:
                temp_kw, temp_ft = queryer.online_search(message)
                knowledge += temp_kw
                format += temp_ft
            if retriever_config['启用数据表'] is True:
                temp_kw, temp_ft = queryer.search_with_sql(message, None)
                knowledge += temp_kw
                format += temp_ft
            if retriever_config['启用多媒体'] is True:
                temp_kw, temp_ft = queryer.search_media(message)
                knowledge += temp_kw
                format += temp_ft
        content = PromptLoader.get_answer_prompt(message, knowledge)
        print(content)
        templates["messages"][0].update({"content": content})
        bot_message = requests.post("http://127.0.0.1:8000/chat/completions", json=templates, stream=True)

        for i in bot_message.iter_lines():
            def replace_last_substring(original_string, substring, replacement):
                # 使用 rsplit() 方法找到最后一个子串的索引
                index = original_string.rfind(substring)
                if index == -1:  # 如果字符串中没有找到子串，直接返回原始字符串
                    return original_string
                else:
                    # 将最后一个子串替换为指定的字符串
                    new_string = original_string[:index] + replacement + original_string[index + len(substring):]
                    return new_string

            if i.decode("utf-8") == "" or i.decode("utf-8") == "\n" or i is None or i.decode("utf-8")[-6:] == "[DONE]" \
                    or "{" not in i.decode("utf-8"):
                continue
            stream_dict = eval(replace_last_substring(i.decode("utf-8")[6:], "null", "None"))["choices"][0]["delta"]
            # stream_dict = eval(i.decode("utf-8")[6:].replace("null","None"))["choices"][0]["delta"]
            if "content" in list(stream_dict.keys()):
                chat_history[-1][1] += stream_dict["content"]
                time.sleep(0.05)
                yield chat_history


    submit.click(user, [msg, chatbot], [msg, chatbot]).then(bot, chatbot, chatbot)
    text_msg.submit(user, [text_msg, chatbot], [text_msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    # msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
    # bot, chatbot, chatbot
    # )
    clear.click(lambda: None, None, chatbot, queue=False)

    ##options 设置选项
    with gr.Accordion("问答模式设置"):
        with gr.Column():
            gr.Interface(
                set_parameters,
                [
                    gr.CheckboxGroup(["启用本地知识库", "启用网络搜索", "启用数据表", "启用多媒体"], label="知识库选项",
                                     info="选择问答时参考哪些知识，可选多个"),
                    gr.Checkbox(label='只根据被选择的文档作答', info="启用后将只根据被选择的文档生成回答"),
                    gr.Checkbox(label="启用tools", info="启用后将使用工具来完成特定任务"),
                ],
                "text"
            )
        with gr.Column():
            choose_file = gr.Dropdown(
                get_path_list(processor.get_list()), value=None, multiselect=True, label="选择文档",
                info="选择特定文档，被选择的文档将一定被作为问答时的前置知识"
            )
            gr.Interface(
                reference_file,
                choose_file,
                "text"
            )

    ##语音输入和文件上传
    with gr.Accordion("文件输入", open=False):
        with gr.Tab("文件输入"):
            gr.Interface(upload_files, "file", ["text", choose_file])


demo.queue()
demo.launch()

temp_retriever.delete_all()