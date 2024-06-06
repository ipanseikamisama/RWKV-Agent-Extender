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
import os
import glob
import chromadb
import transformers

# with gr.Blocks() as demo:
#   chatbot = gr.Chatbot()
#   msg = gr.Textbox()
#   clear = gr.ClearButton([msg, chatbot])
#
#   def user(user_message, history):
#     return "", history + [[user_message, None]]
#
#   def bot(chat_history):
#     app = FastAPI()
#     chat_history[-1][1] = ""
#
#     tempelates = {
#       "frequency_penalty": 1,
#       "max_tokens": 1000,
#       "messages": [
#         {
#           "content": "",
#           "raw": False,
#           "role": "user"
#         }
#       ],
#       "model": "rwkv",
#       "presence_penalty": 0,
#       "presystem": True,
#       "prompt":None,
#       "stream": True,
#       "temperature": 1,
#       "top_p": 0.3
#     }
#     content = chat_history[-1][0]
#     print(content)
#
#     tempelates["messages"][0].update({"content": content})
#     bot_message = requests.post("http://127.0.0.1:8000/chat/completions", json=tempelates, stream=True)
#
#     for i in bot_message.iter_lines():
#
#       def replace_last_substring(original_string, substring, replacement):
#         # 使用 rsplit() 方法找到最后一个子串的索引
#         index = original_string.rfind(substring)
#         if index == -1:  # 如果字符串中没有找到子串，直接返回原始字符串
#           return original_string
#         else:
#           # 将最后一个子串替换为指定的字符串
#           new_string = original_string[:index] + replacement + original_string[index + len(substring):]
#           return new_string
#
#       if i.decode("utf-8") == "" or i.decode("utf-8") == "\n" or i is None or i.decode("utf-8")[-6:] == "[DONE]":
#         continue
#       stream_dict = eval(replace_last_substring(i.decode("utf-8")[6:], "null", "None"))["choices"][0]["delta"]
#       # stream_dict = eval(i.decode("utf-8")[6:].replace("null","None"))["choices"][0]["delta"]
#       if "content" in list(stream_dict.keys()):
#         chat_history[-1][1] += stream_dict["content"]
#         time.sleep(0.05)
#         yield chat_history
#
#
#   msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#     bot, chatbot, chatbot
#   )
#   clear.click(lambda: None, None, chatbot, queue=False)
#
#
# demo.queue()
# demo.launch()
import whisper

def get_text_from_audio(audio):
  model = whisper.load_model("small")

  audio = whisper.load_audio("audio")
  audio = whisper.pad_or_trim(audio)

  # make log-Mel spectrogram and move to the same device as the model
  mel = whisper.log_mel_spectrogram(audio).to(model.device)

  # detect the spoken language
  _, probs = model.detect_language(mel)
  print(f"Detected language: {max(probs, key=probs.get)}")

  # decode the audio
  options = whisper.DecodingOptions()
  result = whisper.decode(model, mel, options)

  # print the recognized text
  return result.text

def predict(message,history):
  tempelates = {
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
  path = "D:/rwkv_runner/rwkv_extend/somedoc/全球化文档.docx"
  mp = "rbt6"
  dbp = "D:/rwkv_runner/rwkv_extend/vector_temp"
  # processor = DataProcessor(model_path=mp, db_path=dbp)
  # processor.add_file([path])
  queryer = DataQuery(mp, dbp)

  content = queryer.relate_search(message)
  # content = message
  print(content)
  chat_history = ""
  tempelates["messages"][0].update({"content": content})
  bot_message = requests.post("http://127.0.0.1:8000/chat/completions", json=tempelates, stream=True)

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
      chat_history += stream_dict["content"]
      print(chat_history)
      yield chat_history

# predict("全球化的主要问题是什么")
def zip_to_json(file_obj):
  files = []
  with ZipFile(file_obj.name) as zfile:
    for zinfo in zfile.infolist():
      files.append(
        {
          "name": zinfo.filename,
          "file_size": zinfo.file_size,
          "compressed_size": zinfo.compress_size,
        }
      )
  return files

file_list = ["全球化问题手册", "线性代数教程"]

with gr.Blocks() as demo:
  gr.ChatInterface(predict)
  with gr.Accordion("问答模式设置"):
    with gr.Row():
      gr.CheckboxGroup(["启用本地知识库", "启用网络搜索", "启用数据表", "启用多媒体"],label="知识库选项",info="选择问答时参考哪些知识，可选多个")
    with gr.Row():
      with gr.Column():
        gr.Checkbox(label='只根据被选择的文档作答', info="启用后将只根据被选择的文档生成回答")
      with gr.Column():
        gr.Checkbox(label="启用tools", info="启用后将使用工具来完成特定任务")
    with gr.Row():
      gr.Dropdown(
        file_list, value=["全球化问题手册"], multiselect=True, label="选择文档",
        info="选择特定文档，被选择的文档将一定被作为问答时的前置知识"
      ),
  with gr.Accordion("语音输入与多媒体", open=False):
    with gr.Tab("语音输入"):
      gr.Interface(get_text_from_audio, "audio", "text", title="语音输入")
    with gr.Tab("文件输入"):
      gr.Interface(zip_to_json, "file", "json")


demo.launch()


