# # from utils.DataReader import LocalReader
# # from utils.DataProcessor import DataProcessor
# # from utils.DataQuery import DataQuery
# import os
# import glob
# import chromadb
# # import transformers
# # from langchain.llms import openlm
# from langchain_openai import ChatOpenAI
# from langchain import OpenAI
# # llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="rwkv_key", base_url="http://127.0.0.1:8000", streaming=True)
# # llm_2 = OpenAI(model="gpt-3.5-turbo", api_key="rwkv_key", base_url="http://127.0.0.1:8000")
# #
# # print(llm.invoke("hello"))
#
# import whisper
# import gradio as gr
# # audio = whisper.load_audio("D:/rwkv_runner/rwkv_extend/test.mp3")
# def get_text_from_audio(audio):
#     model = whisper.load_model("small")
#
#     audio = whisper.load_audio("audio")
#     audio = whisper.pad_or_trim(audio)
#
#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#
#     # detect the spoken language
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")
#
#     # decode the audio
#     options = whisper.DecodingOptions()
#     result = whisper.decode(model, mel, options)
#
#     # print the recognized text
#     return result.text
#
# # inputs = gr.Audio()
# # outputs = gr.outputs.Textbox()
#
# # 创建 Gradio 的应用界面
# app = gr.Interface(get_text_from_audio, "audio", "text", title="语音转文字 Demo")
# # app.launch()
#
# import sqlalchemy
# from sqlalchemy import create_engine
# from sqlalchemy.orm import Session
# from sqlalchemy import MetaData, Table, Column
# from sqlalchemy.dialects.mysql import INTEGER, DOUBLE, BIGINT, VARCHAR, CHAR, TEXT, DATETIME
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.schema import CreateTable
# import pandas as pd
# engine = create_engine('mysql://root:Xingyuelingfeng1m@localhost/rwkv_knowledge', echo=True)
# connection = engine.connect()


# import gradio as gr
#
#
# def sentence_builder(quantity, animal, countries, place, activity_list, morning):
#     return f"""The {quantity} {animal}s from {" and ".join(countries)} went to the {place} where they {" and ".join(activity_list)} until the {"morning" if morning else "night"}"""
#
#
# demo = gr.Interface(
#     sentence_builder,
#     [
#         gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),
#         gr.Dropdown(
#             ["cat", "dog", "bird"], label="Animal", info="Will add more animals later!"
#         ),
#         gr.CheckboxGroup(["USA", "Japan", "Pakistan"], label="Countries", info="Where are they from?"),
#         gr.Radio(["park", "zoo", "road"], label="Location", info="Where did they go?"),
#         gr.Dropdown(
#             ["ran", "swam", "ate", "slept"], value=["swam", "slept"], multiselect=True, label="Activity",
#             info="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl."
#         ),
#         gr.Checkbox(label="Morning", info="Did they do it in the morning?"),
#     ],
#     "text",
#     examples=[
#         [2, "cat", ["Japan", "Pakistan"], "park", ["ate", "swam"], True],
#         [4, "dog", ["Japan"], "zoo", ["ate", "swam"], False],
#         [10, "bird", ["USA", "Pakistan"], "road", ["ran"], False],
#         [8, "cat", ["Pakistan"], "zoo", ["ate"], True],
#     ]
# )
#
# if __name__ == "__main__":
#     demo.launch()


# import torch
# from PIL import Image
# import numpy as np
# import chromadb
# import cn_clip.clip as clip
# from cn_clip.clip import load_from_name, available_models
# print("Available models:", available_models())
# # Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
# chroma_client = chromadb.Client()
# collection = chroma_client.create_collection(name="my_collection")
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = load_from_name("ViT-L-14", device=device, download_root='pretrained_models')
# model.eval()
# image = preprocess(Image.open("somedoc/pokemon.jpeg")).unsqueeze(0).to(device)
# image2 = preprocess(Image.open("somedoc/jieni.jpg")).unsqueeze(0).to(device)
# image3 = preprocess(Image.open("somedoc/apple.jpg")).unsqueeze(0).to(device)
#
# text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)
#
#     image_2_feature = model.encode_image(image2)
#     image_2_feature /= image_2_feature.norm(dim=-1, keepdim=True)
#
#     image_3_feature = model.encode_image(image3)
#     image_3_feature /= image_3_feature.norm(dim=-1, keepdim=True)
#
#     x = image_features.tolist()
#     y = image_2_feature.tolist()
#     z = image_3_feature.tolist()
#     x += y
#     x += z
#     collection.add(
#         documents=["假杰尼龟图片", "假皮卡丘图片", "假苹果图片"],
#         embeddings=x,
#         metadatas=[{"source": "my_source"}, {"source": "my_source"}, {"source": "my_source"}],
#         ids=["id0", "id1", "id2"]
#     )
#
#
#     logits_per_image, logits_per_text = model.get_similarity(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]
# query_text = text = clip.tokenize(["苹果"]).to(device)
# text1_features = model.encode_text(query_text)
# text1_features /= text1_features.norm(dim=-1, keepdim=True)
# query = text1_features.tolist()
# print(collection.get(ids=["id0"]))
# print(collection.query(query_embeddings=query, n_results=3))

# from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
# ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
# img_path = 'somedoc/pokemon.jpeg'
# result = ocr.ocr(img_path, cls=True)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line[1][0])

# from PIL import Image
# result = result[0]
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')

# from utils.DataReader import OnlineReader
#
# olr = OnlineReader(api_key="8266f303b478d750b38461172669c48fb30197b08b805ec2243a9aed4dd8276e", search_engine="google")
# olr.search("who is neuro-sama")

# from utils.DataReader import LocalReader
# from utils.DataProcessor import DataProcessor
# from utils.DataQuery import DataQuery
# import os
# import glob
# import chromadb
# # import transformers
# # from langchain.llms import openlm
# from langchain_openai import ChatOpenAI
# from langchain import OpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="rwkv_key", base_url="http://127.0.0.1:8000", streaming=True)
# llm_2 = OpenAI(model="gpt-3.5-turbo", api_key="rwkv_key", base_url="http://127.0.0.1:8000")
#
# print(llm.invoke("hello"))

# from utils.Prompt import SqlPrompt
# import requests
#
# query = "How many departments are led by heads who are not mentioned?"
# schema = "CREATE TABLE management (department_id VARCHAR); CREATE TABLE department (department_id VARCHAR)"
# t_prompt = SqlPrompt.get_prompt(query, schema)
# # cont = f"{query},and the schema is{schema}, tell me what is the sqlquery with json format"
# templates = {
#   "frequency_penalty": 1,
#   "max_tokens": 1000,
#   "messages": [
#     {
#       "content": t_prompt,
#       "raw": False,
#       "role": "user"
#     }
#   ],
#   "model": "rwkv",
#   "presence_penalty": 0,
#   "presystem": True,
#   "stream": False,
#   "temperature": 1,
#   "top_p": 0.3
# }
#
# # templates = {
# #   "frequency_penalty": 1,
# #   "max_tokens": 200,
# #   "model": "rwkv",
# #   "presence_penalty": 0,
# #   "prompt": t_prompt,
# #   "stream": False,
# #   "temperature": 1,
# #   "top_p": 0.3
# # }
#
# # print(content)
# print(t_prompt)
# # templates["messages"][0].update({"content": content})
# # templates["prompt"] = t_prompt
# bot_message = requests.post("http://127.0.0.1:8000/chat/completions", json=templates)
# # result = bot_message.json()['choices'][0]["text"]
# result = bot_message.json()['choices'][0]["message"]['content']
# print(result)

# import easyocr
# reader = easyocr.Reader(['ch_sim', 'en'])
# result = reader.readtext('somedoc/apple.jpg')
# print(result)
# for line in result:
#     print(line[1])

# import pandas as pd
# import numpy as np
# import sqlalchemy
# from sqlalchemy import create_engine
# from sqlalchemy.orm import Session
# from sqlalchemy.schema import CreateTable
# from utils.DataReader import LocalReader
# from utils.DataProcessor import DataProcessor
# from utils.DataQuery import DataQuery
# path = "somedoc/16lite.png"
# mp = "rbt6"
# dbp = "D:/rwkv_runner/rwkv_extend/vector_temp"
# process = DataProcessor(mp, dbp)
# process.add_file([path])
# queryer = DataQuery(mp, dbp)
# query = "16bit的感动这部作品都犯了哪些问题"
# a = queryer.search_media(query)
# print(a)
#
# system_config = {"启用知识库": True, "启用网络": False, "启用数据表": True, "启用多媒体": True, "图片搜索": False,
#                  "文档问答": False, "tools": False}
# FileLock = False
# TempLock = False
# def Chain(query):
#     global system_config
#     knowledge = ""
#     format = ""
#     if system_config["文档问答"] is True and TempLock is False:
#         temp(query)
#     elif FileLock is False:
#         if system_config["启用知识库"] is True:
#             knowledge, format = queryer.relate_search(query)
#         if system_config['启用网络'] is True:
#             knowledge, format = onlinesearch(query)
#         if system_config['启用数据表'] is True:
#             knowledge, format = queryer.search_with_sql(query, None)
#         if system.config['启用多媒体'] is True:
#             knowledge, format = queryer.search_media(query)
#     if system_config['图片搜索'] is True:
#         photos = queryer.search_photo(query)
#     if system_config['tools'] is True:
#         tools(query)
#
#     content = get_answer_prompt(knowledge, query)
#     template(content)
#     answer =get_answer(templates)
#     yield answer
#     yield format
#

#
# import whisper
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# import os
#
# model = whisper.load_model("small")
#
#
# # 使用Whisper本地进行音频转录
# def transcribe_audio_whisper(path):
#     result = model.transcribe(path)
#     text = result['text']
#     return text
#
#
# # 将音频文件根据静音部分分割成块，并使用Whisper API进行语音识别的函数
# def get_large_audio_transcription_on_silence_whisper(path, export_chunk_len):
#     sound = AudioSegment.from_file(path)
#     chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=sound.dBFS - 14, keep_silence=500)
#
#     folder_name = "audio-chunks"
#     if not os.path.isdir(folder_name):
#         os.mkdir(folder_name)
#
#     # 现在重新组合这些块，使得每个部分至少有export_chunk_len长。
#     output_chunks = [chunks[0]]
#     for chunk in chunks[1:]:
#         if len(output_chunks[-1]) < export_chunk_len:
#             output_chunks[-1] += chunk
#         else:
#             # 如果最后一个输出块的长度超过目标长度， 我们可以开始一个新的块
#             output_chunks.append(chunk)
#
#     whole_text = ""
#     for i, audio_chunk in enumerate(output_chunks, start=1):
#         chunk_filename = os.path.join(folder_name, f"chunk{i}.mp3")
#         audio_chunk.export(chunk_filename, format="mp3")
#
#         try:
#             text = transcribe_audio_whisper(chunk_filename)
#         except Exception as e:
#             print("Error:", str(e))
#         else:
#             text = f"{text.capitalize()}. "
#             print(chunk_filename, ":", text)
#             whole_text += text
#
#     return whole_text
#
#
# path = "somedoc/test.mp3"
# export_chunk_len = 90 * 1000
#
# # audio_text = get_large_audio_transcription_on_silence_whisper(path, export_chunk_len)
# audio_text = transcribe_audio_whisper(path)
# print("\nAudio Full text:", audio_text)


# import re
#
# s = "```sql\nSELECT * FROM '分娩妇女个人心理及生理状况表' WHERE '编号' = '202183021';\n```"
# start = "```sql"
# end = "```"
#
# def extract_between(text, start_char, end_char):
#     pattern = re.escape(start_char) + r"(.*?)" + re.escape(end_char)
#     match = re.search(pattern, text, re.DOTALL)
#
#     if match:
#         result = match.group(1)
#         return result
#     else:
#         return None
#
# result = extract_between(s, start, end)
#
# print(result)

# from sqlalchemy import MetaData, Table, Column
# from sqlalchemy.dialects.mysql import INTEGER, DOUBLE, BIGINT, VARCHAR, CHAR, TEXT, DATETIME
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# table = pd.read_excel(path)
# engine = create_engine('mysql://root:Xingyuelingfeng1m@localhost/db2', echo=True)
# connection = engine.connect()
# meta = MetaData()
#
# table_search = sqlalchemy.text("show tables")
# tables = connection.execute(table_search)
# table_list = []
# for row in tables:
#   table_list.append(row[0])
# if "分娩妇女个人心理及生理状况表" not in table_list:
#   table.to_sql("分娩妇女个人心理及生理状况表", engine)
#
# data_table = Table("分娩妇女个人心理及生理状况表", meta, autoload_with=engine)
# schema = str(CreateTable(data_table))
# print(schema)
#
# show_query = sqlalchemy.text("select * from 分娩妇女个人心理及生理状况表 limit 10;")
# result = connection.execute(show_query)
# rows = result.fetchall()
#
# # 将每个行的结果转换为字符串
# row_result = ""
# for row in rows:
#     # 将每个值转换为字符串，然后连接起来
#     row_str = ', '.join(str(value) for value in row)
#     row_result += "(" + row_str + ")\n"
#     print(row_str)
#
# print("____________\n"+row_result)
# find = ""
# for row in result:
#
#   find.join(str(row))
#
# print(find)
    # dbp = "D:/rwkv_runner/rwkv_extend/vector_temp"
    # 'mysql://root:Xingyuelingfeng1m@localhost/db2'
    # 'mysql://root:Xingyuelingfeng1m@localhost/rwkv_temp_database'
import json
diction = {
    "save": "file_save.json",
    'embedding_model': 'rbt6',
    'chroma_path': "D:/rwkv_runner/rwkv_extend/vector_temp",
    'sql_path': 'mysql://root:Xingyuelingfeng1m@localhost/db2',
    'temp_sql_path': 'mysql://root:Xingyuelingfeng1m@localhost/rwkv_temp_database'
}
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(diction, f, ensure_ascii=False)