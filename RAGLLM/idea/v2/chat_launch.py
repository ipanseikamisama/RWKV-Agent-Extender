########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, copy, types, gc, sys
import numpy as np
import torch


# args.strategy = 'cuda:0 fp16 -> cuda:1 fp16'
# args.strategy = 'cuda fp16i8 *10 -> cuda fp16'
# args.strategy = 'cuda fp16i8'
# args.strategy = 'cuda fp16i8 -> cpu fp32 *10'
# args.strategy = 'cuda fp16i8 *10+'

os.environ["RWKV_JIT_ON"] = '1'  # '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_CUDA_ON"] = '1'  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

CHAT_LANG = 'Chinese'  # English // Chinese // more to come
args = None
user = ""
bot = ""
interface = ""
init_prompt = ""
srv_list = ['dummy_server']


def launch():
    global current_path
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(f'{current_path}/../rwkv_pip_package/src')

    global args
    from prompt_toolkit import prompt
    global prompt
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    except:
        pass
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    args = types.SimpleNamespace()

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    args.strategy = 'cuda fp16'

    os.environ["RWKV_JIT_ON"] = '1'  # '1' or '0', please use torch 1.13+ and benchmark speed
    os.environ["RWKV_CUDA_ON"] = '1'  # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

    global CHAT_LANG
    CHAT_LANG = 'Chinese'  # English // Chinese // more to come
    if CHAT_LANG == 'English':
        args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192'

    elif CHAT_LANG == 'Chinese':  # Raven系列可以对话和 +i 问答。Novel系列是小说模型，请只用 +gen 指令续写。
        args.MODEL_NAME = 'D:/RWKV-5-World-3B-v2-20231118-ctx16k'

    elif CHAT_LANG == 'Japanese':
        args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-raven/RWKV-4-Raven-7B-v10-Eng89%-Jpn10%-Other1%-20230420-ctx4096'

    PROMPT_FILE = f'{current_path}/prompt/default/{CHAT_LANG}-1.py'

    global CHAT_LEN_SHORT, CHAT_LEN_LONG, FREE_GEN_LEN, GEN_TEMP, GEN_TOP_P,\
        GEN_alpha_presence, GEN_alpha_frequency, GEN_penalty_decay, CHUNK_LEN, AVOID_REPEAT
    CHAT_LEN_SHORT = 40
    CHAT_LEN_LONG = 150
    FREE_GEN_LEN = 256

    GEN_TEMP = 1.2  # It could be a good idea to increase temp when top_p is low
    GEN_TOP_P = 0.5  # Reduce top_p (to 0.5, 0.2, 0.1 etc.) for better Q&A accuracy (and less diversity)
    GEN_alpha_presence = 0.4  # Presence Penalty
    GEN_alpha_frequency = 0.4  # Frequency Penalty
    GEN_penalty_decay = 0.996
    AVOID_REPEAT = '，：？！'

    CHUNK_LEN = 256  # split input into chunks to save VRAM (shorter -> slower)

    if args.MODEL_NAME.endswith('/'):  # for my own usage
        if 'rwkv-final.pth' in os.listdir(args.MODEL_NAME):
            args.MODEL_NAME = args.MODEL_NAME + 'rwkv-final.pth'
        else:
            latest_file = sorted([x for x in os.listdir(args.MODEL_NAME) if x.endswith('.pth')],
                                 key=lambda x: os.path.getctime(os.path.join(args.MODEL_NAME, x)))[-1]
            args.MODEL_NAME = args.MODEL_NAME + latest_file

    print(f'\n{CHAT_LANG} - {args.strategy} - {PROMPT_FILE}')

    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE
    print(f'Loading model - {args.MODEL_NAME}')
    global model
    model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
    global END_OF_TEXT, END_OF_LINE, END_OF_LINE_DOUBLE, pipeline
    if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
        pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
        END_OF_TEXT = 0
        END_OF_LINE = 11
    else:
        pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
        END_OF_TEXT = 0
        END_OF_LINE = 187
        END_OF_LINE_DOUBLE = 535

    global model_tokens, model_state
    model_tokens = []
    model_state = None

    global AVOID_REPEAT_TOKENS
    AVOID_REPEAT_TOKENS = []
    for i in AVOID_REPEAT:
        dd = pipeline.encode(i)
        assert len(dd) == 1
        AVOID_REPEAT_TOKENS += dd

    global all_state
    all_state = {}

    print(f'\nRun prompt...')

    global user, bot, interface, init_prompt, out
    user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
    out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
    save_all_stat('', 'chat_init', out)
    gc.collect()
    torch.cuda.empty_cache()

    global srv_list
    srv_list = ['dummy_server']
    for s in srv_list:
        save_all_stat(s, 'chat', out)

    if CHAT_LANG == 'English':
        HELP_MSG = '''Commands:
    say something --> chat with bot. use \\n for new line.
    + --> alternate chat reply
    +reset --> reset chat

    +gen YOUR PROMPT --> free single-round generation with any prompt. use \\n for new line.
    +i YOUR INSTRUCT --> free single-round generation with any instruct. use \\n for new line.
    +++ --> continue last free generation (only for +gen / +i)
    ++ --> retry last free generation (only for +gen / +i)

    Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B (especially https://huggingface.co/BlinkDL/rwkv-4-raven) for best results.
    '''
    elif CHAT_LANG == 'Chinese':
        HELP_MSG = f'''指令:
    直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行，必须用 Raven 模型
    + --> 让机器人换个回答
    +reset --> 重置对话，请经常使用 +reset 重置机器人记忆

    +i 某某指令 --> 问独立的问题（忽略聊天上下文），用\\n代表换行，必须用 Raven 模型
    +gen 某某内容 --> 续写内容（忽略聊天上下文），用\\n代表换行，写小说用 testNovel 模型
    +++ --> 继续 +gen / +i 的回答
    ++ --> 换个 +gen / +i 的回答

    作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957
    如果喜欢，请看我们的优质护眼灯: https://withablink.taobao.com

    中文 Novel 模型，可以试这些续写例子（不适合 Raven 模型）：
    +gen “区区
    +gen 以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\\n第一章
    +gen 这是一个修真世界，详细世界设定如下：\\n1.
    '''
    elif CHAT_LANG == 'Japanese':
        HELP_MSG = f'''コマンド:
    直接入力 --> ボットとチャットする．改行には\\nを使用してください．
    + --> ボットに前回のチャットの内容を変更させる．
    +reset --> 対話のリセット．メモリをリセットするために，+resetを定期的に実行してください．

    +i インストラクトの入力 --> チャットの文脈を無視して独立した質問を行う．改行には\\nを使用してください．
    +gen プロンプトの生成 --> チャットの文脈を無視して入力したプロンプトに続く文章を出力する．改行には\\nを使用してください．
    +++ --> +gen / +i の出力の回答を続ける．
    ++ --> +gen / +i の出力の再生成を行う.

    ボットとの会話を楽しんでください。また、定期的に+resetして、ボットのメモリをリセットすることを忘れないようにしてください。
    '''

    print(HELP_MSG)
    print(f'{CHAT_LANG} - {args.MODEL_NAME} - {args.strategy}')

    print(f'{pipeline.decode(model_tokens)}'.replace(f'\n\n{bot}', f'\n{bot}'), end='')


def load_prompt(PROMPT_FILE):
    variables = {}
    with open(PROMPT_FILE, 'rb') as file:
        exec(compile(file.read(), PROMPT_FILE, 'exec'), variables)
    user, bot, interface, init_prompt = variables['user'], variables['bot'], variables['interface'], variables['init_prompt']
    init_prompt = init_prompt.strip().split('\n')
    for c in range(len(init_prompt)):
        init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
    init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'
    return user, bot, interface, init_prompt

# Load Model



########################################################################################################

def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    # print(f'### model ###\n{tokens}\n[{pipeline.decode(model_tokens)}]')

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    out[END_OF_LINE] += newline_adj # adjust \n probability

    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out

def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

# Model only saw '\n\n' as [187, 187] before, but the tokenizer outputs [535] for it at the end
def fix_tokens(tokens):
    if 'world/' in args.MODEL_NAME or '-World-' in args.MODEL_NAME:
        return tokens
    if len(tokens) > 0 and tokens[-1] == END_OF_LINE_DOUBLE:
        tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
    return tokens

########################################################################################################



def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

def reset(message):
    srv = 'dummy_server'

    msg = message.replace('\\n', '\n').strip()
    msg = msg.strip()
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return 'True'
    else:
        return 'False'

def add_prompt(message):
    global model_tokens, model_state, user, bot, interface, init_prompt
    result = ""
    srv = 'dummy_server'

    msg = message.replace('\\n', '\n').strip()

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    msg = msg.strip()

    if msg[:8].lower() == '+prompt ':
        try:
            PROMPT_FILE = msg[8:].strip()
            user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
            out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
            save_all_stat(srv, 'chat', out)
            print("Prompt set up.")
            gc.collect()
            torch.cuda.empty_cache()
            return 'True'
        except:
            return 'False'

def get_response(message):
    global dq
    message = dq.search_relate_index(message, n_result=2, key_results=2, final_result=3)
    srv = 'dummy_server'
    msg = message.replace('\\n','\n').strip()
    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp=" + f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p=" + f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    msg = msg.strip()

    out = load_all_stat(srv, 'chat')
    msg = msg.strip().replace('\r\n', '\n').replace('\n\n', '\n')
    new = f"{user}{interface} {msg}\n\n{bot}{interface}"
    # print(f'### add ###\n[{new}]')
    out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
    save_all_stat(srv, 'chat_pre', out)

    response = ""
    begin = len(model_tokens)
    out_last = begin
    print(f'{bot}{interface}', end='', flush=True)
    occurrence = {}
    for i in range(999):
        if i <= 0:
            newline_adj = -999999999
        elif i <= CHAT_LEN_SHORT:
            newline_adj = (i - CHAT_LEN_SHORT) / 10
        elif i <= CHAT_LEN_LONG:
            newline_adj = 0
        else:
            newline_adj = min(3, (i - CHAT_LEN_LONG) * 0.25)  # MUST END THE GENERATION

        for n in occurrence:
            out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
        token = pipeline.sample_logits(
            out,
            temperature=x_temp,
            top_p=x_top_p,
        )
        # if token == END_OF_TEXT:
        #     break
        for xxx in occurrence:
            occurrence[xxx] *= GEN_penalty_decay
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        out = run_rnn([token], newline_adj=newline_adj)
        out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

        xxx = pipeline.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:  # avoid utf-8 display issues
            print(xxx, end='', flush=True)
            out_last = begin + i + 1

        send_msg = pipeline.decode(model_tokens[begin:])
        response = send_msg
        if '\n\n' in send_msg:
            send_msg = send_msg.strip()
            response = send_msg
            break
    save_all_stat(srv, 'gen_1', out)
    return response

def on_message(message, history):
    history_transformer_format = history + [[message, ""]]
    global model_tokens, model_state, user, bot, interface, init_prompt
    result = ""
    srv = 'dummy_server'

    msg = message.replace('\\n','\n').strip()

    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    msg = msg.strip()
    
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return
    
    # use '+prompt {path}' to load a new prompt
    elif msg[:8].lower() == '+prompt ':
        print("Loading prompt...")
        try:
            PROMPT_FILE = msg[8:].strip()
            user, bot, interface, init_prompt = load_prompt(PROMPT_FILE)
            out = run_rnn(fix_tokens(pipeline.encode(init_prompt)))
            save_all_stat(srv, 'chat', out)
            print("Prompt set up.")
            gc.collect()
            torch.cuda.empty_cache()
        except:
            print("Path error.")

    elif msg[:5].lower() == '+gen ' or msg[:3].lower() == '+i ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:3].lower() == '+i ':
            msg = msg[3:].strip().replace('\r\n','\n').replace('\n\n','\n')
            new = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

# Instruction:
{msg}

# Response:
'''
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qq ':
            new = '\nQ: ' + msg[4:].strip() + '\nA:'
            # print(f'### prompt ###\n[{new}]')
            model_state = None
            model_tokens = []
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            out = run_rnn(pipeline.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+++':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '++':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return
        begin = len(model_tokens)
        out_last = begin
        occurrence = {}
        for i in range(FREE_GEN_LEN+100):
            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            if token == END_OF_TEXT:
                break
            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

            if msg[:4].lower() == '+qa ':# or msg[:4].lower() == '+qq ':
                out = run_rnn([token], newline_adj=-2)
            else:
                out = run_rnn([token])
            
            xxx = pipeline.decode(model_tokens[out_last:])
            result = xxx
            yield result
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
                if i >= FREE_GEN_LEN:
                    break
        print('\n')

        # send_msg = pipeline.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)


    else:
        if msg.lower() == '+':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            msg = msg.strip().replace('\r\n','\n').replace('\n\n','\n')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(pipeline.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        occurrence = {}
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= CHAT_LEN_SHORT:
                newline_adj = (i - CHAT_LEN_SHORT) / 10
            elif i <= CHAT_LEN_LONG:
                newline_adj = 0
            else:
                newline_adj = min(3, (i - CHAT_LEN_LONG) * 0.25) # MUST END THE GENERATION

            for n in occurrence:
                out[n] -= (GEN_alpha_presence + occurrence[n] * GEN_alpha_frequency)
            token = pipeline.sample_logits(
                out,
                temperature=x_temp,
                top_p=x_top_p,
            )
            # if token == END_OF_TEXT:
            #     break
            for xxx in occurrence:
                occurrence[xxx] *= GEN_penalty_decay            
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1
            
            out = run_rnn([token], newline_adj=newline_adj)
            out[END_OF_TEXT] = -999999999  # disable <|endoftext|>

            xxx = pipeline.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx: # avoid utf-8 display issues
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
            
            send_msg = pipeline.decode(model_tokens[begin:])
            result = send_msg
            yield result
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                yield send_msg
                break
            
            # send_msg = pipeline.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{pipeline.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)


########################################################################################################

########################################################################################################

import socket
import threading
from toTest import DataQuery
import gradio as gr
import tkinter as tk
from tkinter import scrolledtext

class ChatApp:
    def __init__(self, root):
        self.root = root
        root.title("简单对话应用")

        self.chat_history = scrolledtext.ScrolledText(root, width=40, height=10)
        self.chat_history.pack(pady=10)

        self.input_entry = tk.Entry(root, width=40)
        self.input_entry.pack(pady=10)

        self.send_button = tk.Button(root, text="发送", command=self.send_message)
        self.send_button.pack()

    def send_message(self):
        user_input = self.input_entry.get()
        response = get_response(user_input)

        self.chat_history.insert(tk.END, f"你: {user_input}\n")
        self.chat_history.insert(tk.END, f"应用: {response}\n")
        self.chat_history.insert(tk.END, "\n")

        self.input_entry.delete(0, tk.END)  # 清空输入框

def receive_data(connection):
    full_data = b''  # 用于保存完整数据的字节串
    chunk_size = 50  # 每次接收的块大小

    while True:
        # try:
        chunk = connection.recv(chunk_size)
        print(chunk.decode('utf-8', errors='ignore'))
        if len(chunk) <= 8:
            if chunk.decode('utf-8', errors='ignore')[-8:] not in "__stop__":
                full_data += chunk
                sub_chunk = connection.recv(chunk_size)
                full_data += sub_chunk
                break
            else:
                full_data += chunk
                break
        else:
            if chunk.decode('utf-8', errors='ignore')[-8:] in "__stop__":
                full_data += chunk
                break
        full_data += chunk
        # except UnicodeDecodeError as e:
        #     print("meet bytes can't be handled by utf-8 and skipped")
        #     continue

    full_data = full_data.decode('utf-8', errors='ignore')
    if len(full_data) <= 8:
        return full_data
    else:
        return full_data[:-8]

def send_data(connection, data):
    chunk_size = 50  # 每次发送的块大小

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        connection.send(chunk.encode('utf-8'))

    connection.send('__stop__'.encode('utf-8'))  # 关闭写入，表示数据发送完毕毕


def handle_client(connection, address):
    print('已连接：', address)

    while True:
        received_data = receive_data(connection)
        print('接收到的数据：', received_data)
        if received_data.lower() == '+exit':
            break
        elif received_data[:8].lower() == '+prompt ':
            message = add_prompt(received_data)
            print("get prompt answer: "+message)
        elif received_data == '' or received_data is None:
            message = "please say somthing"
        else:
            message = get_response(received_data)
            print("none")
        if message is not None and message != '':
            send_data(connection, message)

# 其他函数和 receive_data、send_data 函数的定义
def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8080))
    server_socket.listen(5)
    print('等待连接...')

    while True:
        connection, address = server_socket.accept()

        # 创建新线程处理客户端
        # threading.Thread(target=handle_client, args=(connection, address)).start()
        handle_client(connection, address)

# def talking():
#     launch()
#     dq = DataQuery(r'D:\bigPythonProject\LLMtest\rbt6', r"D:\testLLMdb")
#     while True:
#         # msg = prompt(f'{user}{interface} ')
#         msg = input(f'{user}{interface}')
#         msg = dq.search_relate_index(msg, n_result=2, key_results=2, final_result=3)
#         print("we finally got the msg"+msg)
#         if len(msg.strip()) > 0:
#             chat = get_response(msg)
#             print(chat, flush=True)
#         else:
#             print('Error: please say something')

# 重置用户输入
def resetinput():
    return gr.update(value='')

#重置状态
def reset_textbox():
    return gr.update(value=""),""

"""
chatbot = gr.Chatbot()
input_text = gr.Text(label = "输入信息")
demo = gr.Interface(
    answer,
    [input_text,"state"],
    [chatbot,"state"]
)
demo.launch()
"""
#创建头像
def answer_1(msg, history):
    history = history or []
    global dq
    msg = dq.search_relate_index(msg, n_result=2, key_results=2, final_result=3)
    response = get_response(msg)
    history.append((response, response))

    return history, history

def talking():
    # global app
    launch()
    # root = tk.Tk()
    # app = ChatApp(root)
    # root.mainloop()
    title = "你好，这里是智能机器人小十"
    global state, button_txt, button_reset, button_2, button_reset2, input_txt, demo
    with gr.Blocks() as demo:
        # gr.Markdown(f'<p style="font-size:20px;">{"这里是"}</p>')

        gr.Markdown("<center>" + f'<p style="font-size:20px;">{"你好，这里是智能机器人小十"}</p>' + "</center>")
        chatbot = gr.Chatbot(
            bubble_full_width=False,
            label="消息记录",
        )
        with gr.Column():
            state = gr.State([])
            # chatbot = gr.Chatbot(label="消息记录")
            input_txt = gr.Textbox(label="输入信息", placeholder="快来向我提问吧！")
            with gr.Tab("模型1"):
                with gr.Row():
                    button_txt = gr.Button("🤖 提问")
                    button_reset = gr.Button("🔄 新对话")
            with gr.Tab("模型2"):
                with gr.Row():
                    button_2 = gr.Button("🤖 提问")
                    button_reset2 = gr.Button("🔄 新对话")

        input_txt.submit(answer_1, [input_txt, state], [chatbot, state])
        button_txt.click(answer_1, [input_txt, state], [chatbot, state])
        input_txt.submit(resetinput, [], [input_txt])
        button_reset.click(reset_textbox, [], [chatbot, state])
    # demo.queue().launch()
    auth_list = [(str(i), "830" + str(i).zfill(2)) for i in range(1, 41)]
    demo.queue().launch(auth=auth_list, auth_message="🤖请输入你的账号和密码")

if __name__ == "__main__":
    threading.Thread(target=talking).start()
    threading.Thread(target=main).start()
    dq = DataQuery(r'D:\bigPythonProject\LLMtest\rbt6', r"D:\testLLMdb")

