import requests
import gradio
import os
import re
query = "Please help me play the music file in my local system named \"Suger\""

tools = [
    {
        "name": "search_path",
        "parameters": ["str"],
        "description": "find the local path of an object and return it."
    },
    {
        "name": "play_media",
        "parameters": ["path"],
        "description": "find the local media by the path and play it."
    },
    {
        "name": "show_database",
        "parameters": [],
        "description": "show the current database."
    },
    {
        "name": "print",
        "parameters": ["str"],
        "description": "print the string."
    },
    {
        "name": "search_web",
        "parameters": ["str"],
        "description": "search the web with the given string, returning raw web page data."
    },
    {
        "name": "extract_information",
        "parameters": ["str", "key"],
        "description": "extract the key information in the web data."
    }

]

def get_template(query, tool_list):
    tools = ""
    for tool in tool_list:
        param = "("
        for i in tool['parameters']:
            param += f"{i}, "
        param = param[:-2] + ")" if param != "(" else "()"
        tools += f"- `{tool['name']}{param}`: {tool['description']}\n    "


    AGENT_TMPL = f'''
    Complete the content based on the example, stop generating when meet `#Finished`.
    user: What's the phone number of "Riverside Grill"
    Available Tools (some may not be used):
    {tools}
    
    Assistant:
    Thoughts:
    - I need to find the restaurant's information from a search engine.
    - I need to extract the phone number of the restaurant.
    - I need to print the phone number of the restaurant.
    Reasoning:
    - `search_web` can search the web for the restaurant's information.
    - `extract_information` can extract the phone number from the search result. The key should be "phone number".
    - `print` can print the phone number to the output.
    Execution:
    [
    - web_result = search_web("Riverside Grill")
    - phone_number = extract_information(web_result, "phone number")
    - print(phone_number)
    ]
    #Finished

    user: {query}
    Available Tools (some may not be used):
    {tools}
    Assistant:
    Thoughts:
    '''

    content = f'''

    '''
    return AGENT_TMPL

# templates = {
#   "frequency_penalty": 1,
#   "max_tokens": 1000,
#   "messages": [
#     {
#       "content": "",
#       "raw": False,
#       "role": "user"
#     }
#   ],
#   "model": "rwkv",
#   "presence_penalty": 0,
#   "presystem": False,
#   "prompt": None,
#   "stream": False,
#   "temperature": 1,
#   "top_p": 0.3
# }

templates = {
  "frequency_penalty": 1,
  "max_tokens": 200,
  "model": "rwkv",
  "presence_penalty": 0,
  "prompt": "The following is an epic science fiction masterpiece that is immortalized, with delicate descriptions and grand depictions of interstellar civilization wars.\nChapter 1.\n",
  "stream": False,
  "temperature": 1,
  "top_p": 0.3
}

prompt = get_template(query, tools)
# print(content)
print(prompt)
# templates["messages"][0].update({"content": content})
templates["prompt"] = prompt
bot_message = requests.post("http://127.0.0.1:8000/completions", json=templates)
result = bot_message.json()

def search_path(object):
    print(f"search path of {object}")

def play_media(media_path, player = "MPC-HC"):
    print(f"play file in {media_path}, by player {player}")

def extract_between(text, start_char, end_char):
    pattern = re.escape(start_char) + r"(.*?)" + re.escape(end_char)
    match = re.search(pattern, text, re.DOTALL)

    if match:
        result = match.group(1)
        return result
    else:
        return None

print(result['choices'][0]['text'])
command = extract_between(result['choices'][0]['text'], "[", "]")
command = command.replace("- ", "").replace("    ","")
print(command)
global_namespace = globals()
local_namespace = locals()
exec(command, {'search_path':search_path, 'play_media': play_media})