interface = ":"

# If you modify this, make sure you have newlines between user and bot words too

user = "Q"
bot = "A"

init_prompt = f'''
{bot} will judge if the query whether a query is question...

{user}{interface} Is the query below a question: 冰淇淋的主要成分有哪些

{bot}{interface} Yes

{user}{interface} Is the query below a question: 今天天气怎么样

{bot}{interface} Yes

{user}{interface} Is the query below a question: 解释一下相对论的基本概念

{bot}{interface} Yes

{user}{interface} Is the query below a question: 原神到底有没有抄袭

{bot}{interface} Yes

{user}{interface} Is the query below a question: 我最近开始学习弹吉他，发现音乐对心情有很大的影响

{bot}{interface} No
'''