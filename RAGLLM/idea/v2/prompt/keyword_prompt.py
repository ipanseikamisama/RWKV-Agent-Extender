interface = ":"

# If you modify this, make sure you have newlines between user and bot words too

user = "Q"
bot = "A"
query_1 = "原神》是由上海米哈游网络科技股份有限公司制作发行的一款开放世界冒险游戏，于2017年1月底立项 [27]，原初测试于2019年6月21日开启 [1]，再临测试于2020年3月19日开启，启程测试于2020年6月11日开启 [2]，PC版技术性开放测试于9月15日开启，公测于2020年9月28日开启 [3]。在数据方面，同在官方服务器的情况下，iOS、PC、Android平台之间的账号数据互通，玩家可以在同一账号下切换设备。"
query_2 = "我在访问某个网页的时候因为发生了代号503的错误而无法进入网页，我该怎么办？"
query_3 = "我是一名大三的学生，现在面临期末，但是我想要去预约一下学校校医院的牙医，我该预约什么时候的牙医比较好？"

init_prompt = f'''
{bot} is intelligent and will extract the key words from the sentence given.

{user}{interface} please extract the key words from the sentence: {query_1}

{bot}{interface} 原神,上海米哈游网络科技股份有限公司,开放世界冒险游戏,2017年1月底,立项,原初测试,2019年6月21日,再临测试,2020年3月19日,启程测试,2020年6月11日,PC版,技术性开放测试,9月15日,公测,2020年9月28日,数据方面,官方服务器,iOS,PC,Android,平台,账号数据互通,玩家,同一账号,切换设备

{user}{interface} please extract the key words from the sentence:: {query_2}

{bot}{interface} 访问,网页,503错误,无法进入,怎么办,解决办法,网络问题,服务器错误,状态码,问题处理,故障排除

{user}{interface} please extract the key words from the sentence: {query_3}

{bot}{interface} 大三,学生,面临期末,预约,学校,校医院,牙医,什么时候,预约时机,牙医预约,建议,最佳时机
'''