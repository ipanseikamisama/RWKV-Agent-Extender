import requests
import urllib.parse
tools_list = []

def search_anime(s):
    result = None
    def is_url(s):
        try:
            result = urllib.parse.urlparse(s)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    if is_url(s):
        result = requests.get("https://api.trace.moe/search?url={}".format(urllib.parse.quote_plus(s))).json()
    else:
        result = requests.post("https://api.trace.moe/search",
                data=open(s, "rb"),
                headers={"Content-Type": "image/jpeg"}
                ).json()

    rs = f"动画名： {result['result']['filename']}\n集数： {result['result']['episode']}\n " \
         f"时间： {int(result['result']['from']/60)}:{result['result']['from']%60} " \
         f"- {int(result['result']['to']/60)}:{result['result']['to']%60}"

    return rs

