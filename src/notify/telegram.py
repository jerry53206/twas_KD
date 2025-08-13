import requests

def send_telegram(bot_token: str, chat_id: str, text: str, timeout: int = 30):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=timeout)
    r.raise_for_status()
    return r.status_code
