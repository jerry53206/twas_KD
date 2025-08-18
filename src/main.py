# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from finlab import data
from datetime import datetime
import telegram

# ===============================
# 參數設定
# ===============================
KD_K_PERIOD = 9
KD_D_PERIOD = 3

CONSEC_DAYS = 2        # 連續出現至少幾天
ALLOW_GAP_DAYS = 3     # 容忍週末/假日差距天數

# Telegram 多帳號/群組設定
TELEGRAM_TOKENS = ["YOUR_BOT_TOKEN"]   # 可放多個 token，若同一 bot 可只一個
CHAT_IDS = [12345678, 98765432, -111222333]  # 帳號或群組 ID, 群組為負數

# ===============================
# 取得資料
# ===============================
close = data.get('price:收盤價')
rev = data.get('monthly_revenue:當月營收')

# ===============================
# 計算 TWAS 因子
# ===============================
rev_pct = rev.pct_change().fillna(0)
twas_signal = (rev_pct > 0)

# ===============================
# 計算 KD 指標
# ===============================
kd_signal = pd.DataFrame(index=close.index, columns=close.columns)

for stock in close.columns:
    try:
        low = data.get(f'price:最低價:{stock}')
        high = data.get(f'price:最高價:{stock}')
        k, d = talib.STOCH(
            high.values, low.values, close[stock].values,
            fastk_period=KD_K_PERIOD,
            slowk_period=KD_K_PERIOD,
            slowk_matype=0,
            slowd_period=KD_D_PERIOD,
            slowd_matype=0
        )
        kd_signal[stock] = (k > d)
    except:
        kd_signal[stock] = False

# ===============================
# 合併選股訊號
# ===============================
signal = twas_signal & kd_signal

records = []
for date in signal.index:
    for stock in signal.columns:
        if signal.loc[date, stock]:
            records.append({'date': date, 'stock': stock})

df_signal = pd.DataFrame(records)
df_signal['date'] = pd.to_datetime(df_signal['date'])
df_signal = df_signal.sort_values(['stock','date']).drop_duplicates()

# ===============================
# 計算連續出現股票
# ===============================
def calc_consecutive(group):
    group = group.sort_values('date')
    group['prev_date'] = group['date'].shift(1)
    group['diff_days'] = (group['date'] - group['prev_date']).dt.days
    group['consecutive'] = (group['diff_days'] <= ALLOW_GAP_DAYS).cumsum()
    return group

df_signal = df_signal.groupby('stock').apply(calc_consecutive)

consec_stocks = df_signal.groupby('stock').filter(lambda x: len(x) >= CONSEC_DAYS)['stock'].unique()

# ===============================
# Telegram 發送訊息給多帳號/群組
# ===============================
message = f"今日連續出現股票 ({CONSEC_DAYS}天以上)：\n" + ("\n".join(consec_stocks) if len(consec_stocks) > 0 else "無符合條件股票")

for token in TELEGRAM_TOKENS:
    bot = telegram.Bot(token=token)
    for chat_id in CHAT_IDS:
        bot.send_message(chat_id=chat_id, text=message)

print(message)
