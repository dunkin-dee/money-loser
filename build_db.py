import pytz
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine

mt5.initialize()
timezone = pytz.timezone("Etc/UTC")
engine = create_engine("mysql+pymysql://rex:#Pass123@localhost/new_ml")
account=53619597
authorized = mt5.login(account, password="6ufnqlrw", server="MetaQuotes-Demo")


utc_from = datetime(2013, 1, 1, tzinfo=timezone)
utc_to = datetime(2021, 10, 10, tzinfo=timezone)

all_pairs = [
"AUDCAD",
"AUDCHF",
"AUDJPY",
"AUDNZD",
"AUDUSD",
"CADCHF",
"CADJPY",
"CHFJPY",
"EURAUD",
"EURCAD",
"EURCHF",
"EURGBP",
"EURJPY",
"EURUSD",
"GBPAUD",
"GBPCAD",
"GBPCHF",
"GBPJPY",
"GBPNZD",
"GBPUSD",
"NZDCAD",
"NZDCHF",
"NZDJPY",
"NZDUSD",
"USDCAD",
"USDCHF",
"USDJPY"
]

for pair in all_pairs:
  print(f"Fetching data for {pair} at "+datetime.now().strftime("%H:%M:%S"))

  candle_data = mt5.copy_rates_range(pair, mt5.TIMEFRAME_H1, utc_from, utc_to)
  df = pd.DataFrame(candle_data)
  df["time"] = pd.to_datetime(df["time"], unit="s")
  df["time"] = df["time"].astype('datetime64[s]')
  df.rename({"time": "index"}, axis=1, inplace=True)
  df.set_index("index", inplace=True)
  df.drop(["tick_volume", "spread", "real_volume"], axis=1, inplace=True)
  df["bodysize"] = abs(df["close"]-df["open"])
  df['direction'] = np.where(df['open'] > df['close'] , 'down', 'up')

  df["shadow"] = np.where(df["direction"]=="up", df["open"]-df["low"], df["close"]-df["low"])
  df["wick"] = np.where(df["direction"]=="up", df["high"]-df["close"], df["high"]-df["open"])

  print(f"Adding data for {pair} to mysql at "+datetime.now().strftime("%H:%M:%S"))


  table_name = (pair+"_1h").lower()
  df.to_sql(table_name, engine, if_exists='replace', index=True)
  with engine.connect() as con:
     con.execute('ALTER TABLE `%s` ADD PRIMARY KEY(`index`)'%table_name)