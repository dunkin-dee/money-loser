import pandas as pd
import numpy as np
import time
import multiprocessing
import pytz
import json
import MetaTrader5 as mt5
import telegram_send
from sqlalchemy import create_engine
from db_update import columnize_d, columnize_h
from joblib import load
from datetime import date, datetime, timedelta
from pprint import pprint
from time import sleep, strftime

engine = create_engine("mysql+pymysql://rex:#Pass123@localhost/new_ml")
mt5.initialize()
timezone = pytz.timezone("Etc/UTC")
account = 53619597
authorized = mt5.login(account, password="6ufnqlrw", server="MetaQuotes-Demo")

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

stop_points = {
"AUDCAD": [0.0040, 0.0018],
"AUDCHF": [0.0040, 0.0015],
"AUDJPY": [0.55, 0.20],
"AUDNZD": [0.0025, 0.0010],
"AUDUSD": [0.004, 0.002],
"CADCHF": [0.0033, 0.0015],
"CADJPY": [0.70, 0.25],
"CHFJPY": [0.60, 0.25],
"EURAUD": [0.008, 0.003],
"EURCAD": [0.006, 0.0025],
"EURCHF": [0.0025, 0.001],
"EURGBP": [0.0025, 0.001],
"EURJPY": [0.4, 0.15],
"EURUSD": [0.0035, 0.0015],
"GBPAUD": [0.006, 0.0025],
"GBPCAD": [0.005, 0.002],
"GBPCHF": [0.006, 0.0025],
"GBPJPY": [0.6, 0.25],
"GBPNZD": [0.006, 0.0025],
"GBPUSD": [0.004, 0.002],
"NZDCAD": [0.003, 0.0015],
"NZDCHF": [0.0035, 0.0015],
"NZDJPY": [0.45, 0.15],
"NZDUSD": [0.004, 0.0015],
"USDCAD": [0.0042, 0.0018],
"USDCHF": [0.003, 0.0012],
"USDJPY": [0.32, 0.1],
}

def get_mt5_candles(pair, time_period, count=300):
  tfs = {
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D": mt5.TIMEFRAME_D1
  }
  candle_data = np.empty([0,1])
  try_again = True
  while try_again:
    candle_data = mt5.copy_rates_from_pos(pair, tfs[time_period], 0, count)
    if str(type(candle_data)) == "<class 'numpy.ndarray'>" :
      if len(candle_data) >= count:
        try_again = False
    else:
      time.sleep(0.5)
  df = pd.DataFrame(candle_data)
  df["time"] = pd.to_datetime(df["time"], unit="s")
  df.rename({"time": "index"}, axis=1, inplace=True)
  df.set_index("index", inplace=True)
  df.drop(["tick_volume", "spread", "real_volume"], axis=1, inplace=True)
  df["bodysize"] = abs(df["close"]-df["open"])
  df['direction'] = np.where(df['open'] > df['close'] , 'down', 'up')
  df["shadow"] = np.where(df["direction"]=="up", df["open"]-df["low"], df["close"]-df["low"])
  df["wick"] = np.where(df["direction"]=="up", df["high"]-df["close"], df["high"]-df["open"])
  return(df)



def shape_that(unshaped_df_name, last_time, all_tables, shaped_tables):
  if unshaped_df_name.split("_")[1] == "d":
    shaped_table = columnize_d(all_tables[unshaped_df_name])
    shaped_tables[unshaped_df_name] = shaped_table[shaped_table.index.get_loc(last_time[unshaped_df_name])+1:-1]
  
  else:
    shaped_table = columnize_h(all_tables[unshaped_df_name], all_tables[f"{unshaped_df_name.split('_')[0]}_d"])
    shaped_tables[unshaped_df_name] = shaped_table[shaped_table.index.get_loc(last_time[unshaped_df_name])+1:-1]





if __name__ == "__main__":
  repeat_time = datetime.now()
  while True:
    if datetime.now() > repeat_time:
      sleep(2)
      print(f"Collecting data and comparing to database at {datetime.now().strftime('%H:%M:%S')}")
      manager = multiprocessing.Manager()
      wrong_shapable_number = True
      while wrong_shapable_number:
        all_tables = manager.dict()
        last_time = manager.dict()
        shaped_tables = manager.dict()
        reg_dfs = {}

      
        for pair in all_pairs:
          df_h = get_mt5_candles(pair, "H1")
          df_h4 = get_mt5_candles(pair, "H4")
          df_d = get_mt5_candles(pair, "D", count=200)

          table_name_h = f"{pair.lower()}_1h"
          table_name_h4 = f"{pair.lower()}_4h"
          table_name_d = f"{pair.lower()}_d"

          h_db = pd.read_sql(f"select * from {table_name_h} order by `index` desc limit 1", engine, index_col="index")
          h4_db = pd.read_sql(f"select * from {table_name_h4} order by `index` desc limit 1", engine, index_col="index")
          d_db = pd.read_sql(f"select * from {table_name_d} order by `index` desc limit 1", engine, index_col="index")

          last_time_h = h_db.iloc[-1].name
          last_time_h4 = h4_db.iloc[-1].name
          last_time_d = d_db.iloc[-1].name

          reg_dfs[table_name_h4] = h4_db.copy(deep=True)
          reg_dfs[table_name_d] = d_db.copy(deep=True)

          if df_h.iloc[-2].name > last_time_h:
            last_time[f"{pair.lower()}_1h"] = last_time_h

          if df_h4.iloc[-2].name > last_time_h4:
            last_time[f"{pair.lower()}_4h"] = last_time_h4

          if df_d.iloc[-2].name > last_time_d:
            last_time[f"{pair.lower()}_d"] = last_time_d


          all_tables[f"{pair.lower()}_1h"] = df_h.copy(deep=True)
          all_tables[f"{pair.lower()}_4h"] = df_h4.copy(deep=True)
          all_tables[f"{pair.lower()}_d"] = df_d.copy(deep=True)
        if len(last_time) == 0 or len(last_time) == 27 or len(last_time) == 54 or len(last_time) == 81 or len(last_time) == 63:
          wrong_shapable_number = False
        else:
          print(f"Wrong number of dataframes recognised for shaping at: {str(len(last_time))}")
          print(f"Trying again at {datetime.now().strftime('%H:%M:%S')}")

      
      if len(last_time) > 0:
        print(f"All tables: {str(len(all_tables))} Shapable tables: {str(len(last_time))}.")
        print("Now shaping data...")
        pool = multiprocessing.Pool(5)
        my_args = []
        for x in last_time.keys():
          my_args.append((x, last_time, all_tables, shaped_tables))

        pool.starmap(shape_that, my_args)
        pool.close()

        print(f"Done at {datetime.now().strftime('%H:%M:%S')}.")

        with open("./models/model_metrics.json") as json_file:
          metrics_dict = json.load(json_file)

        all_preds = {}
        predicted_pairs = set()
        for x in shaped_tables.keys():
          if x.split("_")[-1] == "1h":
            pair_name = x.split("_")[0]
            all_preds[pair_name] = [] 
            up_clf = load(filename=f"./models/{pair_name}_h_up.joblib")
            down_clf = load(filename=f"./models/{pair_name}_h_down.joblib")

            predicting_row = shaped_tables[x][-1:]

            
            pred_up = up_clf.predict_proba(predicting_row)[-1][-1]
            pred_down = down_clf.predict_proba(predicting_row)[-1][-1]

            levels = []
            cols = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5', 
                    'pp', 'standard_r1', 'standard_s1', 'fib_r1', 'fib_s1',]
            for col in cols:
              levels.append(predicting_row.iloc[-1][col])

            print(f"Predictions for {pair_name.upper()} on timeperiod {predicting_row.iloc[-1].name}:\t\tUP - {'{:.2f}'.format(pred_up)}\tDOWN - {'{:.2f}'.format(pred_down)} \n{'-'*30}")

            thresholds = [0.45, 0.5, 0.55, 0.6]
            #For Long Positions
            if pred_up >= 0.5:
              for temp_threshold in thresholds:
                if temp_threshold <= pred_up:
                  threshold = temp_threshold
              correct_guesses = int(metrics_dict[pair.upper()]["Hourly UP"][str(threshold)]["positive"])
              wrong_guesses = int(metrics_dict[pair.upper()]["Hourly UP"][str(threshold)]["false_positive"])

              proba = int((correct_guesses/(correct_guesses+wrong_guesses))*100)

              up_sl = predicting_row.iloc[-1]["close"] - stop_points[pair_name.upper()][-1]
              up_tp = '{:.5f}'.format(predicting_row.iloc[-1]["close"] + stop_points[pair_name.upper()][0] - (stop_points[pair_name.upper()][-1]/2))
              up_levels = []
              for level in levels:
                if level > predicting_row.iloc[-1]["close"]:
                  up_levels.append(level)
              if len(up_levels) > 0:
                str_up_levels = []
                for up_level in sorted(up_levels):
                  str_up_levels.append('{:.5f}'.format(up_level))
                up_tp = "\n".join(str_up_levels)
              
              up_dict = {
                "position": "LONG",
                "stop_loss": up_sl,
                "take_profit": up_tp,
                "probability": proba,
                "pred_threshold": pred_up
              }

              all_preds[pair_name].append(up_dict)
              predicted_pairs.add(pair_name)

            #For Short positions
            if pred_down >= 0.5:
              for temp_threshold in thresholds:
                if temp_threshold <= pred_down:
                  threshold = temp_threshold
              correct_guesses_d = int(metrics_dict[pair.upper()]["Hourly DOWN"][str(threshold)]["positive"])
              wrong_guesses_d = int(metrics_dict[pair.upper()]["Hourly DOWN"][str(threshold)]["false_positive"])

              d_proba = int((correct_guesses_d/(correct_guesses_d+wrong_guesses_d))*100)

              down_sl = predicting_row.iloc[-1]["close"] + stop_points[pair_name.upper()][-1]
              down_tp = '{:.5f}'.format(predicting_row.iloc[-1]["close"] - stop_points[pair_name.upper()][0] + (stop_points[pair_name.upper()][-1]/2))
              down_levels = []
              for level in levels:
                if level < predicting_row.iloc[-1]["close"]:
                  down_levels.append(level)
              if len(down_levels) > 0:
                str_down_levels = []
                for down_level in sorted(down_levels, reverse=True):
                  str_down_levels.append('{:.5f}'.format(down_level))
                down_tp = "\n".join(str_down_levels)

              down_dict = {
                "position": "SHORT",
                "stop_loss": down_sl,
                "take_profit": down_tp,
                "probability": d_proba,
                "pred_threshold": pred_down
              }

              all_preds[pair_name].append(down_dict)
              predicted_pairs.add(pair_name)
        
        final_string = ""
        for predicted_pair in list(predicted_pairs):
          h4_model = load(filename=f"./models/{predicted_pair}_4h.joblib")
          d_model = load(filename=f"./models/{predicted_pair}_d.joblib")

          if f"{predicted_pair}_4h" in list(shaped_tables.keys()):
            h4_row = shaped_tables[f"{predicted_pair}_4h"][-1:]
          else:
            h4_row = reg_dfs[f"{predicted_pair}_4h"]

          if f"{predicted_pair}_d" in list(shaped_tables.keys()):
            d_row = shaped_tables[f"{predicted_pair}_d"][-1:]
          else:
            d_row = reg_dfs[f"{predicted_pair}_d"]


          h4_pred = h4_model.predict(h4_row)[-1]
          d_pred = d_model.predict(d_row)[-1]

          h4_time = h4_row.iloc[-1].name + timedelta(hours=4)
          d_time = d_row.iloc[-1].name + timedelta(days=1)
         
          for pred_info in all_preds[predicted_pair]:
            pred_info["day_time"] = d_time.strftime("%d-%m-%Y")
            pred_info["day_pred"] = d_pred

            pred_info["4h_time"] = h4_time.strftime("%d-%m-%Y %H:%M:%S")
            pred_info["4h_pred"] = h4_pred

            if pred_info["position"] == "SHORT":
              my_emoji = "&#128308;"
            else:
              my_emoji = "&#128994;"

            final_string = f"{final_string} \n{predicted_pair.upper()}\n<b>{pred_info['position']}</b> {my_emoji}"
            final_string = f"{final_string}\nStop Loss: {'{:.5f}'.format(pred_info['stop_loss'])}"
            final_string = f"{final_string}\nSuggested Take Profits:\n{pred_info['take_profit']}"
            final_string = f"{final_string}\nProbability Index: {'{:.2f}'.format(pred_info['pred_threshold'])}"
            final_string = f"{final_string}\nConfidence: Above {str(pred_info['probability'])}%\n\n\n"

        if len(final_string) > 0:
          telegram_send.send(messages=[final_string], parse_mode="html")
        else:
          telegram_send.send(messages=["Nothing this time."])
        print("Adding tables to Database...")
        for x in shaped_tables.keys():
          if str(type(shaped_tables[x])) == "<class 'pandas.core.frame.DataFrame'>":

            shaped_tables[x].to_sql(x, engine, if_exists='append', index=True)
        print(f"Done at {datetime.now().strftime('%H:%M:%S')}.")

      else:
        print(f"All pairs already up to date at {datetime.now().strftime('%H:%M:%S')}.")

      repeat_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
      sleep(20)

