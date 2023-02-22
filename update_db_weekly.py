import time
import pandas
from sqlalchemy import create_engine, text
from sqlalchemy import inspect
from get_candles import get_candles

from db_creds import *

instruments = [
  "GBP_USD",
  "EUR_USD",
  "USD_CHF",
  "USD_JPY",
  "USD_CAD",
  "AUD_USD",
  "EUR_CHF",
  "EUR_JPY",
  "EUR_GBP",
  "EUR_CAD",
  "GBP_CHF",
  "GBP_JPY",
  "AUD_JPY"
]
engine = create_engine("mysql+pymysql://%s:%s@localhost/%s"%(db_username, db_pass, db_name))
insp = inspect(engine)
table_names = insp.get_table_names()
create_primary_key = False

for instrument in instruments:
  candles = get_candles(instrument, count=200, gran="W")
  table_name = lower_case = instrument.lower()+"_w"
  if table_name in table_names:
    create_primary_key = False
  else:
    create_primary_key = True
  
  try:
    if create_primary_key:
      print("Creating "+table_name)
      candles.to_sql(table_name, engine, if_exists='append', index=True)
      with engine.connect() as con:
        con.execute('ALTER TABLE `%s` ADD PRIMARY KEY(`index`)'%table_name)
    else:
      print("Adding to "+table_name)
      with engine.connect() as con:
        t = "SELECT * FROM %s ORDER BY `index` DESC LIMIT 10"%table_name
        last_times = pandas.read_sql(t, engine, index_col='index')
        last_time = max(list(last_times.index))
        last_loc = candles.index.get_loc(last_time)

        appendable_candles = candles[last_loc+1:]
        appendable_candles.to_sql(table_name, engine, if_exists='append', index=True)
      
  except Exception as e:
    print(e)

  time.sleep(3)


engine.dispose()