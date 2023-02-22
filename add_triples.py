import pandas as pd
import numpy as np
import rex
from sqlalchemy import create_engine
from datetime import datetime

pd.set_option('display.max_rows', 1000)
engine = create_engine("mysql+pymysql://rex:#Pass123@localhost/rex_ml")


all_pairs = ["AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDUSD", "CADJPY", "CHFJPY", "EURAUD", "EURCAD", "EURCHF",
             "EURGBP", "EURJPY", "EURUSD", "GBPAUD", "GBPJPY", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY"]

time_periods = ["1h", "4h", "1d"]

for pair in all_pairs:
    for time_period in time_periods:
        table_name = (pair + "_" + time_period).lower()
        print("Working on "+table_name+" at "+ datetime.now().strftime("%H:%M:%S"))

        sql = f"SELECT * FROM `{table_name}` ORDER BY `index` ASC"
        df = pd.read_sql(sql, engine, index_col="index")
        #direction required for rex.get_triple_candlesticks
        df['direction'] = np.where(df['open'] > df['close'] , 'down', 'up')

        triples = rex.get_triple_candlestick_patterns(df)
        bull_triples = triples["bullish"]
        bear_triples = triples["bearish"]
        bull_triples = bull_triples | bear_triples
        bull_dict = {}


        #getting timestamps from tuples in returned dictionaries
        for bull_type in bull_triples.keys():
            temp_list = []
            try:
                for temp_tuple in bull_triples[bull_type]:
                    temp_list.append(temp_tuple[0])
                bull_dict[bull_type] = temp_list
            except IndexError:
                df[bull_type] = 0
        #adding to dataframes 1 for each column where pattern appears
        for bull_type in bull_dict.keys():
            df[bull_type] = 0
            for index in bull_dict[bull_type]:
                df.loc[index, bull_type] = 1
        #dropping direction
        df.drop("direction", axis=1, inplace=True)

        print("Creating mysql table for "+table_name)
        df.to_sql(table_name, con=engine, if_exists='replace', index=True)
        with engine.connect() as con:
            con.execute(f'ALTER TABLE {table_name} ADD PRIMARY KEY(`index`)')
print("Completed at "+ datetime.now().strftime("%H:%M:%S"))
