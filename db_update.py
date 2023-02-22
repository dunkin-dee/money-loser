import pandas as pd
import numpy as np
import rex


def add_dict_to_df(base_df, my_dict, prefix):
    """
    takes the dataframe to be added to, the dictionary to be added and prefix for column names and returns dataframe with added
    columns
    """
    my_df =  pd.DataFrame.from_dict(my_dict, orient="index")
    for col_name in my_df.columns:
        my_df.rename({col_name: prefix+col_name}, axis=1, inplace=True)
    return(pd.concat([base_df, my_df], axis=1))



def columnize_h(df, df_d):
    """
    Takes pandas dataframe for 1h or 4h periods and dataframe for 1 day periods, and returns df with added required columns.
    """
    singles = rex.get_single_candlestick_patterns(df)
    all_singles = []
    for x in singles.keys():
        all_singles = all_singles + singles[x]
    all_singles = set(all_singles)
    all_singles = sorted(list(all_singles))
    singles_series = pd.Series(dtype="float")
    for x in range(len(df)):
        if df.iloc[x].name in all_singles:
            singles_series.loc[df.iloc[x].name] = 1
        else:
            singles_series.loc[df.iloc[x].name] = 0
    df.insert(len(df.columns), "singles", singles_series)

    doubles = rex.get_double_candlestick_patterns(df)
    bull_doubles = doubles["bullish"]
    bear_doubles = doubles["bearish"]
    all_bull_doubles = []
    all_bear_doubles = []
    for x in bull_doubles.keys():
        all_bull_doubles = all_bull_doubles + bull_doubles[x]
    all_bull_doubles = set(all_bull_doubles)
    all_bull_doubles = sorted(list(all_bull_doubles))
    bull_doubles_series = pd.Series(dtype="float")
    for x in range(len(df)):
        if df.iloc[x].name in all_bull_doubles:
            bull_doubles_series.loc[df.iloc[x].name] = 1
        else:
            bull_doubles_series.loc[df.iloc[x].name] = 0
    df.insert(len(df.columns), "bull_doubles", bull_doubles_series)
    for x in bear_doubles.keys():
        all_bear_doubles = all_bear_doubles + bear_doubles[x]
    all_bear_doubles = set(all_bear_doubles)
    all_bear_doubles = sorted(list(all_bear_doubles))
    bear_doubles_series = pd.Series(dtype="float")
    for x in range(len(df)):
        if df.iloc[x].name in all_bear_doubles:
            bear_doubles_series.loc[df.iloc[x].name] = 1
        else:
            bear_doubles_series.loc[df.iloc[x].name] = 0
    df.insert(len(df.columns), "bear_doubles", bear_doubles_series)
    
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
    

    tp = rex.get_turns(df)
    tp_keys = sorted(list(tp.keys()))
    level_dict = {}
    for index, row in df.iterrows():
        stop_point = tp_keys[0]
        stop_point = next((x for x in tp_keys if x > index ), None)
        if stop_point:
            tp_index = tp_keys.index(stop_point)
            if tp_index > 4:
                level_list = []
                working_keys = tp_keys[tp_index-5:tp_index]
                for some_key in working_keys:
                    #getting the 5 levels, dependent on movement of turn
                    if tp[some_key] == "up":
                        level_list.append(min([float(df.loc[some_key, "open"]), float(df.loc[some_key, "close"])]))
                    else:
                        level_list.append(max([float(df.loc[some_key, "open"]), float(df.loc[some_key, "close"])]))
                level_dict[index] = {"level_1": level_list[0],
                                     "level_2": level_list[1],
                                     "level_3": level_list[2],
                                     "level_4": level_list[3],
                                     "level_5": level_list[4]}
        else:
            level_list = []
            working_keys = tp_keys[-5:]
            for some_key in working_keys:
                #getting the 5 levels, dependent on movement of turn
                if tp[some_key] == "up":
                    level_list.append(min([float(df.loc[some_key, "open"]), float(df.loc[some_key, "close"])]))
                else:
                    level_list.append(max([float(df.loc[some_key, "open"]), float(df.loc[some_key, "close"])]))
            level_dict[index] = {"level_1": level_list[0],
                                 "level_2": level_list[1],
                                 "level_3": level_list[2],
                                 "level_4": level_list[3],
                                 "level_5": level_list[4]}
    tp_levels_df = pd.DataFrame.from_dict(level_dict, orient="index")
    df = pd.concat([df, tp_levels_df], axis=1)

    daily_keys = sorted(list(df_d.index))
    pp_dict = {}
    for index, row in df.iterrows():
        today_index = daily_keys.index(pd.Timestamp(pd.to_datetime(index).date()))
        if today_index > 0:
            yesterday = daily_keys[today_index-1]
            yesterdict = {"open": df_d.loc[yesterday, "open"],
                          "high": df_d.loc[yesterday, "high"],
                          "low": df_d.loc[yesterday, "low"],
                          "close": df_d.loc[yesterday, "close"]}
            all_pp = rex.get_pp_daily(yesterdict)
            pp_dict[index] = {"pp" : all_pp["daily"]["standard"]["pp"],
                              "standard_r1" : all_pp["daily"]["standard"]["r1"],
                              "standard_s1" : all_pp["daily"]["standard"]["s1"],
                              "fib_r1" : all_pp["daily"]["fibonacci"]["r1"],
                              "fib_s1" : all_pp["daily"]["fibonacci"]["s1"],}
    pp_df = pd.DataFrame.from_dict(pp_dict,orient='index')
    df = pd.concat([df, pp_df], axis=1)

    for ma in [5 ,10]:
        sma = rex.get_sma(df, ma)
        ema = rex.get_ema(df, ma)
        df = pd.concat([df, pd.Series(sma).to_frame(f"sma{ma}"), pd.Series(ema).to_frame(f"ema{ma}")], axis=1)

    bb = rex.get_bollinger(df)
    df = add_dict_to_df(df, bb, "bb_")

    kelt = rex.get_keltner(df)
    df = add_dict_to_df(df, kelt, "kelt_")

    macd = rex.get_macd(df)
    df = add_dict_to_df(df, macd, "macd_")

    rsi = rex.get_rsi(df)
    df = pd.concat([df, pd.Series(rsi).to_frame("rsi")], axis=1)

    psar = rex.get_parabolic_sar(df)
    df = add_dict_to_df(df, psar, "psar_")

    stoch = rex.get_stochastic(df)
    df = add_dict_to_df(df, stoch, "stoch_")

    adx = rex.get_adx(df)
    df = add_dict_to_df(df, adx, "adx_")

    will = rex.get_williams_r(df)
    df = pd.concat([df, pd.Series(will).to_frame("williams")], axis=1)

    turns = rex.get_turns(df)
    turns_keys = sorted(list(turns.keys()))
    turns_series = pd.Series(dtype="object")
    for index, row in df.iterrows():
        stop_time = next((x for x in turns_keys if x > index), None)
        if stop_time:
            future_index = turns_keys.index(stop_time)
            if future_index > 0:
                use_time = turns_keys[future_index-1]
                turns_series.loc[index] = turns[use_time]
        else:
            turns_series[index] = turns[turns_keys[-1]]
    df = pd.concat([df, turns_series.shift(3).to_frame('curr_trend')], axis=1)
    
    #getting 3 and 6 period sums of signed bodysizes
    df['bodysize'] = np.where(df['direction'] == "up" , df["bodysize"], (df["bodysize"]*-1))
    for list_iter in range(1, 6):
        temp_df = df[["bodysize", "wick", "shadow"]].copy(deep=True)
        temp_df.rename({"bodysize":"bodysize"+str(list_iter),
                        "wick":"wick"+str(list_iter),
                        "shadow":"shadow"+str(list_iter)}, axis=1, inplace=True)
        df = pd.concat([df, temp_df], axis=1)
        
    df["bs3"] = df["bodysize"] + df["bodysize1"] +df["bodysize2"]
    df["bs6"] = df["bs3"] + df["bodysize3"] + df["bodysize4"] + df["bodysize5"]
    df["w3"] = (df["wick"] + df["wick1"] + df["wick2"])/3
    df["w6"] = (df["wick"] + df["wick1"] + df["wick2"] + df["wick3"] + df["wick4"] + df["wick5"])/6
    df["s3"] = (df["shadow"] + df["shadow1"] + df["shadow2"])/3
    df["s6"] = (df["shadow"] + df["shadow1"] + df["shadow2"] + df["shadow3"] + df["shadow4"] + df["shadow5"])/6
        
    

    df = pd.concat([df.drop(["direction", "curr_trend", "psar_direction"], axis=1), 
                    pd.get_dummies(df[["direction", "curr_trend","psar_direction"]])], axis=1)
    df.dropna(inplace=True)
    
    for list_iter in range(1, 6):
        df.drop([f"bodysize{str(list_iter)}", f"shadow{str(list_iter)}", f"wick{str(list_iter)}"], axis=1, inplace=True)

    return(df)   
        
    

#############################################################################################################################
#############################################################################################################################     
        
def columnize_d(df):
    singles = rex.get_single_candlestick_patterns(df)
    all_singles = []
    for x in singles.keys():
        all_singles = all_singles + singles[x]
    all_singles = set(all_singles)
    all_singles = sorted(list(all_singles))
    singles_series = pd.Series(dtype="float")
    for x in range(len(df)):
        if df.iloc[x].name in all_singles:
            singles_series.loc[df.iloc[x].name] = 1
        else:
            singles_series.loc[df.iloc[x].name] = 0
    df.insert(len(df.columns), "singles", singles_series)

    doubles = rex.get_double_candlestick_patterns(df)
    bull_doubles = doubles["bullish"]
    bear_doubles = doubles["bearish"]
    all_bull_doubles = []
    all_bear_doubles = []
    for x in bull_doubles.keys():
        all_bull_doubles = all_bull_doubles + bull_doubles[x]
    all_bull_doubles = set(all_bull_doubles)
    all_bull_doubles = sorted(list(all_bull_doubles))
    bull_doubles_series = pd.Series(dtype="float")
    for x in range(len(df)):
        if df.iloc[x].name in all_bull_doubles:
            bull_doubles_series.loc[df.iloc[x].name] = 1
        else:
            bull_doubles_series.loc[df.iloc[x].name] = 0
    df.insert(len(df.columns), "bull_doubles", bull_doubles_series)
    for x in bear_doubles.keys():
        all_bear_doubles = all_bear_doubles + bear_doubles[x]
    all_bear_doubles = set(all_bear_doubles)
    all_bear_doubles = sorted(list(all_bear_doubles))
    bear_doubles_series = pd.Series(dtype="float")
    for x in range(len(df)):
        if df.iloc[x].name in all_bear_doubles:
            bear_doubles_series.loc[df.iloc[x].name] = 1
        else:
            bear_doubles_series.loc[df.iloc[x].name] = 0
    df.insert(len(df.columns), "bear_doubles", bear_doubles_series)
    
    triples = rex.get_triple_candlestick_patterns(df)
    bull_triples = triples["bullish"]
    bear_triples = triples["bearish"]
    bull_triples = bull_triples | bear_triples
    bull_dict = {}
    for bull_type in bull_triples.keys():
        temp_list = []
        try:
            for temp_tuple in bull_triples[bull_type]:
                temp_list.append(temp_tuple[0])
            bull_dict[bull_type] = temp_list
        except IndexError:
            df[bull_type] = 0
    for bull_type in bull_dict.keys():
        df[bull_type] = 0
        for index in bull_dict[bull_type]:
            df.loc[index, bull_type] = 1

    tp = rex.get_turns(df)
    tp_keys = sorted(list(tp.keys()))
    level_dict = {}
    for index, row in df.iterrows():
        stop_point = tp_keys[0]
        stop_point = next((x for x in tp_keys if x > index ), None)
        if stop_point:
            tp_index = tp_keys.index(stop_point)
            if tp_index > 4:
                level_list = []
                working_keys = tp_keys[tp_index-5:tp_index]
                for some_key in working_keys:
                    #getting the 5 levels, dependent on movement of turn
                    if tp[some_key] == "up":
                        level_list.append(min([float(df.loc[some_key, "open"]), float(df.loc[some_key, "close"])]))
                    else:
                        level_list.append(max([float(df.loc[some_key, "open"]), float(df.loc[some_key, "close"])]))
                level_dict[index] = {"level_1": level_list[0],
                                     "level_2": level_list[1],
                                     "level_3": level_list[2],
                                     "level_4": level_list[3],
                                     "level_5": level_list[4]}
        else:
            level_list = []
            working_keys = tp_keys[-5:]
            for some_key in working_keys:
                #getting the 5 levels, dependent on movement of turn
                if tp[some_key] == "up":
                    level_list.append(min([float(df.loc[some_key, "open"]), float(df.loc[some_key, "close"])]))
                else:
                    level_list.append(max([float(df.loc[some_key, "open"]), float(df.loc[some_key, "close"])]))
            level_dict[index] = {"level_1": level_list[0],
                                 "level_2": level_list[1],
                                 "level_3": level_list[2],
                                 "level_4": level_list[3],
                                 "level_5": level_list[4]}
    tp_levels_df = pd.DataFrame.from_dict(level_dict, orient="index")
    df = pd.concat([df, tp_levels_df], axis=1)

    
    for ma in [5 ,10]:
        sma = rex.get_sma(df, ma)
        ema = rex.get_ema(df, ma)
        df = pd.concat([df, pd.Series(sma).to_frame(f"sma{ma}"), pd.Series(ema).to_frame(f"ema{ma}")], axis=1)

    bb = rex.get_bollinger(df)
    df = add_dict_to_df(df, bb, "bb_")

    kelt = rex.get_keltner(df)
    df = add_dict_to_df(df, kelt, "kelt_")

    macd = rex.get_macd(df)
    df = add_dict_to_df(df, macd, "macd_")

    rsi = rex.get_rsi(df)
    df = pd.concat([df, pd.Series(rsi).to_frame("rsi")], axis=1)

    psar = rex.get_parabolic_sar(df)
    df = add_dict_to_df(df, psar, "psar_")

    stoch = rex.get_stochastic(df)
    df = add_dict_to_df(df, stoch, "stoch_")

    adx = rex.get_adx(df)
    df = add_dict_to_df(df, adx, "adx_")

    will = rex.get_williams_r(df)
    df = pd.concat([df, pd.Series(will).to_frame("williams")], axis=1)

    turns = rex.get_turns(df)
    turns_keys = sorted(list(turns.keys()))
    turns_series = pd.Series(dtype="object")
    for index, row in df.iterrows():
        stop_time = next((x for x in turns_keys if x > index), None)
        if stop_time:
            future_index = turns_keys.index(stop_time)
            if future_index > 0:
                use_time = turns_keys[future_index-1]
                turns_series.loc[index] = turns[use_time]
        else:
            turns_series[index] = turns[turns_keys[-1]]
    df = pd.concat([df, turns_series.shift(3).to_frame('curr_trend')], axis=1)
    
    df['bodysize'] = np.where(df['direction'] == "up" , df["bodysize"], (df["bodysize"]*-1))
    for list_iter in range(1, 6):
        temp_df = df[["bodysize", "wick", "shadow"]].copy(deep=True)
        temp_df.rename({"bodysize":"bodysize"+str(list_iter),
                        "wick":"wick"+str(list_iter),
                        "shadow":"shadow"+str(list_iter)}, axis=1, inplace=True)
        df = pd.concat([df, temp_df], axis=1)
        
    df["bs3"] = df["bodysize"] + df["bodysize1"] +df["bodysize2"]
    df["bs6"] = df["bs3"] + df["bodysize3"] + df["bodysize4"] + df["bodysize5"]
    df["w3"] = (df["wick"] + df["wick1"] + df["wick2"])/3
    df["w6"] = (df["wick"] + df["wick1"] + df["wick2"] + df["wick3"] + df["wick4"] + df["wick5"])/6
    df["s3"] = (df["shadow"] + df["shadow1"] + df["shadow2"])/3
    df["s6"] = (df["shadow"] + df["shadow1"] + df["shadow2"] + df["shadow3"] + df["shadow4"] + df["shadow5"])/6

    df = pd.concat([df.drop(["direction", "curr_trend", "psar_direction"], axis=1), 
                    pd.get_dummies(df[["direction", "curr_trend","psar_direction"]])], axis=1)

    for list_iter in range(1, 6):
        df.drop([f"bodysize{str(list_iter)}", f"shadow{str(list_iter)}", f"wick{str(list_iter)}"], axis=1, inplace=True)
    df.dropna(inplace=True)
    
    return(df)
