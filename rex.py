from datetime import time
from numpy import average
import pandas as pd




def get_turns(candles, fac=3, ml_adjustment=False):
    turns = {}
    ml_turns = {}

    first_df = candles[0:fac]
    second_df = candles[fac:fac*2]

    first_sum = first_df["close"].sum()
    second_sum = second_df["close"].sum()

    if (second_sum/fac) > (first_sum/fac):
        prev_direction = "up"
    else:
        prev_direction = "down"
    
    for x in range(1, len(candles)-(fac*2)+1):
        value_times = []

        df_1 = candles[x:x+fac]
        df_2 = candles[x+fac:x+fac+fac]

        mean_1 = df_1["close"].sum()/fac
        mean_2 = df_2["close"].sum()/fac

        if mean_2 > mean_1:
            curr_direction = "up"
        else:
            curr_direction = "down"

        if not curr_direction == prev_direction:


            value_times.append((df_1.iloc[-1].name, df_1.iloc[-1]["close"]))
            value_times.append((df_1.iloc[-1].name, df_1.iloc[-1]["open"]))
            value_times.append((df_1.iloc[-2].name, df_1.iloc[-2]["close"]))
            value_times.append((df_1.iloc[-2].name, df_1.iloc[-2]["open"]))
            value_times.append((df_2.iloc[0].name, df_2.iloc[0]["close"]))
            value_times.append((df_2.iloc[0].name, df_2.iloc[0]["open"]))
            value_times.append((df_2.iloc[1].name, df_2.iloc[1]["close"]))
            value_times.append((df_2.iloc[1].name, df_2.iloc[1]["open"]))

            # if df_2['bodysize'].sum()/fac > df_1['bodysize'].sum()/fac:
            #     value_times.append((df_1.iloc[-1].name, df_1.iloc[-1]["close"]))
            #     value_times.append((df_1.iloc[-1].name, df_1.iloc[-1]["open"]))
            #     for index, row in df_2.iterrows():
            #         value_times.append((index, row["close"]))
            #         value_times.append((index, row["open"]))
            
            # if df_2['bodysize'].sum()/fac < df_1['bodysize'].sum()/fac:
            #     value_times.append((df_2.iloc[0].name, df_2.iloc[0]["close"]))
            #     value_times.append((df_2.iloc[0].name, df_2.iloc[0]["open"]))
            #     for index, row in df_1.iterrows():
            #         value_times.append((index, row["close"]))
            #         value_times.append((index, row["open"]))

            (final_time, final_value) = value_times[0]
            if curr_direction == "up":
                for (temp_time, temp_value) in value_times:
                    if temp_value <= final_value:
                        final_time = temp_time
                        final_value = temp_value
                turns[final_time] = "up"
            if curr_direction == "down":
                for (temp_time, temp_value) in value_times:
                    if temp_value >= final_value:
                        final_time = temp_time
                        final_value = temp_value
                turns[final_time] = "down"
            prev_direction = curr_direction

            ml_turns[df_2.iloc[-1].name] = turns[final_time]

    if ml_adjustment:
        return(ml_turns)
    else:
        return(turns)

def get_heikin_ashi(candles):
    times = sorted(list(candles.index))
    
    column_names = ['open', 'high', 'low', 'close', 'bodysize', 'direction']
    new_candles = pd.DataFrame(columns=column_names)

    for x in range(1, len(times)):
        open = (candles.loc[times[x-1], "open"] + candles.loc[times[x-1], "close"])/2
        close = (candles.loc[times[x], "open"]+candles.loc[times[x], "high"]+candles.loc[times[x], "low"]+candles.loc[times[x], "close"])/4
        high = max([candles.loc[times[x], "high"], open, close])
        low = min([candles.loc[times[x], "low"], open, close])
        bodysize = abs(open-close)
        if open < close:
            direction = "up"
        elif close < open:
            direction = "down"
        else:
            direction = "none"

        new_candles.loc[times[x]] = [
            open,
            high,
            low,
            close,
            bodysize,
            direction
        ]

    return(new_candles)

def get_smoothed_turns(candles, fac=6, smoothing_fac=2, get_turns=get_turns, get_heikin_ashi=get_heikin_ashi):

    times = sorted(list(candles.index))
    mean_candle = candles["bodysize"].sum()/len(candles)
    h_candles = get_heikin_ashi(candles)
    h_turns = get_turns(h_candles, fac=fac)
    h_turn_times = sorted(list(h_turns.keys()))
    h_times = sorted(list(h_candles.index))
    turns = {}
    final_turns = {}

    for time in h_turn_times:
        h_values = []
        h_index = h_times.index(time)

        h_values.append((h_times[h_index], candles.loc[h_times[h_index], "open"]))
        h_values.append((h_times[h_index], candles.loc[h_times[h_index], "close"]))
        h_values.append((h_times[h_index-1], candles.loc[h_times[h_index-1], "open"]))
        h_values.append((h_times[h_index-1], candles.loc[h_times[h_index-1], "close"]))
        h_values.append((h_times[h_index+1], candles.loc[h_times[h_index+1], "open"]))
        h_values.append((h_times[h_index+1], candles.loc[h_times[h_index+1], "close"]))


        (final_time, final_value) = h_values[0]
        if h_turns[time] == "up":
            for (temp_time, temp_value) in h_values:
                if temp_value <= final_value:
                    final_value = temp_value
                    final_time = temp_time
            turns[final_time] = "up"

        else:
            for (temp_time, temp_value) in h_values:
                if temp_value >= final_value:
                    final_value = temp_value
                    final_time = temp_time
            turns[final_time] = "down"

    turn_times = sorted(list(turns.keys()))
    final_times_exclude = []

    for x in range(0, len(turn_times) - 2):
        current_low = min([candles.loc[turn_times[x], "close"], candles.loc[turn_times[x], "open"]])
        next_low = min([candles.loc[turn_times[x+2], "close"], candles.loc[turn_times[x+2], "open"]]) 
        current_high = max([candles.loc[turn_times[x], "close"], candles.loc[turn_times[x], "open"]])
        next_high = max([candles.loc[turn_times[x+2], "close"], candles.loc[turn_times[x+2], "open"]]) 
        mid_high = max([candles.loc[turn_times[x+1], "close"], candles.loc[turn_times[x+1], "open"]])
        mid_low = min([candles.loc[turn_times[x+1], "close"], candles.loc[turn_times[x+1], "open"]])

        if turns[turn_times[x]] == "up" and turns[turn_times[x+2]] == "up":
            if abs(current_low - mid_high) < smoothing_fac*mean_candle:
                if next_low < current_low:
                    final_times_exclude.append(turn_times[x+1])
                    final_times_exclude.append(turn_times[x])
        elif turns[turn_times[x]] == "down" and turns[turn_times[x+2]] == "down":
            if abs(current_high - mid_low) < smoothing_fac*mean_candle:
                if next_high > current_high:
                    final_times_exclude.append(turn_times[x+1])
                    final_times_exclude.append(turn_times[x])
            
    for time in turn_times:
        if not time in final_times_exclude:
            final_turns[time] = turns[time]
    return(final_turns)

def get_single_candlestick_patterns(df):
    """
    accepts dataframe with bodysize, wick, shadow and returns dict
    """
    df2 = df.copy(deep=True)
    mbs = df['bodysize'].mean()
    
    df2.loc[(df2["wick"]<df2["bodysize"]*1.5) & (df2["shadow"]>df2["bodysize"]*3) & (df2["bodysize"]<mbs/2), "hammers"] = 1
    df2.loc[(df2["wick"]>df2["bodysize"]*3) & (df2["shadow"]<df2["bodysize"]*1.5) & (df2["bodysize"]<mbs/2), "ihammers"] = 1

    patterns = {}
    patterns["hammers"] = list(df2[df2["hammers"]==1].index)
    patterns["ihammers"] = list(df2[df2["ihammers"]==1].index)
    return(patterns)

def get_double_candlestick_patterns(df):
    """
    Accepts dataframe with bodysize, direction, wick, shadow and returns dict
    """
    mbs = df["bodysize"].mean()
    df2 = df.copy(deep=True)
    df3 = df2.shift(1)
    for col_name in df3.columns:
        df3.rename({col_name:"prev"+col_name},axis='columns',inplace=True)
    df2 = pd.concat([df2, df3], axis=1)
    
    df2.loc[(df2["direction"]=="up")&(df2["prevdirection"]=="down")&(df2["bodysize"]>df2["prevbodysize"]),"bullish_engulfing"] = 1
    df2.loc[(df2["direction"]=="down")&(df2["prevdirection"]=="up")&(df2["bodysize"]>df2["prevbodysize"]),"bearish_engulfing"] = 1
    df2.loc[(df2["wick"]>mbs)&(df2["prevwick"]>mbs)&(df["shadow"]<mbs/4),'tweezer_tops'] = 1
    df2.loc[(df2["shadow"]>mbs)&(df2["prevshadow"]>mbs)&(df["wick"]<mbs/4),'tweezer_bottoms'] = 1
    
    bullish={}
    bearish={}
    
    bullish["bullish_engulfing"] = list(df2[df2["bullish_engulfing"]==1].index)
    bullish["tweezer_bottoms"] = list(df2[df2["tweezer_bottoms"]==1].index)
    bearish["bearish_engulfing"] = list(df2[df2["bearish_engulfing"]==1].index)
    bearish["tweezer_tops"] = list(df2[df2["tweezer_tops"]==1].index)
    return({"bullish":bullish, "bearish":bearish})

def get_triple_candlestick_patterns(candles, fac=5):
    mean_body_size =  candles['bodysize'].mean()
    times = sorted(list(candles.index))
    morning_stars = set()
    evening_stars = set()
    white_soldiers = set()
    black_crows = set()
    inside_up = set()
    inside_down = set()

    # getting a rolling list of length fac from within the times list
    for list_x in range(fac-1, len(times)):
        working_times = times[list_x-fac+1:list_x+1]
        #performin checks within that rolling list for spreadout triple candlesticks

        #morning star
        morning_star_working_times = working_times
        first_morning_star = None
        second_morning_star = None
        third_morning_star = None
        first_morning_star = next(filter(lambda time: candles.loc[time]["direction"] == "down" and candles.loc[time]['bodysize']>=mean_body_size/2, morning_star_working_times), None)
        if first_morning_star:
            morning_star_working_times = morning_star_working_times[morning_star_working_times.index(first_morning_star)+1:]
            second_morning_star = next(filter(lambda time: candles.loc[time]["bodysize"] < (mean_body_size/3), morning_star_working_times), None)
        
        if second_morning_star:
            morning_star_working_times = morning_star_working_times[morning_star_working_times.index(second_morning_star)+1:]
            third_morning_star = next(filter(lambda time: candles.loc[time]["direction"] == "up", morning_star_working_times), None)

        if third_morning_star:
            if candles.loc[third_morning_star]["close"] > (candles.loc[first_morning_star]["close"]+(candles.loc[first_morning_star]["bodysize"]/2)):
                morning_stars.add((third_morning_star, (times.index(third_morning_star)-times.index(first_morning_star))))
        
        #evening star
        evening_star_working_times = working_times
        first_evening_star = None
        second_evening_star = None
        third_evening_star = None
        first_evening_star = next(filter(lambda time: candles.loc[time]["direction"] == "up" and candles.loc[time]['bodysize']>=mean_body_size/2, evening_star_working_times), None)
        if first_evening_star:
            evening_star_working_times = evening_star_working_times[evening_star_working_times.index(first_evening_star)+1:]
            second_evening_star = next(filter(lambda time: candles.loc[time]["bodysize"] < (mean_body_size/3), evening_star_working_times), None)
        
        if second_evening_star:
            evening_star_working_times = evening_star_working_times[evening_star_working_times.index(second_evening_star)+1:]
            third_evening_star = next(filter(lambda time: candles.loc[time]["direction"] == "down", evening_star_working_times), None)

        if third_evening_star:
            if candles.loc[third_evening_star]["close"] < (candles.loc[first_evening_star]["close"]-(candles.loc[first_evening_star]["bodysize"]/2)):
                evening_stars.add((third_evening_star, (times.index(third_evening_star)-times.index(first_evening_star))))

        #white soldiers
        white_soldiers_working_times = working_times
        first_white_soldier = None
        second_white_soldier = None
        third_white_soldier = None
        first_white_soldier = next(filter(lambda time: candles.loc[time]["direction"] == "up" and candles.loc[time]['bodysize']>(mean_body_size/4), white_soldiers_working_times), None)
        if first_white_soldier:
            white_soldiers_working_times = white_soldiers_working_times[white_soldiers_working_times.index(first_white_soldier)+1:]
            second_white_soldier = next(filter(lambda time: candles.loc[time]["direction"] == "up" and (candles.loc[time]["high"]-candles.loc[time]["close"]) < (mean_body_size/4), white_soldiers_working_times), None)
            
        if second_white_soldier:
            if candles.loc[second_white_soldier]["bodysize"] > candles.loc[first_white_soldier]["bodysize"]:
                white_soldiers_working_times = white_soldiers_working_times[white_soldiers_working_times.index(second_white_soldier)+1:]
                third_white_soldier = next(filter(lambda time: candles.loc[time]["direction"] == "up" and (candles.loc[time]["open"]-candles.loc[time]["low"])<(mean_body_size/2), white_soldiers_working_times), None)
            else:
                second_white_soldier = None

        if third_white_soldier and second_white_soldier:
            if candles.loc[third_white_soldier]["bodysize"] >= candles.loc[second_white_soldier]["bodysize"]:
                if abs(times.index(third_white_soldier) - times.index(first_white_soldier)) < 3:
                    white_soldiers.add((third_white_soldier, 3))
        
        #black crows
        black_crows_working_times = working_times
        first_black_crow = None
        second_black_crow = None
        third_black_crow = None
        first_black_crow = next(filter(lambda time: candles.loc[time]["direction"] == "down" and candles.loc[time]['bodysize']>(mean_body_size/4), black_crows_working_times), None)
        if first_black_crow:
            black_crows_working_times = black_crows_working_times[black_crows_working_times.index(first_black_crow)+1:]
            second_black_crow = next(filter(lambda time: candles.loc[time]["direction"] == "down" and (candles.loc[time]["close"]-candles.loc[time]["low"]) < (mean_body_size/4), black_crows_working_times), None)
            
        if second_black_crow:
            if candles.loc[second_black_crow]["bodysize"] > candles.loc[first_black_crow]["bodysize"]:
                black_crows_working_times = black_crows_working_times[black_crows_working_times.index(second_black_crow)+1:]
                third_black_crow = next(filter(lambda time: candles.loc[time]["direction"] == "down" and (candles.loc[time]["close"]-candles.loc[time]["low"])<(mean_body_size/2), black_crows_working_times), None)
            else:
                second_black_crow = None

        if third_black_crow and second_black_crow:
            if candles.loc[third_black_crow]["bodysize"] >= candles.loc[second_black_crow]["bodysize"]:
                if abs(times.index(third_black_crow) - times.index(first_black_crow)) < 3:
                    black_crows.add((third_black_crow, 3))
 
        #inside up
        inside_up_working_times = working_times
        first_inside_up = None
        second_inside_up = None
        third_inside_up = None
        first_inside_up = next(filter(lambda time: candles.loc[time]["direction"] == "down" and candles.loc[time]["bodysize"] >= mean_body_size, inside_up_working_times),None)
        if first_inside_up:
            inside_up_working_times = inside_up_working_times[inside_up_working_times.index(first_inside_up)+1:]
            second_inside_up = next(filter(lambda time: candles.loc[time]["direction"] == "up", inside_up_working_times), None)

        if second_inside_up:
            if candles.loc[second_inside_up]["close"] >= candles.loc[first_inside_up]["close"] + (candles.loc[first_inside_up]["bodysize"]/2):
                inside_up_working_times = inside_up_working_times[inside_up_working_times.index(second_inside_up)+1:]
                third_inside_up = next(filter(lambda time: candles.loc[time]["direction"] == "up" and candles.loc[time]["close"] > candles.loc[first_inside_up]["high"], inside_up_working_times), None)
            else:
                second_inside_up = None

        if third_inside_up and second_inside_up:
            inside_up.add((third_inside_up, (times.index(third_inside_up)-times.index(first_inside_up))))
             

        #inside down
        inside_down_working_times = working_times
        first_inside_down = None
        second_inside_down = None
        third_inside_down = None
        first_inside_down = next(filter(lambda time: candles.loc[time]["direction"] == "up" and candles.loc[time]["bodysize"] >= mean_body_size, inside_down_working_times),None)
        if first_inside_down:
            inside_down_working_times = inside_down_working_times[inside_down_working_times.index(first_inside_down)+1:]
            second_inside_down = next(filter(lambda time: candles.loc[time]["direction"] == "down", inside_down_working_times),None)

        if second_inside_down:
            if candles.loc[second_inside_down]["close"] < candles.loc[first_inside_down]["open"] + (candles.loc[first_inside_down]["bodysize"]/2):
                inside_down_working_times = inside_down_working_times[inside_down_working_times.index(second_inside_down)+1:]
                third_inside_down = next(filter(lambda time: candles.loc[time]["direction"] == "down" and candles.loc[time]["close"] < candles.loc[first_inside_down]["low"], inside_down_working_times), None)
            else:
                second_inside_down = None

        if third_inside_down and second_inside_down:
            inside_down.add((third_inside_down, (times.index(third_inside_down) - times.index(first_inside_down))))


    
    bullish_candles = {
        "morningstars": list(morning_stars),
        "whitesoldiers": list(white_soldiers),
        "insideup": list(inside_up)
    }

    bearish_candles = {
        "eveningstars": list(evening_stars),
        "blackcrows": list(black_crows),
        "insidedown": list(inside_down)
    }

    return({
        "bullish": bullish_candles,
        "bearish": bearish_candles
    })

def get_sma(candles, fac=5):
    """
    takes dataframe with column labeled close
    returns dictionary
    """
    averages = pd.Series(candles["close"])
    for loop_iter in range(1, fac):
        averages = averages + candles.shift(loop_iter)["close"]
    averages = averages/fac
    return(averages[fac-1:])

def get_ema(candles, fac=5, get_sma=get_sma):
    times = sorted(list(candles.index))
    emas = {}
    smas = (get_sma(candles, fac=fac))

    w_fac = 2/(fac+1)

    emas[times[fac-1]] = smas[times[fac-1]]
    working_times = times[fac:]

    for time in working_times:
        prev_time = times[times.index(time)-1]
        prev_ema = emas[prev_time]
        curr_price = candles.loc[time]["close"]

        ema = (w_fac * (curr_price - prev_ema)) + prev_ema

        emas[time] = ema

    return(emas)

def get_bollinger(candles, period=20, std_fac=2, get_sma=get_sma):
    bollingers = {}

    sma = get_sma(candles, fac=period)
    x = period

    for index, row in candles[period-1:].iterrows():
        std = candles[x-period:x]["close"].std()
      
        upper_band = sma[index] + (std*std_fac)
        lower_band = sma[index] - (std*std_fac)
        bollinger = {
            "center" : sma[index],
            "upper": upper_band,
            "lower": lower_band,
        }

        bollingers[index] = bollinger

        x += 1

    return bollingers

def get_atr(candles, fac=14):
    trs = {}
    atrs = {}

    #getting trs
    tr_1 = candles.iloc[0]["high"] - candles.iloc[0]["low"]
    trs[candles.iloc[0].name] = tr_1
    for index, row in candles[1:].iterrows():
        prev_row = candles.iloc[candles.index.get_loc(row.name)-1]

        meth_1 = row["high"] - row["low"]
        meth_2 = abs(row["high"] - prev_row["close"])
        meth_3 = abs(row["low"] - prev_row["close"])
        tr = max([meth_1, meth_2, meth_3])

        trs[index] = tr

    #getting atrs
    times = sorted(list(trs.keys()))
    atr_1_sum = 0

    for time in times[:fac]:
        atr_1_sum = atr_1_sum + trs[time]

    atr_1 = atr_1_sum/fac
    atrs[times[fac-1]] = atr_1

    for time in times[fac:]:
        prev_time = times[times.index(time)-1]
        atr = ((atrs[prev_time]*(fac-1)) + trs[time])/fac

        atrs[time] = atr

    return(atrs)

def get_keltner(candles, ema_fac=20, atr_fac=20, atr_multiplier=2, get_ema=get_ema, get_atr=get_atr):
    keltners = {}
    
    super_fac = max([ema_fac, atr_fac])
    times = sorted(list(candles.index))
    atrs = get_atr(candles, fac=super_fac)
    emas = get_ema(candles, fac=super_fac)

    for time in times[super_fac-1:]:
        keltner_center = emas[time]
        keltner_upper = keltner_center + (atrs[time] * 2)
        keltner_lower = keltner_center - (atrs[time] * 2)

        keltner = {
            "center": keltner_center,
            "upper": keltner_upper,
            "lower": keltner_lower,
        }

        keltners[time] = keltner

    return keltners

def get_macd(candles, get_ema=get_ema, ema_1_fac=12, ema_2_fac=26, signal_fac=9):
    macd = {}
    final_macd = {}
    times = sorted(list(candles.index))
    ema_1 = get_ema(candles, fac=ema_1_fac)
    ema_2 = get_ema(candles, fac=ema_2_fac)

    for time in times[ema_2_fac-1:]:
        macd[time] = ema_1[time] - ema_2[time]
   

    signal_times = sorted(list(macd.keys()))

    temp_candles = candles.copy(deep=True)
    for time in signal_times:
        temp_candles.loc[time, "close"] = macd[time]

    signal = get_ema(temp_candles[ema_2_fac-1:], fac=signal_fac)

    new_signal_times = sorted(list(signal.keys()))

    for time in new_signal_times:
        final_macd[time] = {
            "macd": macd[time],
            "signal": signal[time]
        }

    return final_macd

def get_stochastic(candles, fac=14, ma_fac=3, get_sma=get_sma):
    stochastic_1 = {}
    stochastic_final = {}
    
    for x in range(fac-1, len(candles.index)):
        temp_frame = candles[x-fac+1:x+1]
        closing = temp_frame.iloc[fac-1]["close"]
        high = temp_frame.max()["high"]
        low = temp_frame.min()["low"]

        stoch = ((closing - low)/(high-low))*100

        stochastic_1[temp_frame.iloc[fac-1].name] = stoch

    temp_candles = candles.copy(deep=True)
    for time in sorted(list(stochastic_1.keys())):
        temp_candles.loc[time, "close"] = stochastic_1[time]

    signals = get_sma(temp_candles[fac-1:], fac=ma_fac)
    
    for time in sorted(list(signals.keys())):
        stochastic_final[time]= {
            "stochastic":stochastic_1[time],
            "signal": signals[time]
        }

    return(stochastic_final)

def get_rsi(candles, fac=14):

    smoothing_fac = (1/fac)
    up_moves = {}
    down_moves = {}

    up_moves_averages = {}
    down_moves_averages = {}

    rsi_dic = {}
    
    up_move_1_sum = 0
    down_move_1_sum = 0

    for index, row in candles.iterrows():
        if row["direction"] == "up":
            up_moves[index] = row["bodysize"]
            down_moves[index] = 0
        elif row["direction"] == "down":
            down_moves[index] = row["bodysize"]
            up_moves[index] = 0
        else:
            down_moves[index] = 0
            up_moves[index] = 0
    

    moves_times = sorted(list(up_moves.keys()))

    #getting first moves averages

    for x in moves_times[:fac]:
        up_move_1_sum += up_moves[x]
        down_move_1_sum += down_moves[x]

    
    up_moves_averages[moves_times[fac-1]] = (up_move_1_sum/fac)
    down_moves_averages[moves_times[fac-1]] = (down_move_1_sum/fac)

    
    for x in range(fac, len(moves_times)):
        up_move_average = (smoothing_fac*up_moves[moves_times[x]])+((1-smoothing_fac)*up_moves_averages[moves_times[x-1]])
        down_move_average = (smoothing_fac*down_moves[moves_times[x]])+((1-smoothing_fac)*down_moves_averages[moves_times[x-1]])
        
        up_moves_averages[moves_times[x]] = up_move_average
        down_moves_averages[moves_times[x]] = down_move_average

    
    for time in sorted(list(up_moves_averages.keys())):
        try:
            rs = (up_moves_averages[time]/down_moves_averages[time])
            rsi = 100-(100/(1+rs))
        except(ZeroDivisionError):
            rsi = 99

        rsi_dic[time] = rsi

    return rsi_dic

def get_williams_r(candles, fac=14):
    williams = {}

    for x in range(fac-1, len(candles.index)):
        temp_frame = candles[x-fac+1:x+1]
        closing = temp_frame.iloc[fac-1]["close"]
        high = temp_frame.max()["high"]
        low = temp_frame.min()["low"]

        billy = (high-closing)/(high-low)
        williams[temp_frame.iloc[fac-1].name] = billy*-100

    return williams

def get_adx(candles, fac=14):
    times = sorted(list(candles.index))


    # getting average true range
    for x in range(1, len(times)):
        tr_1 = candles.loc[times[x], "high"] - candles.loc[times[x], "low"]
        tr_2 = abs(candles.loc[times[x], "high"] - candles.loc[times[x-1], "close"])
        tr_3 = abs(candles.loc[times[x], "low"] - candles.loc[times[x-1], "close"])
        tr = max(tr_1, tr_2, tr_3)
        candles.loc[times[x], "truerange"] = tr

    tr_a = candles[1:fac+1]["truerange"].sum()/fac
    candles.loc[times[fac], "atr"] = tr_a
    for x in range(fac+1, len(times)):
        atr = (((candles.loc[times[x-1], "atr"])*(fac-1))+candles.loc[times[x], "truerange"])/fac
        candles.loc[times[x], "atr"] = atr

    #getting pdx and ndx
    for x in range(1, len(times)):
        initial_pos = candles.loc[times[x], "high"] - candles.loc[times[x-1], "high"]
        initial_neg = candles.loc[times[x-1], "low"] - candles.loc[times[x], "low"]

        if initial_pos > initial_neg and initial_pos > 0:
            pdx = initial_pos
        else:
            pdx = 0

        if initial_neg > initial_pos and initial_neg > 0:
            ndx = initial_neg
        else:
            ndx = 0

        candles.loc[times[x], "pdx"] = pdx
        candles.loc[times[x], "ndx"] = ndx

    #getting pdx and ndx averages
    pdx_a = candles[1:fac+1]["pdx"].sum()/fac
    candles.loc[times[fac], "apdx"] = pdx_a
    ndx_a = candles[1:fac+1]["ndx"].sum()/fac
    candles.loc[times[fac], "andx"] = ndx_a

    for x in range(fac+1, len(times)):
        apdx = ((candles.loc[times[x-1], "apdx"]*(fac-1))+candles.loc[times[x], "pdx"])/fac
        andx = ((candles.loc[times[x-1], "andx"]*(fac-1))+candles.loc[times[x], "ndx"])/fac

        candles.loc[times[x], "apdx"] = apdx
        candles.loc[times[x], "andx"] = andx

    for x in range(fac, len(times)):

        apdx = candles.loc[times[x], "apdx"]
        andx = candles.loc[times[x], "andx"]

        pdmi = (apdx/candles.loc[times[x],"atr"])*100
        ndmi = (andx/candles.loc[times[x],"atr"])*100
        
        candles.loc[times[x], "pdmi"] = pdmi
        candles.loc[times[x], "ndmi"] = ndmi

        dx = ((abs(pdmi-ndmi))/(pdmi+ndmi))*100
        candles.loc[times[x], "dx"] = dx

    candles.loc[times[(fac*2)-1], "adx"] = candles[fac:fac*2]["dx"].sum()/fac


    for x in range((fac*2), len(times)):
        adx = ((candles.loc[times[x-1], "dx"] * (fac-1))+candles.loc[times[x], "dx"])/fac
        candles.loc[times[x], "adx"] = adx

    adx = {}

    for x in range((fac*2)-1, len(times)):
        adx[times[x]] = {
            "adx" : candles.loc[times[x], "adx"],
            "pdmi" : candles.loc[times[x], "pdmi"],
            "ndmi" : candles.loc[times[x], "ndmi"],
        }

    candles.drop(['truerange', 'atr', 'pdx', 'ndx', 'apdx', 'andx', 'pdmi', 'ndmi', 'dx', 'adx'], axis=1, inplace=True)
    return adx

def get_parabolic_sar(candles, af=0.02, max_af=0.2):

    times = sorted(list(candles.index))

    candles.loc[times[0], 'AF'] =af
    candles.loc[times[0], 'PSAR'] = candles.loc[times[0], 'high']
    candles.loc[times[0], 'EP'] = candles.loc[times[0], 'low']
    candles.loc[times[0], 'dir'] = "bear"
    candles.loc[times[0], "Peastar"] = (candles.loc[times[0], "PSAR"] - candles.loc[times[0], "EP"])* candles.loc[times[0], 'AF']

    for a in range(1, len(times)):

        if candles.loc[times[a-1], 'dir'] == 'bear':

            ipsar_1 = candles.loc[times[a-1], 'PSAR'] - candles.loc[times[a-1], 'Peastar']
            ipsar_2 = candles.loc[times[a-1], 'high']
            try:
                ipsar_3 = candles.loc[times[a-2], 'high']
            except:
                ipsar_3 = 0

            ipsar = max([ipsar_1, ipsar_2, ipsar_3])

        else:
            ipsar_1 = candles.loc[times[a-1], 'PSAR'] - candles.loc[times[a-1], 'Peastar']
            ipsar_2 = candles.loc[times[a-1], 'low']
            try:
                ipsar_3 = candles.loc[times[a-2], 'low']
            except:
                ipsar_3 = 999999999

            ipsar = min([ipsar_1, ipsar_2, ipsar_3])

        if candles.loc[times[a-1], "dir"] == "bear" and candles.loc[times[a], "high"] < ipsar:
            psar = ipsar

        elif candles.loc[times[a-1], "dir"] == "bull" and candles.loc[times[a], "low"] > ipsar:
            psar = ipsar

        elif candles.loc[times[a-1], "dir"] == "bear" and candles.loc[times[a], "high"] >= ipsar:
            psar = candles.loc[times[a-1], 'EP']

        elif candles.loc[times[a-1], "dir"] == "bull" and candles.loc[times[a], "low"] <= ipsar:
            psar = candles.loc[times[a-1], 'EP']

        candles.loc[times[a], "PSAR"] = psar

        if candles.loc[times[a], "PSAR"] > candles.loc[times[a], 'close']:
            candles.loc[times[a], 'dir'] = 'bear'
        else:
            candles.loc[times[a], 'dir'] = 'bull'
   

        if candles.loc[times[a], 'dir'] == "bear":
            ep = min([candles.loc[times[a-1], 'EP'], candles.loc[times[a], 'low']])
        elif candles.loc[times[a], 'dir'] == "bull":
            ep = max([candles.loc[times[a-1], 'EP'], candles.loc[times[a], 'high']])
        candles.loc[times[a], 'EP'] = ep

        if candles.loc[times[a-1], 'dir'] == candles.loc[times[a], 'dir']:
            if not candles.loc[times[a-1], 'EP'] == candles.loc[times[a], 'EP']:
                if candles.loc[times[a-1], 'AF'] < max_af:
                    candles.loc[times[a], 'AF'] = candles.loc[times[a-1], 'AF'] + af
                else:
                    candles.loc[times[a], 'AF'] = candles.loc[times[a-1], 'AF']
            else:
               candles.loc[times[a], 'AF'] = candles.loc[times[a-1] , 'AF']

        else:
            candles.loc[times[a], 'AF'] = af

        candles.loc[times[a], "Peastar"] = (candles.loc[times[a], "PSAR"] - candles.loc[times[a], "EP"])* candles.loc[times[a], 'AF']

    final_psar = {}

    for x in range(0, len(times)):
        final_psar[times[x]] = {
            "direction": candles.loc[times[x], "dir"],
            "psar": candles.loc[times[x], "PSAR"]
        }

    candles.drop([ 'AF', 'PSAR', 'EP', 'dir', 'Peastar'], axis=1, inplace=True)
    return(final_psar)

def check_for_patterns(candles, get_turns=get_turns, get_smoothed_turns=get_smoothed_turns, fac=3, dirty=True):
    patterns = {}
    if dirty:
        turns = get_turns(candles, fac=fac)
    else:
        turns = get_smoothed_turns(candles, fac=fac)
    mean_candle = candles["bodysize"].mean()
    times = sorted(list(candles.index))
    turn_times = sorted(list(turns.keys()))

    turn_1 = turns[turn_times[-1]]
    turn_2 = turns[turn_times[-2]]
    turn_3 = turns[turn_times[-3]]
    turn_4 = turns[turn_times[-4]]
    turn_5 = turns[turn_times[-5]]

    #check for double bottoms

    double_bottom = None


    if turn_1 == "up" and turn_2 == "down" and turn_3 == "up":
        bottom_1_index = times.index(turn_times[-1])
        bottom_2_index = times.index(turn_times[-3])

        bottom_low_1 = min([
            candles.loc[times[bottom_1_index], "low"],
            candles.loc[times[bottom_1_index-1], "low"],
            candles.loc[times[bottom_1_index+1], "low"],
            candles.loc[times[bottom_1_index+2], "low"],
            candles.loc[times[bottom_1_index-2], "low"],
        ])
        bottom_low_2 = min([
            candles.loc[times[bottom_2_index], "low"],
            candles.loc[times[bottom_2_index-1], "low"],
            candles.loc[times[bottom_2_index+1], "low"],
            candles.loc[times[bottom_2_index+2], "low"],
            candles.loc[times[bottom_2_index-2], "low"],
        ])

        bottoms_line_up = abs(bottom_low_1 - bottom_low_2) < 2*mean_candle
        outside = candles.loc[turn_times[-4]]
        inside = candles.loc[turn_times[-2]]
        if outside["direction"] == "up":
            outside_high = outside["close"]
        else:
            outside_high = outside["open"]
    
        if inside["direction"] == "up":
            inside_high = inside["close"]
        else:
            inside_high = inside["open"]

        outsides_good = abs(inside_high - outside_high) > 2*mean_candle

        print(inside_high, outside_high)
        if bottoms_line_up and outsides_good:
            double_bottom = {
                "entry" : inside_high,
                "exit" : inside_high + abs(inside_high - max([bottom_low_1, bottom_low_2]))
            }
    patterns["doublebottom"] = double_bottom

    #check for doubletops
    double_top = None
    if turn_1 == "down" and turn_2 == "up" and turn_3 == "down":
        top_1_index = times.index(turn_times[-1])
        top_2_index = times.index(turn_times[-3])
        top_high_1 = max([
            candles.loc[times[top_1_index], "high"],
            candles.loc[times[top_1_index-1], "high"],
            candles.loc[times[top_1_index+1], "high"],
            candles.loc[times[top_1_index+2], "high"],
            candles.loc[times[top_1_index-2], "high"],
        ])
        top_high_2= max([
            candles.loc[times[top_2_index], "high"],
            candles.loc[times[top_2_index-1], "high"],
            candles.loc[times[top_2_index+1], "high"],
            candles.loc[times[top_2_index+2], "high"],
            candles.loc[times[top_2_index-2], "high"],
        ])
        tops_line_up = abs(top_high_1 - top_high_2) < 2*mean_candle
        outside = candles.loc[turn_times[-4]]
        inside = candles.loc[turn_times[-2]]
        if outside["direction"] == "up":
            outside_low = outside["open"]
        else:
            outside_low = outside["close"]
        if inside["direction"] == "up":
            inside_low = inside["open"]
        else:
            inside_low = inside["close"]
        outsides_good = abs(inside_low - outside_low) > 2*mean_candle
        if tops_line_up and outsides_good:
            double_top = {
                "entry" : inside_low,
                "exit" : inside_low - abs(min([top_high_1, top_high_2]) - inside_low)
            }

    patterns["doubletop"] = double_top
    
    #check for head and shoulders
    head_and_shoulders = None
    if turn_1 == "down" and turn_2 == "up" and turn_3 =="down" and turn_4 == "up" and turn_5 == "down":
        top_1_index = times.index(turn_times[-1])
        top_2_index = times.index(turn_times[-3])
        top_3_index = times.index(turn_times[-5])
        top_high_1 = max([
            candles.loc[times[top_1_index], "high"],
            candles.loc[times[top_1_index-1], "high"],
            candles.loc[times[top_1_index+1], "high"],
            candles.loc[times[top_1_index+2], "high"],
            candles.loc[times[top_1_index-2], "high"],
        ])
        top_high_2= max([
            candles.loc[times[top_2_index], "high"],
            candles.loc[times[top_2_index-1], "high"],
            candles.loc[times[top_2_index+1], "high"],
            candles.loc[times[top_2_index+2], "high"],
            candles.loc[times[top_2_index-2], "high"],
        ])
        top_high_3= max([
            candles.loc[times[top_3_index], "high"],
            candles.loc[times[top_3_index-1], "high"],
            candles.loc[times[top_3_index+1], "high"],
            candles.loc[times[top_3_index+2], "high"],
            candles.loc[times[top_3_index-2], "high"],
        ])
    
        tops_line_up = top_high_2 > top_high_1 and top_high_2 > top_high_3
        
        outside = candles.loc[turn_times[-6]]
        inside_1 = candles.loc[turn_times[-2]]
        inside_2 = candles.loc[turn_times[-4]]
        if outside["direction"] == "up":
            outside_low = outside["open"]
        else:
            outside_low = outside["close"]
        
        if inside_1["direction"] == "up":
            inside_1_low = inside_1["open"]
        else:
            inside_1_low = inside_1["close"]
        
        if inside_2["direction"] == "up":
            inside_2_low = inside_2["open"]
        else:
            inside_2_low = inside_2["close"]

        outside_good = outside_low < inside_2_low and abs(outside_low - inside_2_low) > 2*mean_candle and abs(outside_low - inside_1_low) > 2*mean_candle

        if tops_line_up and outside_good:
            head_and_shoulders = {
                "gradient": inside_1_low <= inside_2_low,
                "entry": inside_1_low,
                "exit" : inside_1_low - abs(top_high_2 - max([inside_1_low, inside_2_low]))
            }

    patterns["has"] = head_and_shoulders
    
    #check for invers head and shoulders
    i_head_and_shoulders = None
    if turn_1 == "up" and turn_2 == "down" and turn_3 =="up" and turn_4 == "down" and turn_5 == "up":
        bottom_1_index = times.index(turn_times[-1])
        bottom_2_index = times.index(turn_times[-3])
        bottom_3_index = times.index(turn_times[-5])
        bottom_low_1 = min([
            candles.loc[times[bottom_1_index], "low"],
            candles.loc[times[bottom_1_index-1], "low"],
            candles.loc[times[bottom_1_index+1], "low"],
            candles.loc[times[bottom_1_index+2], "low"],
            candles.loc[times[bottom_1_index-2], "low"],
        ])
        bottom_low_2= min([
            candles.loc[times[bottom_2_index], "low"],
            candles.loc[times[bottom_2_index-1], "low"],
            candles.loc[times[bottom_2_index+1], "low"],
            candles.loc[times[bottom_2_index+2], "low"],
            candles.loc[times[bottom_2_index-2], "low"],
        ])
        bottom_low_3= min([
            candles.loc[times[bottom_3_index], "low"],
            candles.loc[times[bottom_3_index-1], "low"],
            candles.loc[times[bottom_3_index+1], "low"],
            candles.loc[times[bottom_3_index+2], "low"],
            candles.loc[times[bottom_3_index-2], "low"],
        ])
    
        bottoms_line_up = bottom_low_2 < bottom_low_1 and bottom_low_2 < bottom_low_3
        
        outside = candles.loc[turn_times[-6]]
        inside_1 = candles.loc[turn_times[-2]]
        inside_2 = candles.loc[turn_times[-4]]
        if outside["direction"] == "up":
            outside_high = outside["close"]
        else:
            outside_high = outside["open"]
        
        if inside_1["direction"] == "up":
            inside_1_high = inside_1["close"]
        else:
            inside_1_high = inside_1["open"]
        
        if inside_2["direction"] == "up":
            inside_2_high = inside_2["close"]
        else:
            inside_2_high = inside_2["open"]

        outside_good = outside_high > inside_2_high and outside_high - inside_1_high > 2*mean_candle and outside_high - inside_2_high > 2*mean_candle

        if bottoms_line_up and outside_good:
            head_and_shoulders = {
                "gradient": inside_1_high >= inside_2_high,
                "entry": inside_1_high,
                "exit" : inside_1_high + abs(min([inside_1_high, inside_2_high]) - bottom_low_2)
            }

    patterns["ihas"] = i_head_and_shoulders
    


    return(patterns)

def get_pp_daily(yester_dict):
    """
    Accecpts dictionary with keys "high" "low' and "close". Dictionary should be of prior trading period.
    Returns dictionary with key "daily" for value of dictionary for all pivot points.
    "standard"
    "woodie"
    "camarilla"
    "fibonacci"
    """

    my_pp = {}
    pp_ranges = ["daily"]
    for pp_range in pp_ranges:
        if pp_range == "daily":
            gran = "d"
        else:
            gran = "w"


        high = yester_dict["high"]
        low = yester_dict["low"]
        close = yester_dict["close"]


        #standand pivot points
        spp = (high+low+close)/3

        sr1 = (spp*2) - low
        ss1 = (spp*2) - high

        sr2 = spp + (high-low)
        ss2 = spp - (high-low)

        sr3 = high + (2*(spp-low))
        ss3 = low - (2*(high-spp))

        s_final_levels = {
            "pp": spp,
            "r1": sr1,
            "r2": sr2,
            "r3": sr3,
            "s1": ss1,
            "s2": ss2,
            "s3": ss3,
        }
        
        #woodie pivot point

        wpp = (high + low + (2*close))/4
        wr2 = wpp + high - low
        wr1 = (2*wpp) - low
        ws1 = (2*wpp) - high
        ws2 = wpp - high + low

        w_final_levels = {
            "pp": wpp,
            "r1": wr1,
            "r2": wr2,
            "s1": ws1,
            "s2": ws2,
        }

        #camarilla pivot point
        cpp = (high + low + close)/3
        cr4 = close + ((high-low)*1.5)
        cr3 = close + ((high-low)*1.25)
        cr2 = close + ((high-low)*1.1666)
        cr1 = close + ((high-low)*1.0833)
        cs1 = close - ((high-low)*1.0833)
        cs2 = close - ((high-low)*1.1666)
        cs3 = close - ((high-low)*1.25)
        cs4 = close - ((high-low)*1.5)

        c_final_levels = {
            "pp": cpp,
            "r1": cr1,
            "r2": cr2,
            "r3": cr3,
            "r4": cr4,
            "s1": cs1,
            "s2": cs2,
            "s3": cs3,
            "s4": cs4,
        }

        #fibonacci pivot point
        
        fpp = (high + low + close)/3
        fr3 = fpp + ((high-low)*1.00)
        fr2 = fpp + ((high-low)*0.618)
        fr1 = fpp + ((high-low)*0.382)
        fs1 = fpp - ((high-low)*0.382)
        fs2 = fpp - ((high-low)*0.618)
        fs3 = fpp - ((high-low)*1.00)

        f_final_levels = {
            "pp": fpp,
            "r1": fr1,
            "r2": fr2,
            "r3": fr3,
            "s1": fs1,
            "s2": fs2,
            "s3": fs3,
        }

        my_pp[pp_range] = {
            "standard": s_final_levels,
            "woodie": w_final_levels,
            "camarilla": c_final_levels,
            "fibonacci": f_final_levels,
        }

    return(my_pp)

def get_harmonic(candles, error_fac=1, get_turns=get_smoothed_turns):
    patterns = {}
    mean_candle = candles["bodysize"].sum()/len(candles)
    turns = get_turns(candles)
    turn_times = sorted(list(turns.keys()))
    times = sorted(list(candles.index))
    if len(turn_times) < 4:
        print("not enough")
        return
    

    #gettin max highs

    turn_1_high = max([
        candles.loc[turn_times[-1], "high"],
        candles.loc[times[times.index(turn_times[-1])+1], "high"],
        candles.loc[times[times.index(turn_times[-1])-1], "high"],
    ])
    
    turn_2_high = max([
        candles.loc[turn_times[-2], "high"],
        candles.loc[times[times.index(turn_times[-2])+1], "high"],
        candles.loc[times[times.index(turn_times[-2])-1], "high"],
    ])
    
    turn_3_high = max([
        candles.loc[turn_times[-3], "high"],
        candles.loc[times[times.index(turn_times[-3])+1], "high"],
        candles.loc[times[times.index(turn_times[-3])-1], "high"],
    ])
    
    turn_4_high = max([
        candles.loc[turn_times[-4], "high"],
        candles.loc[times[times.index(turn_times[-4])+1], "high"],
        candles.loc[times[times.index(turn_times[-4])-1], "high"],
    ])
    
    #getting max lows

    turn_1_low = min([
        candles.loc[turn_times[-1], "low"],
        candles.loc[times[times.index(turn_times[-1])+1], "low"],
        candles.loc[times[times.index(turn_times[-1])-1], "low"],
    ])
    
    turn_2_low = min([
        candles.loc[turn_times[-2], "low"],
        candles.loc[times[times.index(turn_times[-2])+1], "low"],
        candles.loc[times[times.index(turn_times[-2])-1], "low"],
    ])
    
    turn_3_low = min([
        candles.loc[turn_times[-3], "low"],
        candles.loc[times[times.index(turn_times[-3])+1], "low"],
        candles.loc[times[times.index(turn_times[-3])-1], "low"],
    ])
    
    turn_4_low = min([
        candles.loc[turn_times[-4], "low"],
        candles.loc[times[times.index(turn_times[-4])+1], "low"],
        candles.loc[times[times.index(turn_times[-4])-1], "low"],
    ])


    #getting turn values

    if turns[turn_times[-1]] == "up":
        turn_1 = min([
            candles.loc[turn_times[-1], "close"],
            candles.loc[turn_times[-1], "open"],
        ])
    else:
        turn_1 = max([
            candles.loc[turn_times[-1], "close"],
            candles.loc[turn_times[-1], "open"],
        ])

    if turns[turn_times[-2]] == "up":
        turn_2 = min([
            candles.loc[turn_times[-2], "close"],
            candles.loc[turn_times[-2], "open"],
        ])
    else:
        turn_2 = max([
            candles.loc[turn_times[-2], "close"],
            candles.loc[turn_times[-2], "open"],
        ])

    if turns[turn_times[-3]] == "up":
        turn_3 = min([
            candles.loc[turn_times[-3], "close"],
            candles.loc[turn_times[-3], "open"],
        ])
    else:
        turn_3 = max([
            candles.loc[turn_times[-3], "close"],
            candles.loc[turn_times[-3], "open"],
        ])

    if turns[turn_times[-4]] == "up":
        turn_4 = min([
            candles.loc[turn_times[-4], "close"],
            candles.loc[turn_times[-4], "open"],
        ])
    else:
        turn_4 = max([
            candles.loc[turn_times[-4], "close"],
            candles.loc[turn_times[-4], "open"],
        ])

   
    #Bearish ABCD
    bearish_abcd = None
    bear_abcd_turn_1_good = turns[turn_times[-1]] == "up"
    bear_abcd_turn_2_good = turns[turn_times[-2]] == "down"
    bear_abcd_turn_3_good = turns[turn_times[-3]] == "up"

    if bear_abcd_turn_1_good and bear_abcd_turn_2_good and bear_abcd_turn_3_good:
        bear_c_fib_level = turn_2_high - ((turn_2_high - turn_3_low)*0.618)
        bear_c_fib_good = abs(bear_c_fib_level - turn_1) < (mean_candle*error_fac)

        bear_d_fib_level = turn_1_low + ((turn_2_high - turn_1_low)*1.272)
        if bear_c_fib_good:
            bearish_abcd = {
                "turning_point": bear_d_fib_level,
                "length": abs(times.index(turn_times[-2])- times.index(turn_times[-3])),
                "last_turn":turn_times[-1]
            }
    patterns["bearish_abcd"] = bearish_abcd
            
    
    #Bullish ABCD
    bullish_abcd = None
    bull_abcd_turn_1_good = turns[turn_times[-1]] == "down"
    bull_abcd_turn_2_good = turns[turn_times[-2]] == "up"
    bull_abcd_turn_3_good = turns[turn_times[-3]] == "down"

    if bull_abcd_turn_1_good and bull_abcd_turn_2_good and bull_abcd_turn_3_good:
        bull_c_fib_level = turn_2_low + ((turn_3_high - turn_2_low)*0.618)
        bull_c_fib_good = abs(bull_c_fib_level - turn_1) < (mean_candle*error_fac)

        bull_d_fib_level = turn_1_high - ((turn_1_high - turn_2_low)*1.272)
        if bull_c_fib_good:
            bullish_abcd = {
                "turning_point" : bull_d_fib_level,
                "length": abs(times.index(turn_times[-2])- times.index(turn_times[-3])),
                "last_turn":turn_times[-1]
                }
    patterns["bullish_abcd"] = bullish_abcd

    #Bearish Gartley
    bearish_gartley = None
    bear_gart_1_good = turns[turn_times[-1]] == "up"
    bear_gart_2_good = turns[turn_times[-2]] == "down"
    bear_gart_3_good = turns[turn_times[-3]] == "up"
    bear_gart_4_good = turns[turn_times[-4]] == "down"

    if bear_gart_1_good and bear_gart_2_good and bear_gart_3_good and bear_gart_4_good:
        bear_b_fib_level = turn_3_low + ((turn_4_high-turn_3_low)*0.618)
        bear_b_fib_good = abs(turn_2 - bear_b_fib_level) < (mean_candle*error_fac)
        if bear_b_fib_good:
            bear_c_fib_level_1 = turn_2_high - ((turn_2_high - turn_3_low)*0.382)
            bear_c_fib_level_2 = turn_2_high - ((turn_2_high - turn_3_low)*0.886)
            if abs(turn_1 - bear_c_fib_level_1)<mean_candle*error_fac or abs(turn_1 - bear_c_fib_level_2)<mean_candle*error_fac:
                bearish_gartley = {
                    "turning_point": turn_3_low + ((turn_4_high - turn_3_low)*0.786)
                }
    patterns["bearish_gartley"] = bearish_gartley

    #Bullish Gartley
    bullish_gartley = None
    bull_gart_1_good = turns[turn_times[-1]] == "down"
    bull_gart_2_good = turns[turn_times[-2]] == "up"
    bull_gart_3_good = turns[turn_times[-3]] == "down"
    bull_gart_4_good = turns[turn_times[-4]] == "up"

    if bull_gart_1_good and bull_gart_2_good and bull_gart_3_good and bull_gart_4_good:
        bull_b_fib_level = turn_3_high - ((turn_3_high-turn_4_low)*0.618)
        bull_b_fib_good = abs(turn_2 - bull_b_fib_level) < (mean_candle*error_fac)
        if bull_b_fib_good:
            bull_c_fib_level_1 = turn_2_low + ((turn_3_high - turn_2_low)*0.382)
            bull_c_fib_level_2 = turn_2_low + ((turn_3_high - turn_2_low)*0.886)
            if abs(turn_1 - bull_c_fib_level_1)<mean_candle*error_fac or abs(turn_1 - bull_c_fib_level_2)<mean_candle*error_fac:
                bullish_gartley = {
                    "turning_point": turn_3_high - ((turn_3_high - turn_4_low)*0.786)
                }
    patterns["bullish_gartley"] = bullish_gartley


    #Bearish Crab
    bearish_crab = None
    bear_crab_1_good = turns[turn_times[-1]] == "up"
    bear_crab_2_good = turns[turn_times[-2]] == "down"
    bear_crab_3_good = turns[turn_times[-3]] == "up"
    bear_crab_4_good = turns[turn_times[-4]] == "down"

    if bear_crab_1_good and bear_crab_2_good and bear_crab_3_good and bear_crab_4_good:
        bear_b_fib_level_1 = turn_3_low + ((turn_4_high-turn_3_low)*0.618)
        bear_b_fib_level_2 = turn_3_low + ((turn_4_high-turn_3_low)*0.382)
        bear_b_fib_good = abs(turn_2 - bear_b_fib_level_1) < (mean_candle*error_fac) or abs(turn_2 - bear_b_fib_level_2) < (mean_candle*error_fac)
        if bear_b_fib_good:
            bear_c_fib_level_1 = turn_2_high - ((turn_2_high - turn_3_low)*0.382)
            bear_c_fib_level_2 = turn_2_high - ((turn_2_high - turn_3_low)*0.886)
            if abs(turn_1 - bear_c_fib_level_1)<mean_candle*error_fac or abs(turn_1 - bear_c_fib_level_2)<mean_candle*error_fac:
                bearish_crab = {
                    "turning_point": turn_3_low + ((turn_4_high - turn_3_low)*1.618)
                }
    patterns["bearish_crab"] = bearish_crab


    #Bullish Crab
    bullish_crab = None
    bull_crab_1_good = turns[turn_times[-1]] == "down"
    bull_crab_2_good = turns[turn_times[-2]] == "up"
    bull_crab_3_good = turns[turn_times[-3]] == "down"
    bull_crab_4_good = turns[turn_times[-4]] == "up"

    if bull_crab_1_good and bull_crab_2_good and bull_crab_3_good and bull_crab_4_good:
        bull_b_fib_level_1 = turn_3_high - ((turn_3_high-turn_4_low)*0.618)
        bull_b_fib_level_2 = turn_3_high - ((turn_3_high-turn_4_low)*0.382)
        bull_b_fib_good = abs(turn_2 - bull_b_fib_level_1) < (mean_candle*error_fac) or abs(turn_2-bull_b_fib_level_2) < (mean_candle*error_fac)
        if bull_b_fib_good:
            bull_c_fib_level_1 = turn_2_low + ((turn_3_high - turn_2_low)*0.382)
            bull_c_fib_level_2 = turn_2_low + ((turn_3_high - turn_2_low)*0.886)
            if abs(turn_1 - bull_c_fib_level_1)<mean_candle*error_fac or abs(turn_1 - bull_c_fib_level_2)<mean_candle*error_fac:
                bullish_crab = {
                    "turning_point": turn_3_high - ((turn_3_high - turn_4_low)*1.618)
                }
    patterns["bullish_crab"] = bullish_crab

    #Bearish Bat
    bearish_bat = None
    bear_bat_1_good = turns[turn_times[-1]] == "up"
    bear_bat_2_good = turns[turn_times[-2]] == "down"
    bear_bat_3_good = turns[turn_times[-3]] == "up"
    bear_bat_4_good = turns[turn_times[-4]] == "down"

    if bear_bat_1_good and bear_bat_2_good and bear_bat_3_good and bear_bat_4_good:
        bear_b_fib_level_1 = turn_3_low + ((turn_4_high-turn_3_low)*0.5)
        bear_b_fib_level_2 = turn_3_low + ((turn_4_high-turn_3_low)*0.382)
        bear_b_fib_good = abs(turn_2 - bear_b_fib_level_1) < (mean_candle*error_fac) or abs(turn_2 - bear_b_fib_level_2) < (mean_candle*error_fac)
        if bear_b_fib_good:
            bear_c_fib_level_1 = turn_2_high - ((turn_2_high - turn_3_low)*0.382)
            bear_c_fib_level_2 = turn_2_high - ((turn_2_high - turn_3_low)*0.886)
            if abs(turn_1 - bear_c_fib_level_1)<mean_candle*error_fac or abs(turn_1 - bear_c_fib_level_2)<mean_candle*error_fac:
                bearish_bat = {
                    "turning_point": turn_3_low + ((turn_4_high - turn_3_low)*0.886)
                }
    patterns["bearish_bat"] = bearish_bat


    #Bullish Bat
    bullish_bat = None
    bull_bat_1_good = turns[turn_times[-1]] == "down"
    bull_bat_2_good = turns[turn_times[-2]] == "up"
    bull_bat_3_good = turns[turn_times[-3]] == "down"
    bull_bat_4_good = turns[turn_times[-4]] == "up"

    if bull_bat_1_good and bull_bat_2_good and bull_bat_3_good and bull_bat_4_good:
        bull_b_fib_level_1 = turn_3_high - ((turn_3_high-turn_4_low)*0.5)
        bull_b_fib_level_2 = turn_3_high - ((turn_3_high-turn_4_low)*0.382)
        bull_b_fib_good = abs(turn_2 - bull_b_fib_level_1) < (mean_candle*error_fac) or abs(turn_2-bull_b_fib_level_2) < (mean_candle*error_fac)
        if bull_b_fib_good:
            bull_c_fib_level_1 = turn_2_low + ((turn_3_high - turn_2_low)*0.382)
            bull_c_fib_level_2 = turn_2_low + ((turn_3_high - turn_2_low)*0.886)
            if abs(turn_1 - bull_c_fib_level_1)<mean_candle*error_fac or abs(turn_1 - bull_c_fib_level_2)<mean_candle*error_fac:
                bullish_bat = {
                    "turning_point": turn_3_high - ((turn_3_high - turn_4_low)*0.886)
                }
    patterns["bullish_bat"] = bullish_bat

    #Bearish Butterfly
    bearish_butt = None
    bear_butt_1_good = turns[turn_times[-1]] == "up"
    bear_butt_2_good = turns[turn_times[-2]] == "down"
    bear_butt_3_good = turns[turn_times[-3]] == "up"
    bear_butt_4_good = turns[turn_times[-4]] == "down"

    if bear_butt_1_good and bear_butt_2_good and bear_butt_3_good and bear_butt_4_good:
        bear_b_fib_level_1 = turn_3_low + ((turn_4_high-turn_3_low)*0.786)
        bear_b_fib_good = abs(turn_2 - bear_b_fib_level_1) < (mean_candle*error_fac)
        if bear_b_fib_good:
            bear_c_fib_level_1 = turn_2_high - ((turn_2_high - turn_3_low)*0.382)
            bear_c_fib_level_2 = turn_2_high - ((turn_2_high - turn_3_low)*0.886)
            if abs(turn_1 - bear_c_fib_level_1)<mean_candle*error_fac:
                bearish_butt = {
                    "turning_point": turn_3_low + ((turn_4_high - turn_3_low)*1.27)
                }
            if abs(turn_1 - bear_c_fib_level_2)<mean_candle*error_fac:
                bearish_butt = {
                    "turning_point": turn_3_low + ((turn_4_high - turn_3_low)*1.618)
                }
    patterns["bearish_butterfly"] = bearish_butt


    #Bullish Butterfly
    bullish_butt = None
    bull_butt_1_good = turns[turn_times[-1]] == "down"
    bull_butt_2_good = turns[turn_times[-2]] == "up"
    bull_butt_3_good = turns[turn_times[-3]] == "down"
    bull_butt_4_good = turns[turn_times[-4]] == "up"

    if bull_butt_1_good and bull_butt_2_good and bull_butt_3_good and bull_butt_4_good:
        bull_b_fib_level_1 = turn_3_high - ((turn_3_high-turn_4_low)*0.786)
        bull_b_fib_good = abs(turn_2 - bull_b_fib_level_1) < (mean_candle*error_fac)
        if bull_b_fib_good:
            bull_c_fib_level_1 = turn_2_low + ((turn_3_high - turn_2_low)*0.382)
            bull_c_fib_level_2 = turn_2_low + ((turn_3_high - turn_2_low)*0.886)
            if abs(turn_1 - bull_c_fib_level_1)<mean_candle*error_fac:
                bullish_butt = {
                    "turning_point": turn_3_high - ((turn_3_high - turn_4_low)*1.27)
                }
            if abs(turn_1 - bull_c_fib_level_2)<mean_candle*error_fac:
                bullish_butt = {
                    "turning_point": turn_3_high - ((turn_3_high - turn_4_low)*1.618)
                }
    patterns["bullish_butterfly"] = bullish_butt
    return(patterns)


if __name__ == "__main__":
#    x = get_candles(instrument="GBP_USD", count=200)
#    pp(get_double_candlestick_patterns(x))

    pass

    

