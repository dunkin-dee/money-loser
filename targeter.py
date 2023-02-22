import pandas as pd
import numpy as np

def reproduce_columns(in_df, col_count):
    df = in_df.copy(deep=True)
    shifted_df= {}
    for x in range(1, col_count+1):
        temp_df = df[df.columns].shift(x)
        for col_name in temp_df.columns:
            temp_df.rename({col_name: col_name+f"-{str(x)}"}, axis=1, inplace=True)
        shifted_df[x] = temp_df
    for x in shifted_df.keys():
        df = pd.concat([df, shifted_df[x]], axis=1)
    return(df.dropna(axis=0))


def getfirst_pandas(condition, df):
    cond = df[condition(df)]
    if not cond.empty:
        return(cond.iloc[0].name)
    else:
        return None

def alt_target(in_df, run_range=5):
    df = in_df.copy(deep=True)
    shifted_df= {}
    for x in range(1, run_range+1):
        temp_df = df[df.columns].shift(-x)
        for col_name in temp_df.columns:
            temp_df.rename({col_name: col_name+f"-{str(x)}"}, axis=1, inplace=True)
        shifted_df[x] = temp_df
    for x in shifted_df.keys():
        df = pd.concat([df, shifted_df[x]], axis=1)
    df.dropna(axis=0, inplace=True)
    mean_candle = df["bodysize"].abs().mean()
    rel_columns = []
    for x in range(1, run_range+1):
        rel_columns.append(f"bodysize-{str(x)}")
    df["target"] = np.where(df[rel_columns].mean(axis=1)>=mean_candle/2, 1, 0)
    temp_df = df.copy(deep=True)
    temp_df["target"] = df["target"]
    return(temp_df.dropna(axis=0))


def get_target_up_down(in_df, uptick=0.0050, downtick=0.0020, up_length=16):
    """
    Takes dataframe and returns df with added target_up, where 1 is classified as candle after which there is a raise in price `'uptick'`
    in the next `'up_length'` number  of candles before a drop of `'downtick'` and similar for target_down
    """
    df = in_df.copy(deep=True)
    df_keys = sorted(list(df.index))
    df["target_up"] = np.zeros(len(df))
    df["target_down"] = np.zeros(len(df))
    for index, row in df.iterrows():
        base_point = row["close"]
        goal_point_up = base_point + uptick
        fail_point_up = base_point - downtick
        goal_point_down = base_point - uptick
        fail_point_down = base_point + downtick
        small_df = df[df_keys.index(index)+1:df_keys.index(index)+up_length]
        small_keys = sorted(list(small_df.index))
      
        goal_reached_up = getfirst_pandas(lambda x: x.high >= goal_point_up, small_df)
        if goal_reached_up:
            goal_index = small_keys.index(goal_reached_up)
            if small_df[:goal_index]["low"].min() > fail_point_up:
                df.at[index, 'target_up'] = 1
            
        goal_reached_down = getfirst_pandas(lambda x: x.low <= goal_point_down, small_df)
        if goal_reached_down:
            goal_index = small_keys.index(goal_reached_down)
            if small_df[:goal_index]["high"].max() < fail_point_down:
                df.at[index, "target_down"] = 1

    return(df[:-up_length])


def get_target_regression(in_df):
    df = in_df.copy(deep=True)
    df["target_regression"] = df.shift(-1)["close"]
    df.dropna(axis=0, inplace=True)
    return(df)