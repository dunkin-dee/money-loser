import json
import requests
import datetime as dt
import pandas as pd

from dateutil.parser import parse
from pprint import pprint

from auth_token import token


def get_candles(instrument="GBP_USD", gran="H1", count="50", token=token):
    base_url = "https://api-fxpractice.oanda.com/v3/instruments/%s/candles?count=%s&price=M&granularity=%s"
    url = base_url%(instrument, count, gran)

    headers = {'content-type': 'application/json',
    'Authorization': 'Bearer %s'%token}

    r = requests.get(url, headers=headers)

    if r.status_code == 200:
        column_names = ['open', 'high', 'low', 'close', 'bodysize', 'volume', 'direction']
        df = pd.DataFrame(columns=column_names)

        for candle in json.loads(r.text)['candles']:
            if candle["complete"]:
            
                if float(candle['mid']['o']) > float(candle['mid']['c']):
                    direction = 'down'
                else:
                    if float(candle['mid']['o']) < float(candle['mid']['c']):
                        direction = 'up'
                    else:
                        direction = 'none'
            
                df.loc[parse(candle['time'], ignoretz=True)] = [float(candle['mid']['o']),
                                                float(candle['mid']['h']),
                                                float(candle['mid']['l']),
                                                float(candle['mid']['c']),
                                                abs(float(candle['mid']['o']) - float(candle['mid']['c'])),
                                                int(candle['volume']),
                                                direction]
        return(df)

    else:
        return(r.status_code)


if __name__ == "__main__":
    final = get_candles()
    print(final)