from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://rex:#Pass123@localhost/new_ml")

all_pairs = [
# "AUDCAD",
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

time_periods = ["1h", 
                "4h", 
                "d"]

my_string = []

for x in range(1,6):
  my_string.append(f"DROP `shadow{str(x)}`")
  my_string.append(f"DROP `wick{str(x)}`")
  my_string.append(f"DROP `bodysize{str(x)}`")


for pair in all_pairs:
  for tp in time_periods:
    print(f"Deleting from {pair.lower()}_{tp}")
    sql = f"ALTER TABLE {pair.lower()}_{tp} {', '.join(my_string)}"
    with engine.connect() as con:
      con.execute(sql)
    print("Done!") 
    