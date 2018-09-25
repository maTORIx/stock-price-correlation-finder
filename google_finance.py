import requests
from datetime import datetime, timedelta

def search(
           code,
           lsat_date=datetime.now(),#データの取得開始日
           interval=86400,#データの間隔(秒)。1日 = 86400秒
           period="1Y",#データを取得する期間
           market="TKO",#取引所のコード　TYO=東京証券取引所
           url='https://www.google.com/finance/getprices'
          ):
    
    params = {
        'q': code,
        'i': interval,
        'x': market,
        'p': period,
        'ts': lsat_date.timestamp()
    }
    
    r = requests.get(url, params=params)
    
    lines = r.text.splitlines()
    columns = lines[4].split("=")[1].split(",")
    prices = lines[8:]
    
    #レスポンスの１日目のタイムスタンプをdatetimeに
    first_cols = prices[0].split(",")
    first_date = datetime.fromtimestamp(int(first_cols[0].lstrip('a')))
    result = [[first_date.date(), first_cols[1], first_cols[2], first_cols[3], first_cols[4], first_cols[5]]]
    
    for price in prices[1:]:
      cols = price.split(",")
      date = first_date + timedelta(int(cols[0]))
      result.append([date.date(), cols[1], cols[2], cols[3], cols[4], cols[5]])
    
    df = pd.DataFrame(result, columns = columns)
    return df.to_csv()


print(search(7203))