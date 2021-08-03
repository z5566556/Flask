from flask import Flask #載入 Flask
from flask import request #載入 Request 物件
from flask import render_template #載入 render_template 函示
import numpy as np
import pandas as pd

# get data
import pandas_datareader.data as pdr
import yfinance as yf
# visual
import matplotlib.pyplot as plt

#time
import datetime as datetime

#Prophet
from fbprophet import Prophet

# 繪製K線圖
import talib
import mpl_finance as mpf

from sklearn import metrics


# 建立 Application 物件,可以設定靜態檔案的路徑處理
app=Flask(
    __name__,
    static_folder="static", # 靜態檔案的資料夾名稱
    static_url_path="/" # 靜態檔案對應的網址路徑
)
# http://127.0.0.1:3000/smartphone.png 圖片網址

# 使用 GET方法 建立路徑 / 對應的處理函式
@app.route("/")
def index_1():
    return render_template("index_1.html")

# 處理路徑/adj_close 的對應函式
@app.route("/adj_close", methods=["POST"])
def adj_close():
    stock1 = request.form["z"]
    stock1 = str(stock1)
    yf.pdr_override()
    start = datetime.datetime(2015,1,1)
    df_stock = pdr.get_data_yahoo(stock1, start)
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12,8))
    plt.plot(df_stock['Adj Close'])
    plt.title(stock1)
    fig.savefig(r"C:\Users\han_lai\Desktop\ab\static\{}.png".format(stock1))
    data2 = "/{}.png".format(stock1)
    return render_template("adj.html",data1=df_stock['Adj Close'], data2=str(data2))

# 處理路徑/stock_project 的對應函式
@app.route("/stock_project", methods=["POST"])
def stock_project():
    stock2 = request.form["w"]
    stock2 = str(stock2)
    yf.pdr_override()
    start = datetime.datetime(2015,1,1)
    df_stock = pdr.get_data_yahoo(stock2, start)
    new_df_stock = pd.DataFrame(df_stock['Adj Close']).reset_index().rename(columns={'Date':'ds','Adj Close':'y'})
    # 使用Prophet來預測股票
    new_df_stock['y'] = np.log(new_df_stock['y'])
    # 定義模型
    model = Prophet()
    # 訓練模型
    model.fit(new_df_stock)
    # 建立預測集
    future = model.make_future_dataframe(periods=365)
    # 進行預測
    forecast = model.predict(future)
    plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
    model.plot(forecast,figsize=(10,6))
    # plt.title(stock2+"未來1年預測走勢", verticalalignment='center')
    plt.savefig(r"C:\Users\han_lai\Desktop\ab\static\project_{}.png".format(stock2))
    data5 = "/project_{}.png".format(stock2)
    forecast['yhat1'] = np.exp(forecast.yhat)
    return render_template("project.html",data3=forecast['ds'], data4=forecast['yhat1'], data5=str(data5), data6=stock2)

@app.route("/k_line", methods=["POST"])
def k_line():
    stock3 = request.form["k"]
    stock3 = str(stock3)
    yf.pdr_override()
    start = datetime.datetime(2021,1,1)
    df_stock = pdr.get_data_yahoo(stock3, start)
    df_stock.index = df_stock.index.format(formatter=lambda x : x.strftime('%Y-%m-%d'))
    # 加上10日均線與30日均線
    sma_10 = talib.SMA(np.array(df_stock['Close']),10)
    sma_30 = talib.SMA(np.array(df_stock['Close']),30)
    # KD指標
    df_stock['k'],df_stock['d'] = talib.STOCH(df_stock['High'], df_stock['Low'], df_stock['Close'])
    df_stock['k'].fillna(value=0, inplace= True)
    df_stock['d'].fillna(value=0, inplace= True)
    #繪圖
    fig = plt.figure(figsize=(24, 20))
    ax = fig.add_axes([0.1,0.4,0.8,0.4])
    ax2 = fig.add_axes([0.1,0.3,0.8,0.1])
    ax3 = fig.add_axes([0.1,0.1,0.8,0.2])
    ax.set_xticks(range(0,len(df_stock.index), 10))
    ax.set_xticklabels(df_stock.index[::10])
    mpf.candlestick2_ochl(ax, df_stock['Open'], df_stock['Close'], df_stock['High'],
                        df_stock['Low'], width=0.6, colorup='r',colordown='g',
                        alpha=0.75)
    plt.rcParams['font.sans-serif']=['Microsoft JhengHei'] 
    ax.plot(sma_10, label = '10日均線')
    ax.plot(sma_30, label = '30日均線')

    ax2.plot(df_stock['k'],label='K值')
    ax2.plot(df_stock['d'],label='D值')
    ax2.set_xticks(range(0,len(df_stock.index), 10))
    ax2.set_xticklabels(df_stock.index[::10]) 

    mpf.volume_overlay(ax3, df_stock['Open'], df_stock['Close'], df_stock['Volume'],
                    width=0.5, colorup='r', colordown='g', alpha=0.8)
    ax3.set_xticks(range(0,len(df_stock.index), 10))
    ax3.set_xticklabels(df_stock.index[::10])
    ax.legend()
    ax2.legend()
    ax.set_title(stock3+"的K線圖&KD值圖&交易量圖", fontsize=32)
    fig.savefig(r"C:\Users\han_lai\Desktop\ab\static\kline_{}.png".format(stock3))
    data7 = "/kline_{}.png".format(stock3)
    return render_template("k_line.html", data7 = str(data7))

# 處理路徑/money 的對應函式
@app.route("/money", methods=["POST"])
def money():
    stock4 = request.form["c"]
    money = request.form["m"]
    stock4 = str(stock4)
    money = int(money)
    yf.pdr_override()
    start = datetime.datetime(2021,1,1)
    df_stock = pdr.get_data_yahoo(stock4, start)
    # 收盤價和第一天的收盤價的比較
    df_stock['normalized_price'] = df_stock['Adj Close']/df_stock['Adj Close'].iloc[0]
    # 資產分配
    # 我們將我們的投資金額用以下的比例分配
    # 之後將分配到的比例乘上每一日的收益率
    # stock 100%
    weight=1
    df_stock['weighted daily return']=df_stock['normalized_price']*weight
    # 將股票的收益率乘上你要投資的總金額，現在假設是XX萬
    df_stock['Total Pos']= df_stock['weighted daily return']*money
    # 將剛剛總收益繪製圖表
    plt.rcParams['font.sans-serif']=['Microsoft JhengHei']
    fig = plt.figure(figsize=(10,6))
    plt.plot(df_stock['Total Pos'], '-', label = '總收益')
    plt.title(stock4+'的收益曲線', loc='right')
    plt.xlabel('日期')
    plt.ylabel('金額')
    plt.grid(True, axis = 'y')
    plt.legend()
    fig.savefig(r"C:\Users\han_lai\Desktop\ab\static\money_{}.png".format(stock4))
    data9 = "/money_{}.png".format(stock4)
    return render_template("money.html",data8=df_stock['Total Pos'], data9=str(data9))

# 處理路徑 /page 的對應函式
@app.route("/page")
def page():
    return render_template("page.html")

# 啟動網站伺服器, 可透過 port 參數指定阜號
app.run(port=3000)
