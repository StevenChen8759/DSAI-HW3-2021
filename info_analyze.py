import pandas as pd

from utils import csvIO

if __name__ == "__main__":
    res_data = csvIO.read("download/result/bidresult-72.csv")
    res_data_buy = res_data.loc[res_data["action"]=="buy"]
    res_data_sell = res_data.loc[res_data["action"]=="sell"]
    res_data_buy.to_csv("res_data_buy.csv")
    res_data_sell.to_csv("res_data_sell.csv")