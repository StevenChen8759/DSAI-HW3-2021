import pandas as pd

from utils import csvIO

if __name__ == "__main__":
    # res_data = csvIO.read("download/result/bidresult-73.csv")
    # res_data_buy = res_data.loc[res_data["action"]=="buy"]
    # res_data_sell = res_data.loc[res_data["action"]=="sell"]
    # res_data_buy.to_csv("res_data_buy.csv")
    # res_data_sell.to_csv("res_data_sell.csv")

    ip_train = csvIO.read("./training_data/target0.csv")
    ip_con = ip_train[["time", "consumption"]].iloc[:168]
    ip_gen = ip_train[["time", "generation"]].iloc[:168]
    ip_con.to_csv("./sample_data/consumption.csv")
    ip_gen.to_csv("./sample_data/generation.csv")