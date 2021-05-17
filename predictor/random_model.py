# Python Interpreter Module ImportError
import random
from datetime import datetime, timedelta

# Python External Module ImportError
from loguru import logger

# User Module Import

# User defined global variable in this Module
# Declare lambda function for calculating time
datetime_parse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
fetch_predict_date = lambda x : datetime_parse(x["time"][0]) + timedelta(days = 7)

def generate_bidding_one_day(input_df_con, input_df_gen):

    target_date = fetch_predict_date(input_df_con)

    dt_lb   = target_date                         # 2018-09-01 00:00:00
    dt_ub   = target_date + timedelta(hours = 23) # 2018-09-01 23:00:00
    buy_lb  = target_date + timedelta(hours = 6)  # 2018-09-01 06:00:00
    buy_ub  = target_date + timedelta(hours = 18) # 2018-09-01 18:00:00
    sell_lb = target_date + timedelta(hours = 9)  # 2018-09-01 09:00:00
    sell_ub = target_date + timedelta(hours = 17) # 2018-09-01 17:00:00
    bidding_list = []

    dt_iter = dt_lb
    while True:
        logger.debug(f"Iteration on {dt_iter:%Y-%m-%d %H:%M:%S}")

        # Phase 1 - Determine Buy Action by time -> Buy or None (50%/50%)
        if buy_lb <= dt_iter <= buy_ub:
            logger.debug(f"Generate [buy] bidding")
            buy_action = True
        else:
            buy_action = False
        # temp = random.randint(1, 10)
        # if temp > 5:
        #     buy_action = True
        # else:
        #     buy_action = False

        # Phase 2 - Determine Sell Action by time -> Sell or None (50%/50%)
        if sell_lb <= dt_iter <= sell_ub:
            logger.debug(f"Generate [sell] bidding - normal mode")
            sell_action = True
        else:
            logger.debug(f"Generate [sell] bidding - intentional mode")
            sell_action = False
        # temp = random.randint(1, 10)
        # if temp > 5:
        #     sell_action = True
        # else:
        #     sell_action = False

        # Phase 3 - Generate bidding info - buy_action
        if buy_action:
            buy_price = random.uniform(2.33, 2.48) # Buy price in NTD per kWh
            buy_volume = random.uniform(3.0, 7.0)  # Buy volume in kWh
            templist = []
            templist.append(dt_iter.strftime("%Y-%m-%d %H:%M:%S"))
            templist.append("buy")
            templist.append(round(buy_price, 2))
            templist.append(round(buy_volume, 2))
            bidding_list.append(templist)

        # Phase 4 - Generate bidding info - sell_action
        if sell_action:
            sell_price = random.uniform(2.30, 2.45) # Sell price in NTD per kWh
            sell_volume = random.uniform(2.0, 4.0)  # Sell volume in kWh
        else:
            # Intentionally sell power with price higher than city power 2.53
            sell_price = random.uniform(2.53, 2.75) # Sell price in NTD per kWh
            sell_volume = random.uniform(1.0, 5.0)  # Sell volume in kWh
        templist = []
        templist.append(dt_iter.strftime("%Y-%m-%d %H:%M:%S"))
        templist.append("sell")
        templist.append(round(sell_price, 2))
        templist.append(round(sell_volume, 2))
        bidding_list.append(templist)

        # Phase 5 - Check and  add dt_iter
        if dt_iter == dt_ub:
            logger.debug("End of generate_bidding_one_day")
            break
        else:
            dt_iter = dt_iter + timedelta(hours = 1)

    return bidding_list