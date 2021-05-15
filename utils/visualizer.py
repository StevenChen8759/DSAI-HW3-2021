# Python Interpreter Module ImportError
import os
import argparse
from datetime import datetime
from multiprocessing import Process
from threading import Thread

# Python External Module ImportError
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger

# User Module Import
from utils import csvIO

# User defined global variable / lambda function in this Module
datetime_parse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

@logger.catch
def plot_con_gen(input_df_consumption, input_df_generation, directory='output'):

    date_lb = datetime_parse(input_df_consumption["time"].iloc[0]).date()
    date_ub = datetime_parse(input_df_consumption["time"].iloc[len(input_df_consumption.index) - 1]).date()

    for i in range(0, (date_ub - date_lb).days + 1):

        gendate = datetime_parse(input_df_consumption["time"].iloc[i * 24]).date()
        logger.debug(f"Plotting date {gendate}...")

        fig = plt.figure() # Create matplotlib figure

        ax = fig.add_subplot(111) # Create matplotlib axes
        ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

        width = 0.4

        plt_con_df = input_df_consumption["consumption"].iloc[i * 24 : i * 24 + 24].reset_index(drop=True)
        plt_gen_df = input_df_generation["generation"].iloc[i * 24 : i * 24 + 24].reset_index(drop=True)

        plt_con_df.plot(kind='bar', color='red', ax=ax, width=width, position=1, label='consumption')
        plt_gen_df.plot(kind='bar', color='green', ax=ax2, width=width, position=0, label='generation')

        ax.set_xlabel('Hours')
        ax.set_ylabel('Consumption (kWh)')
        ax2.set_ylabel('Generation (kWh)')

        lim1 = ax.get_ylim()
        lim2 = ax2.get_ylim()

        new_lim = (min(lim1[0], lim2[0]), max(lim1[1], lim2[1]))

        ax.set_xlim((-1,24))
        ax.set_ylim(new_lim)
        ax2.set_ylim(new_lim)

        plt.title(f"Consumption versus Generation for {gendate}")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")

        plt.savefig(f"./{directory}/con_gen_plot_{gendate}.jpg")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--training_data", default="./training_data/target49.csv", help="input target csv to plot")
    parser.add_argument("--op_dir", default="output", help="input target csv to plot")
    args = parser.parse_args()
    processes = []

    for i in range(50):
        filename = f"./training_data/target{i}.csv"
        logger.info(f"Plotting target: {filename}, thread No.: {i}")

        train_df = csvIO.read(filename)
        for j in range(1, len(train_df.index)):
            t1 = datetime_parse(train_df["time"][j - 1])
            t2 = datetime_parse(train_df["time"][j])
            if (t2 - t1).total_seconds() != 3600:
                logger.error(f"Pre-plot data verify failed, time gap [{j-1},{j}] is not 1 hour({t1},{t2})")
                assert(False)

        con = train_df[["time", "consumption"]]
        gen = train_df[["time", "generation"]]

        if not os.path.isdir(f"./output/train_plot_{i}"):
            os.mkdir(f"./output/train_plot_{i}")

        t = Process(target=plot_con_gen, args=(con, gen, f"./output/train_plot_{i}",))
        if len(processes) == 8:
            idx = i % 8
            processes[idx].join()
            processes[idx] = t
        else:
            processes.append(t)
        t.start()

    for i in range(8):
        processes[i].join()
