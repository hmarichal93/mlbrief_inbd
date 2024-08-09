import numpy as np
import pandas as pd

from pathlib import Path

class TrainingAnalysis:
    def __init__(self, stdout_inbd_path, output_dir):
        self.stdout_inbd_path = stdout_inbd_path
        self.output_dir = output_dir + f"/{Path(stdout_inbd_path).stem}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.data = self.read_stdout_inbd(self.stdout_inbd_path)

    @staticmethod
    def read_file(stdout_inbd_path):
        with open(stdout_inbd_path, 'r') as file:
            data = file.readlines()
        return data


    def parse_data(self, data):
        cse, acc, w_loss, w_acc, lr = [], [], [], [], []
        results = {}
        for line in data:
            line = line.replace("\n","")
            line = line.split("|")
            epocs, percentage = line[0], line[1]
            percentage = float(percentage.split("]")[0] )
            epocs = float(epocs.split("[")[1])
            results[str(epocs)] = {}

            for l in line:
                if "cse" in l:
                    cse = float(l.split(":")[1])

                elif "acc" in l and "w_acc" not in l:
                    acc = float(l.split(":")[1])

                elif "w_loss" in l:
                    w_loss = float(l.split(":")[1])

                elif "w_acc" in l:
                    w_acc = float(l.split(":")[1])

                elif "lr" in l:
                    lr = float(l.split(":")[1])

            ###
            results[str(epocs)]["cse"] = cse
            results[str(epocs)]["acc"] = acc
            results[str(epocs)]["w_loss"] = w_loss
            results[str(epocs)]["w_acc"] = w_acc
            results[str(epocs)]["lr"] = lr

        cse = [results[str(epocs)]["cse"] for epocs in results.keys()]
        acc = [results[str(epocs)]["acc"] for epocs in results.keys()]
        w_loss = [results[str(epocs)]["w_loss"] for epocs in results.keys()]
        w_acc = [results[str(epocs)]["w_acc"] for epocs in results.keys()]
        lr = [results[str(epocs)]["lr"] for epocs in results.keys()]


        data = pd.DataFrame({"cse": cse, "acc": acc, "w_loss": w_loss, "w_acc": w_acc, "lr": lr})

        return data

    def read_stdout_inbd(self, stdout_inbd_path):
        data = self.read_file(stdout_inbd_path)
        data = self.parse_data(data)

        return data


    def plot_loss(self, column="w_loss"):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(self.data[column])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(column)
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/{column}.png")
        return

def main(stdout_inbd_path, output_dir):
    analysis = TrainingAnalysis(stdout_inbd_path, output_dir)
    analysis.plot_loss(column="w_loss")
    analysis.plot_loss(column="acc")
    analysis.plot_loss(column="cse")
    analysis.plot_loss(column="w_acc")
    return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout_inbd_path", type=str, required=True,
                        default="./assets/eh_1500_inbd_loss.txt")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()
    main(args.stdout_inbd_path, args.output_dir)

