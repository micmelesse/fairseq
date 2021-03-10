import os
import re
import argparse
import pandas as pd


def read_roberta_log(path):
    log_file = open(path)

    log_file.seek(0)

    pattern = "loss=(.*), nll_loss=(.*), ppl=(.*), wps=(.*), ups=(.*), wpb=(.*), bsz=(.*), num_updates=(.*), lr=(.*), gnorm=(.*), clip=(.*), oom=(.*), wall=(.*), train_wall=(.*)"
    columns = ["loss", "nll_loss", "ppl", "wps", "ups", "wpb", "bsz",
               "num_updates", "lr", "gnorm", "clip", "oom", "wall", "train_wall"]
    metrics = []
    for line in log_file:
        match = re.search(pattern, line)
        if match:
            data = [float(match.group(i)) for i in range(1, len(columns)+1)]
            metrics.append(data)

    metrics = pd.DataFrame(metrics, columns=columns)
    metrics.to_csv(os.path.join(os.path.dirname(path),
                                "metrics.csv"))

    return metrics


parser = argparse.ArgumentParser()
parser.add_argument('--roberta_log_path', type=str, help='roberta log path')
args = parser.parse_args()


if args.roberta_log_path:

    metrics = read_roberta_log(args.roberta_log_path)
    out_path = os.path.join(os.path.dirname(args.roberta_log_path),
                            "metrics.csv")
    metrics.to_csv(out_path)
    for c in ["wps"]:  # metrics.columns:
        metrics[c].to_csv(os.path.join(os.path.dirname(args.roberta_log_path),
                                       c+".csv"))
    print("Csv for model metrics written to", out_path)
