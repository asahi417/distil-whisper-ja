import pandas as pd
from glob import glob
import evaluate


metric = evaluate.load("wer")


def normalization(text: str):
    text = text.replace(" ", "")
    return text


summary = []
pred_list = {}
for i in glob("eval/*/all_predictions.test.csv"):
    model = i.split("/")[1].split(".")[0]
    data = i.split("/")[1].split(".")[-1]
    df = pd.read_csv(i, index_col=0)
    target = [normalization(i) for i in df["Norm Target"].values]
    pred = [normalization(i) for i in df["Norm Pred"].values]
    wer = metric.compute(predictions=pred, references=target) * 100
    summary.append({"model": model, "data": data, "wer": wer})
    if data not in pred_list:
        pred_list[data] = {}
    if "target" not in pred_list[data]:
        pred_list[data]["target"] = target
    pred_list[data][model] = pred


df = pd.DataFrame(summary)
df = df.sort_values(by=["data", "model"])
df.to_csv("eval/metric.csv")
for k, v in pred_list.items():
    pd.DataFrame(v).to_csv(f"eval/prediction.{k}.csv")