import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import config

th = config.PROB_THRESHOLD
df = pd.read_csv("oos_trades_2way.csv", parse_dates=["datetime"])
sig = df[df["proba"] >= th].copy()
sig["wib"] = sig["datetime"] + pd.Timedelta(hours=7)   # UTC -> WIB
sig["jam_wib"] = sig["wib"].dt.hour

print(f"Threshold {th} | total {len(sig)} sinyal (out-of-sample) | jam dalam WIB (UTC+7)\n")
tab = sig.groupby("jam_wib").agg(
    sinyal=("proba","size"),
    menang=("label", lambda x:(x==1).sum()),
).reset_index()
tab["winrate%"] = (tab["menang"]/tab["sinyal"]*100).round(0)
tab["bar"] = tab["sinyal"].apply(lambda n: "█"*n)
print("jam_WIB  sinyal  winrate%  distribusi")
for _,r in tab.iterrows():
    print(f"  {int(r['jam_wib']):02d}:xx    {int(r['sinyal']):3d}    {r['winrate%']:5.0f}    {r['bar']}")

# sesi trading
def sesi(h):
    if 14 <= h < 19: return "London (14-19 WIB)"
    if 19 <= h < 24: return "London+NY overlap (19-24 WIB)"
    if 0 <= h < 5:   return "New York (00-05 WIB)"
    return "Asia/sepi (05-14 WIB)"
sig["sesi"] = sig["jam_wib"].apply(sesi)
print("\nPer SESI:")
s = sig.groupby("sesi").agg(sinyal=("proba","size"), menang=("label",lambda x:(x==1).sum()))
s["winrate%"]=(s["menang"]/s["sinyal"]*100).round(0)
print(s.sort_values("sinyal",ascending=False).to_string())
