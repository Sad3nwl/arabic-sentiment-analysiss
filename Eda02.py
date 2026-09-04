import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('arabic_reviews_clean.csv')
data["len"] = data["clean_text"].astype(str).apply(len)
print("=== توزيع الفئات ===")
print(data["label_text"].value_counts())
print("\n=== إحصائيات طول النص حسب الفئة ===")
print(data.groupby("label_text")["len"].describe())