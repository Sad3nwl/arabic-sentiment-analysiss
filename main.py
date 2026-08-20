import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('arabic_reviews.csv')
# ========== INFO ABOUT DATA ==========
print(data.head(5))
print('='*10)
print(data.info())
print('='*10)
print(data.describe())
print('='*10)
print(f'shape:{data.shape}')
print(f'data columns:{data.columns}')
#=============================================MISSING VALUES & DUPLICATED
print('missing values :')
print(data.isnull().sum())
print('Total missing values:')
print(data.isnull().sum().sum())
print("Duplicate rows:",data.duplicated().sum())
def clean_text(text):
    text=re.sub(r"http\S+|www\S+", "", text)
    text=re.sub(r"<.*?>","",text)
    text=re.sub(r"[\u0617-\u061A\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
data['clean_text']=data['text'].apply(clean_text)
print(data[["text", "clean_text"]].head(10))
data.to_csv(
    "arabic_reviews_clean.csv",
    index=False,
    encoding="utf-8-sig"
)
print("Clean data saved successfully!")