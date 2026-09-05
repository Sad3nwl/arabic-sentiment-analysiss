import re
import pandas as pd
data = pd.read_csv('arabic_reviews.csv')
# ========================================== DATA UNDERSTANDING
print(data.head(5))
print('=' * 10)
print(data.info())
print('=' * 10)
print(data.describe())
print('=' * 10)
print(f'shape:{data.shape}')
print(f'data columns:{data.columns}')
# =============================================MISSING VALUES & DUPLICATED
print('missing values :')
print(data.isnull().sum())
print('Total missing values:')
print(data.isnull().sum().sum())
print("Duplicate rows:", data.duplicated().sum())


# =================================================TEXT CLEANING
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[\u0617-\u061A\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


data['clean_text'] = data['text'].apply(clean_text)
print(data[["text", "clean_text"]].head(10))

# ============================================== delet the empty rows
before = len(data)
data = data[data['clean_text'].str.len() > 0].reset_index(drop=True)
print(f"removed {before - len(data)} empty rows after cleaning")

data.drop(columns=["text"], inplace=True)
 
data.to_csv(
    "arabic_reviews_clean.csv",
    index=False,
    encoding="utf-8-sig"
)

