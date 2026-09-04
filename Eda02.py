import re
import pandas as pd
import  seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
data=pd.read_csv('arabic_reviews_clean.csv')
data["len"] = data["clean_text"].astype(str).apply(len)
print("=== توزيع الفئات ===")
print(data["label_text"].value_counts())
print("\n=== إحصائيات طول النص حسب الفئة ===")
print(data.groupby("label_text")["len"].describe())
plt.figure(figsize=(8,5))
sns.boxplot(data,x='label_text', y='len',color='#FF69B4')
plt.ylim(0, data['len'].quantile(0.95))
plt.title('توزيع طول النص حسب الفئة')
plt.show()
stopwords = {"من","في","على","الى","إلى","عن","مع","هذا","هذه","التي",
             "الذي","كان","كانت","هو","هي","أن","إن","لا","لم","ما"}
def top_words(texts, n=15):
    words = []
    for t in texts:
        words.extend(re.findall(r'[\u0600-\u06FF]+', str(t)))
    words = [w for w in words if w not in stopwords and len(w) > 1]
    return Counter(words).most_common(n)
for label in data['label_text'].unique():
    subset = data.loc[data['label_text'] == label, 'clean_text']
    print(f"\n=== {label} ===")
    for word, count in top_words(subset):
        print(word, count)
short = data[data['len'] < 10]
print("number of short word", len(short))
print(short[['clean_text', 'label_text']].head(10))
def has_elongation(text):
    return bool(re.search(r'(.)\1{2,}', str(text)))
data['has_elongation'] = data['clean_text'].apply(has_elongation)
print("Percentage of texts containing repeated letters", data['has_elongation'].mean().round(3))
print(data.groupby('label_text')['has_elongation'].mean())
plt.figure(figsize=(8, 5))
sns.histplot(
    data=data, x='len', hue='label_text',
    bins=60, element='step', stat='density', common_norm=False
)
plt.xlim(0, data['len'].quantile(0.95))
plt.title('توزيع طول النص حسب الفئة')
plt.savefig('length_by_label_hist.png', dpi=150)
plt.show()