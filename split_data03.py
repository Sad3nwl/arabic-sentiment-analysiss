import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('arabic_reviews_clean.csv')

# الخطوة 1: فصل التست (10%)
train_val, test = train_test_split(
    data,
    test_size=0.1,
    stratify=data['label_text'],
    random_state=42
)

# الخطوة 2: فصل الفاليديشن من الباقي (10% من الأصل)
train, val = train_test_split(
    train_val,
    test_size=1/9,
    stratify=train_val['label_text'],
    random_state=42
)

print("train:", train.shape, train['label_text'].value_counts(normalize=True).round(3).to_dict())
print("val:", val.shape, val['label_text'].value_counts(normalize=True).round(3).to_dict())
print("test:", test.shape, test['label_text'].value_counts(normalize=True).round(3).to_dict())

train.to_csv('train.csv', index=False, encoding='utf-8-sig')
val.to_csv('val.csv', index=False, encoding='utf-8-sig')
test.to_csv('test.csv', index=False, encoding='utf-8-sig')