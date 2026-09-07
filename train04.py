import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

# ============================================== تحميل الداتا
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

# ============================================== نفس معالجة النفي السابقة
NEGATION_WORDS = ['ليس', 'لا', 'لم', 'ما', 'غير', 'بدون', 'مش']

def handle_negation(text):
    words = str(text).split()
    result = []
    i = 0
    while i < len(words):
        if words[i] in NEGATION_WORDS and i + 1 < len(words):
            result.append(words[i] + '_' + words[i + 1])
            i += 2
        else:
            result.append(words[i])
            i += 1
    return ' '.join(result)

for df in [train, val, test]:
    df['neg_text'] = df['clean_text'].apply(handle_negation)

# ============================================== TF-IDF
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train['neg_text'])
X_val = vectorizer.transform(val['neg_text'])
X_test = vectorizer.transform(test['neg_text'])

y_train = train['label_text']
y_val = val['label_text']
y_test = test['label_text']

# ============================================== الموديلات الثلاث
lr = LogisticRegression(C=1, max_iter=1000, class_weight='balanced', random_state=42)

# LinearSVC ما بيعطي احتمالات مباشرة، فبنغلفها بـ CalibratedClassifierCV عشان تقدر تصوت بـ soft voting
svm = CalibratedClassifierCV(
    LinearSVC(class_weight='balanced', random_state=42, max_iter=5000),
    cv=3
)

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# ============================================== الدمج
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# ============================================== التقييم على الفاليديشن
val_preds = ensemble.predict(X_val)
print("=== نتائج الفاليديشن (Ensemble) ===")
print(classification_report(y_val, val_preds))
print("F1-macro:", f1_score(y_val, val_preds, average='macro'))

# ============================================== التقييم النهائي على التست
test_preds = ensemble.predict(X_test)
print("\n=== نتائج التست (Ensemble) ===")
print(classification_report(y_test, test_preds))
print("F1-macro:", f1_score(y_test, test_preds, average='macro'))

print("\nConfusion matrix (rows=true, cols=pred):")
labels_order = ['Negative', 'Positive', 'Mixed']
print(labels_order)
print(confusion_matrix(y_test, test_preds, labels=labels_order))

# ============================================== حفظ
joblib.dump(ensemble, 'sentiment_model_ensemble.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_ensemble.pkl')
print("\nsaved -> sentiment_model_ensemble.pkl, tfidf_vectorizer_ensemble.pkl")