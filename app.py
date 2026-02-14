import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1️⃣ قراءة البيانات
# -------------------------------
DATA_PATH = "data.csv" # تأكد من مسار ملف البيانات
df = pd.read_csv(DATA_PATH, encoding='latin-1')  # بعض ملفات spam.csv تحتاج encoding='latin-1'

# نفترض إن الأعمدة المهمة هي 'v1' (label) و 'v2' (text)
df = df[['Category', 'Message']]
df.columns = ['label', 'email_text']

# -------------------------------
# 2️⃣ تجهيز البيانات
# -------------------------------
X = df['email_text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# -------------------------------
# 3️⃣ تدريب موديل Naive Bayes
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# 4️⃣ تقييم الموديل
# -------------------------------
y_pred = model.predict(X_test)
st.write("**Model Accuracy:**", accuracy_score(y_test, y_pred))
st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# -------------------------------
# 5️⃣ حفظ الموديل والفكتورايزر
# -------------------------------
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# -------------------------------
# 6️⃣ واجهة Streamlit
# -------------------------------
st.title("📧 Email Spam Detector")

email_input = st.text_area("Enter Email Text:")

if st.button("Predict"):
    if email_input:
        vect_input = vectorizer.transform([email_input])
        prediction = model.predict(vect_input)[0]
        st.success(f"This email is: **{prediction.upper()}**")
    else:
        st.warning("Please enter some text!")
