import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading massive datasets. This might take a second...")
real_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

real_news["label"] = 1
fake_news["label"] = 0

all_news = pd.concat([real_news, fake_news])
all_news = all_news.sample(frac=1).reset_index(drop=True)

print("Data loaded! Now chopping it into Training and Testing piles...")
X_text = all_news['title']
y_answers = all_news['label']
X_train, X_test, y_train, y_test = train_test_split(X_text, y_answers, test_size=0.2, random_state=42)

print("Setting up the Word Counter...")
counter = CountVectorizer()
X_train_math = counter.fit_transform(X_train) 

print("Training the AI Brain. This might take 10-20 seconds...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_math, y_train)

print("Brain trained! Now taking the Final Exam...")
X_test_math = counter.transform(X_test)
predictions = model.predict(X_test_math)

score = accuracy_score(y_test, predictions)

print(f"Final Accuracy Score: {score * 100:.2f}%")
print("Freezing the AI brain...")

joblib.dump(model, 'fake_news_model.pkl')

joblib.dump(counter, 'word_counter.pkl')

print("Brain successfully saved to your folder! 🧠")