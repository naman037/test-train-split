import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading massive datasets. This might take a second...")

# --- 1. LOAD THE DATA ---
real_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

real_news["label"] = 1
fake_news["label"] = 0

all_news = pd.concat([real_news, fake_news])
all_news = all_news.sample(frac=1).reset_index(drop=True)

print("Data loaded! Now chopping it into Training and Testing piles...")

# --- 2. PREPARE FOR TRAINING ---
X_text = all_news['title']
y_answers = all_news['label']

# Chop the data! 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X_text, y_answers, test_size=0.2, random_state=42)

print("Setting up the Word Counter...")

# Create the Word Counter and teach it the vocabulary from the training set
counter = CountVectorizer()
X_train_math = counter.fit_transform(X_train) 

print("Training the AI Brain. This might take 10-20 seconds...")

# --- 3. TRAIN THE AI ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_math, y_train)

print("Brain trained! Now taking the Final Exam...")

# --- 4. TEST THE AI ---
# Translate the hidden exam questions into math
X_test_math = counter.transform(X_test)

# Have the AI guess the answers
predictions = model.predict(X_test_math)

# Grade the exam! 
score = accuracy_score(y_test, predictions)

print(f"Final Accuracy Score: {score * 100:.2f}%")
print("Freezing the AI brain...")

# Save the trained ML Model
joblib.dump(model, 'fake_news_model.pkl')
#Save the Word Counter (so it remembers the exact math dictionary)
joblib.dump(counter, 'word_counter.pkl')

print("Brain successfully saved to your folder! 🧠")