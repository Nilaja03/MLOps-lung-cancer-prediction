from flask import Flask, render_template_string
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

def train_model():
    # Load and Preprocess
    df = pd.read_csv('lungcancer.csv')
    df.columns = df.columns.str.strip()
    le = LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])

    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and Train Model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    
    # Evaluate
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return round(accuracy * 100, 2)

@app.route('/')
def home():
    acc = train_model()
    return f"<h1>Lung Cancer Model Accuracy: {acc}%</h1>"

if __name__ == '__main__':
    # Use 0.0.0.0 for Docker compatibility
    app.run(host='0.0.0.0', port=5000)
