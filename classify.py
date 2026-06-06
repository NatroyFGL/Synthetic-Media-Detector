import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

#Config
TRAIN_DIR = "./train"
TEST_DIR  = "./test"
IMG_SIZE  = (32, 32)   


def load_images(folder, img_size=IMG_SIZE):
    X, y = [], []
    for label, class_name in enumerate(["REAL", "FAKE"]):
        class_path = os.path.join(folder, class_name)
        for fname in tqdm(os.listdir(class_path), desc=class_name):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            img = Image.open(os.path.join(class_path, fname)).convert("RGB")
            img = img.resize(img_size)
            X.append(np.array(img).flatten())  # flatten to 1D: 32x32x3 = 3072
            y.append(label)
    return np.array(X), np.array(y)

print("Loading training data...")
X_train, y_train = load_images(TRAIN_DIR)

print("Loading test data...")
X_test, y_test = load_images(TEST_DIR)

#Normalize pixel values to [0, 1]
X_train = X_train / 255.0
X_test  = X_test  / 255.0

#Scale for better convergence
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

#Model
model = LogisticRegression(max_iter=1000, verbose=1)
model.fit(X_train, y_train)

#Evaluate
y_pred = model.predict(X_test)

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Real vs AI-Generated")
plt.show()

#Predict a single image
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    x   = np.array(img).flatten() / 255.0
    x   = scaler.transform([x])
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0]
    label = "REAL" if pred == 0 else "FAKE"
    print(f"{img_path} → {label} (confidence: {max(prob):.2%})")

