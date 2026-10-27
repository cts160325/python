import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score, RocCurveDisplay
)

# ============================================================
# ĐỀ 39: PHÂN TÍCH DỮ LIỆU KIỂM THỬ VÀ DỰ ĐOÁN KHẢ NĂNG
# TEST CASE THẤT BẠI BẰNG LOGISTIC REGRESSION
# ============================================================

# 1. ĐỌC DỮ LIỆU
df = pd.read_csv("test_case_data_de39.csv")
print("===== 1. DỮ LIỆU BAN ĐẦU =====")
print("Kích thước:", df.shape)
print(df.head())
print("\nKiểu dữ liệu:")
print(df.dtypes)

# 2. KIỂM TRA DỮ LIỆU
print("\n===== 2. KIỂM TRA DỮ LIỆU =====")
print("\nGiá trị thiếu:")
print(df.isnull().sum())

print("\nSố dòng trùng:", df.duplicated().sum())
print(df[df.duplicated()])

print("\nComplexity ngoài khoảng 1-10:")
print(df[(df["Complexity"] < 1) | (df["Complexity"] > 10)])

print("\nCodeChanges âm:")
print(df[df["CodeChanges"] < 0])

# Ngoại lệ ExecutionTime theo IQR
q1 = df["ExecutionTime"].quantile(0.25)
q3 = df["ExecutionTime"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print(f"\nNgưỡng ngoại lệ ExecutionTime theo IQR: [{lower:.2f}, {upper:.2f}]")
print(df[(df["ExecutionTime"] < lower) | (df["ExecutionTime"] > upper)])

# 3. TIỀN XỬ LÝ
clean = df.drop_duplicates().copy()

# Giá trị không hợp lệ chuyển thành thiếu để xử lý bằng median
clean.loc[~clean["Complexity"].between(1, 10), "Complexity"] = np.nan
clean.loc[clean["CodeChanges"] < 0, "CodeChanges"] = np.nan

# Ngoại lệ ExecutionTime: winsorize/capping theo ngưỡng IQR
q1 = clean["ExecutionTime"].quantile(0.25)
q3 = clean["ExecutionTime"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
clean["ExecutionTime"] = clean["ExecutionTime"].clip(lower=lower, upper=upper)

print("\n===== 3. SAU TIỀN XỬ LÝ CƠ BẢN =====")
print("Kích thước sau xóa trùng:", clean.shape)

# 4. PHÂN TÍCH THỐNG KÊ
numeric_cols = [
    "ExecutionTime", "Complexity", "PreviousFailures",
    "CodeChanges", "TesterExperience", "Failed"
]

print("\n===== 4. THỐNG KÊ MÔ TẢ =====")
print(clean[numeric_cols].describe())

print("\nMean:")
print(clean[numeric_cols].mean())
print("\nMedian:")
print(clean[numeric_cols].median())
print("\nVariance:")
print(clean[numeric_cols].var())
print("\nStandard deviation:")
print(clean[numeric_cols].std())

corr = clean[numeric_cols].corr()
print("\nCorrelation:")
print(corr)

# 5. TRỰC QUAN HÓA
# Biểu đồ 1: số lượng Pass / Fail
plt.figure(figsize=(6, 4))
clean["Failed"].value_counts().sort_index().plot(kind="bar")
plt.xticks([0, 1], ["Pass (0)", "Fail (1)"], rotation=0)
plt.xlabel("Kết quả")
plt.ylabel("Số test case")
plt.title("Phân bố kết quả test case")
plt.tight_layout()
plt.savefig("bieu_do_1_pass_fail.png", dpi=160)
plt.show()

# Biểu đồ 2: ma trận tương quan
plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Hệ số tương quan")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Ma trận tương quan các biến số")
plt.tight_layout()
plt.savefig("bieu_do_2_correlation.png", dpi=160)
plt.show()

# 6. CHUẨN BỊ DỮ LIỆU CHO LOGISTIC REGRESSION
features = [
    "ExecutionTime", "Complexity", "PreviousFailures",
    "CodeChanges", "TesterExperience", "TestType"
]
X = clean[features]
y = clean["Failed"]

numeric_features = [
    "ExecutionTime", "Complexity", "PreviousFailures",
    "CodeChanges", "TesterExperience"
]
categorical_features = ["TestType"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 7. CHIA TRAIN / TEST: 80% / 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=39, stratify=y
)

# 8. HUẤN LUYỆN
model.fit(X_train, y_train)

# 9. DỰ ĐOÁN
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 10. ĐÁNH GIÁ
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n===== 5. KẾT QUẢ LOGISTIC REGRESSION =====")
print(f"Train: {len(X_train)} test case")
print(f"Test : {len(X_test)} test case")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Pass", "Fail"]))

# Biểu đồ 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pass", "Fail"])
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("bieu_do_3_confusion_matrix.png", dpi=160)
plt.show()

# Biểu đồ 4 (bổ sung): ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Logistic Regression")
plt.tight_layout()
plt.savefig("bieu_do_4_roc_curve.png", dpi=160)
plt.show()

# 11. LƯU KẾT QUẢ DỰ ĐOÁN
ket_qua = X_test.copy()
ket_qua["ThucTe"] = y_test
ket_qua["DuDoan"] = y_pred
ket_qua["XacSuatThatBai"] = np.round(y_prob, 4)
ket_qua.to_csv("ket_qua_du_doan_de39.csv", index=False, encoding="utf-8-sig")

print("\nĐã lưu: ket_qua_du_doan_de39.csv")
print("Đã lưu 4 biểu đồ PNG.")
