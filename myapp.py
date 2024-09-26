import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import BaggingRegressor

# Load dữ liệu
@st.cache_data
def load_data():
    data = pd.read_csv('housing_prices.csv')  # Đảm bảo bạn có file housing_prices.csv từ Kaggle
    return data

# Tiền xử lý dữ liệu
def preprocess_data(data):
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr']
    X = data[features]
    y = data['SalePrice']
    return X, y

# Tải dữ liệu
data = load_data()
X, y = preprocess_data(data)

# Chia tập dữ liệu để huấn luyện mô hình
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thêm chọn loại mô hình
model_type = st.selectbox("Chọn mô hình huấn luyện", ["Linear Regression", "Neural Network", "Ridge Regression", "Bagging"])

# Huấn luyện mô hình dựa trên lựa chọn
if model_type == "Linear Regression":
    model = LinearRegression()
elif model_type == "Neural Network":
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
elif model_type == "Ridge Regression":
    model = Ridge(alpha=1.0)
elif model_type == "Bagging":
    base_model = LinearRegression()  # Bạn có thể chọn mô hình cơ sở là LinearRegression hoặc các mô hình khác
    model = BaggingRegressor(base_model, n_estimators=10, random_state=42)

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán giá nhà trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán các chỉ số MAE, RMSE, R^2
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Tiêu đề ứng dụng
st.title("Dự đoán giá nhà")

# Input từ người dùng
st.header("Nhập các thông tin về ngôi nhà:")
lot_area = st.number_input("Diện tích lô đất (LotArea):", min_value=1000, max_value=20000, value=5000)
year_built = st.number_input("Năm xây dựng (YearBuilt):", min_value=1800, max_value=2023, value=2000)
first_flr_sf = st.number_input("Diện tích tầng 1 (1stFlrSF):", min_value=500, max_value=3000, value=1000)
second_flr_sf = st.number_input("Diện tích tầng 2 (2ndFlrSF):", min_value=0, max_value=3000, value=500)
bedroom_abv_gr = st.number_input("Số phòng ngủ (BedroomAbvGr):", min_value=1, max_value=10, value=3)

# Button để dự đoán
if st.button("Dự đoán giá nhà"):
    # Tạo mảng numpy chứa các giá trị nhập vào
    input_data = np.array([[lot_area, year_built, first_flr_sf, second_flr_sf, bedroom_abv_gr]])
    
    # Dự đoán giá nhà
    predicted_price = model.predict(input_data)
    
    # Hiển thị kết quả
    st.subheader(f"Giá nhà dự đoán: {predicted_price[0]:,.2f} USD")

# Hiển thị các chỉ số thống kê
st.header("Đánh giá mô hình")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.2f}")

# Vẽ biểu đồ dự đoán vs giá trị thật
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.title(f"Dự đoán vs Giá trị thật - {model_type}")
plt.xlabel("Giá trị thật (Actual SalePrice)")
plt.ylabel("Giá trị dự đoán (Predicted SalePrice)")
plt.grid(True)
st.pyplot(plt)
