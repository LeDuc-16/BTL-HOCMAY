import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Đọc dữ liệu
house_price_dataset = pd.read_csv("/Users/leduc/TrainHousePrice/HousingPriceStreamlit/HousePrice.csv")
training_df = house_price_dataset[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'SalePrice']]
X = training_df[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr']]
y = training_df['SalePrice']

# Chia dữ liệu thành tập huấn luyện, xác thực và kiểm tra
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Tạo mô hình Bagging với Linear Regression
bagging_model = BaggingRegressor(estimator=LinearRegression(), n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)

# Giao diện Streamlit
st.title("Dự đoán giá nhà sử dụng Bagging Linear Regression")

# Nhập liệu từ người dùng
lot_area = st.number_input("Lot Area", min_value=0)
year_built = st.number_input("Year Built", min_value=1900)
first_flr_sf = st.number_input("1st Floor Square Feet", min_value=0)
second_flr_sf = st.number_input("2nd Floor Square Feet", min_value=0)
bedroom_abv_gr = st.number_input("Bedrooms Above Grade", min_value=0)

# Khi nhấn nút Dự đoán
if st.button("Dự đoán"):
    user_input = np.array([[lot_area, year_built, first_flr_sf, second_flr_sf, bedroom_abv_gr]])
    prediction = bagging_model.predict(user_input)
    
    # Đánh giá mô hình
    predictions_valid = bagging_model.predict(X_valid)
    mse = mean_squared_error(y_valid, predictions_valid)
    r2 = r2_score(y_valid, predictions_valid)

    # Hiển thị đánh giá mô hình
    st.write(f"Đánh giá mô hình:")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R²): {r2:.2f}")
    
    # Tìm giá trị thực tế tương ứng
    actual_value = training_df[
        (training_df['LotArea'] == lot_area) &
        (training_df['YearBuilt'] == year_built) &
        (training_df['1stFlrSF'] == first_flr_sf) &
        (training_df['2ndFlrSF'] == second_flr_sf) &
        (training_df['BedroomAbvGr'] == bedroom_abv_gr)
    ]

    # Hiển thị kết quả
    if not actual_value.empty:
        actual_price = actual_value['SalePrice'].values[0]
        st.write(f"Giá nhà dự đoán: {int(prediction[0])}")
        st.write(f"Giá nhà thực tế: {actual_price}")
    else:
        st.write(f"Không tìm thấy dữ liệu thực tế cho thông tin đã nhập.")


