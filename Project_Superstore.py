import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Đọc dữ liệu và hiện thị dữ liệu Superstore
df_Superstore = pd.read_csv('Superstore.csv')
print("Thông tin cột dữ liệu:")
print(df_Superstore.columns)
print(df_Superstore.head())
print(df_Superstore.info())
print(df_Superstore.describe())
print(df_Superstore.shape)


#Kiểm tra dữ liệu 
print(df_Superstore.dtypes)
print(df_Superstore.isnull().sum())
df_Superstore = df_Superstore.dropna()
print(df_Superstore)

#Chuyển đổi cột ngày
df_Superstore['Order Date'] = pd.to_datetime(df_Superstore['Order Date'])
# Tạo thêm các cột phân tích thời gian
df_Superstore['Weekday'] = df_Superstore['Order Date'].dt.day_name()
df_Superstore['Hour'] = df_Superstore['Order Date'].dt.hour
df_Superstore['Month'] = df_Superstore['Order Date'].dt.month
df_Superstore['Quarter'] = df_Superstore['Order Date'].dt.quarter
#1. Khách hàng thường mua vào thời điểm nào?
order_by_day = df_Superstore['Weekday'].value_counts().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
order_by_day.plot(kind='bar', title='Số đơn hàng theo ngày trong tuần', ylabel='Số đơn', xlabel='Thứ')
plt.xticks(rotation=45)
plt.savefig("Số đơn hàng theo ngày trong tuần.png")
plt.show()
order_by_hour = df_Superstore['Hour'].value_counts().sort_index()
order_by_hour.plot(kind='bar', title='Số đơn hàng theo giờ trong ngày', ylabel='Số đơn', xlabel='Giờ')
plt.xticks(rotation=0)
plt.savefig("Số đơn hàng theo giờ trong ngày.png")
plt.show()

#2. Sản phẩm nào bán chạy nhất?
top_products = df_Superstore['Product Name'].value_counts().head(10)
print("Top 10 sản phẩm bán chạy nhất:")
print(top_products)
top_products.plot(kind='barh', title='Top 10 sản phẩm bán chạy nhất', xlabel='Số lượng bán')
plt.gca().invert_yaxis()
plt.savefig("Top 10 sản phẩm bán chạy nhất.png")
plt.show()

#3. Doanh thu theo tháng / quý
revenue_by_month = df_Superstore.groupby('Month')['Sales'].sum()
revenue_by_month.plot(kind='bar', title='Doanh thu theo tháng', ylabel='Doanh thu')
plt.savefig("Doanh thu theo tháng.png")
plt.show()

revenue_by_quarter = df_Superstore.groupby('Quarter')['Sales'].sum()
revenue_by_quarter.plot(kind='bar', title='Doanh thu theo quý', ylabel='Doanh thu')
plt.savefig("Doanh thu theo quý.png")
plt.show()

#4. Khách hàng có xu hướng mua lại không?
repeat_customers = df_Superstore['Customer ID'].value_counts()
print("Khách hàng mua lại (có hơn 1 đơn):", sum(repeat_customers > 1))
print("Tổng khách hàng:", len(repeat_customers))
print("Tỷ lệ mua lại: {:.2f}%".format((sum(repeat_customers > 1) / len(repeat_customers)) * 100))

#5. Tần suất mua hàng của khách hàng
purchase_frequency = df_Superstore.groupby('Customer ID')['Order ID'].nunique()
purchase_frequency.hist(bins=20)
plt.title('Tần suất mua hàng của khách hàng')
plt.xlabel('Số lần mua')
plt.ylabel('Số lượng khách')
plt.savefig("Tần suất mua hàng của khách hàng.png")
plt.show()

#6. Phân nhóm khách hàng theo mức chi tiêu
customer_spend = df_Superstore.groupby('Customer ID')['Sales'].sum()

# Gán nhãn theo mức tiêu dùng
def spending_level(x):
    if x < 500:
        return 'Ít'
    elif x < 2000:
        return 'Trung bình'
    else:
        return 'Nhiều'

df_Superstore['Spending Level'] = df_Superstore['Customer ID'].map(customer_spend).map(spending_level)
sns.countplot(data=df_Superstore, x='Spending Level')
plt.title('Phân nhóm khách hàng theo mức chi tiêu')
plt.savefig("Nhóm khách hàng theo mức chi tiêu.png")
plt.show()

#7. Sản phẩm thường được mua cùng nhau
from itertools import combinations
from collections import Counter

# Gom các sản phẩm theo từng đơn hàng
orders_by_product = df_Superstore.groupby('Order ID')['Product Name'].apply(list)

# Tạo các cặp sản phẩm
product_combo_counter = Counter()
for order in orders_by_product:
    combos = combinations(sorted(set(order)), 2)
    product_combo_counter.update(combos)

# Hiển thị top 10 cặp được mua cùng nhau nhiều nhất
print("Top 10 cặp sản phẩm được mua cùng nhau:")
print(product_combo_counter.most_common(10))

#8. Khu vực nào mua nhiều?
if 'Region' in df_Superstore.columns:
    df_Superstore['Region'].value_counts().plot(kind='bar', title='Số đơn hàng theo khu vực')
    plt.xlabel('Region')
    plt.ylabel('Số đơn hàng')
    plt.savefig("Khu vực mua nhiều nhất.png")
    plt.show()
else:
    print("Không có thông tin khu vực trong dữ liệu.")

#9. Thống kê suy diễn và Mô hình học máy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df_ml = df_Superstore[['Sales', 'Profit', 'Discount', 'Quantity', 'Category', 'Sub-Category', 'Segment', 'Region']].copy()

categorical_cols = ['Category', 'Sub-Category', 'Segment', 'Region']
df_ml[categorical_cols] = df_ml[categorical_cols].apply(LabelEncoder().fit_transform)

X = df_ml.drop('Sales', axis=1)
y = df_ml['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("Linear Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("R²:", r2_score(y_test, y_pred_lr))

from sklearn.ensemble import RandomForestRegressor

model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("Random Forest Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("R²:", r2_score(y_test, y_pred_rf))

from xgboost import XGBRegressor

model_xgb = XGBRegressor(random_state=42)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

print("XGBoost Regression:")
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
print("R²:", r2_score(y_test, y_pred_xgb))

results = pd.DataFrame({
    "Mô hình": ["Linear Regression", "Random Forest", "XGBoost"],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    ],
    "R²": [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_rf),
        r2_score(y_test, y_pred_xgb),
    ]
})
print(results)

import scipy.stats as stats
grouped_segment = [group["Sales"].values for name, group in df_Superstore.groupby("Segment")]
anova_result = stats.f_oneway(*grouped_segment)
print("ANOVA giữa các phân khúc khách hàng:")
print("F-statistic:", anova_result.statistic)
print("p-value:", anova_result.pvalue)

#10. Thống kê suy diễn: Kiểm định sự khác biệt doanh thu giữa các khu vực
from scipy.stats import f_oneway
# Kiểm tra sự khác biệt doanh thu trung bình giữa các khu vực
regions = df_Superstore['Region'].unique()
sales_by_region = [df_Superstore[df_Superstore['Region'] == region]['Sales'] for region in regions]
f_stat, p_value = f_oneway(*sales_by_region)
print("=== Thống kê suy diễn: ANOVA giữa các khu vực ===")
print(f"F-statistic: {f_stat}")
print(f"P-value: {p_value}")
if p_value < 0.05:
    print("=> Có sự khác biệt có ý nghĩa thống kê về doanh thu trung bình giữa các khu vực.")
else:
    print("=> Không có sự khác biệt có ý nghĩa thống kê về doanh thu trung bình giữa các khu vực.")
#11.Mô hình học máy: Dự đoán đơn hàng có lãi hay không
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Tạo biến mục tiêu
df_Superstore['Profitable'] = (df_Superstore['Profit'] > 0).astype(int)
features = ['Sales', 'Discount', 'Quantity', 'Category', 'Sub-Category', 'Region', 'Segment']
X = df_Superstore[features]
y = df_Superstore['Profitable']
X_encoded = X.copy()
for col in X_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# 1. Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

# 2. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))

# 3. Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("\n=== Gradient Boosting ===")
print(classification_report(y_test, y_pred_gb))



