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
print(df_Superstore.dropna())

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
plt.show()
order_by_hour = df_Superstore['Hour'].value_counts().sort_index()
order_by_hour.plot(kind='bar', title='Số đơn hàng theo giờ trong ngày', ylabel='Số đơn', xlabel='Giờ')
plt.xticks(rotation=0)
plt.show()

#2. Sản phẩm nào bán chạy nhất?
top_products = df_Superstore['Product Name'].value_counts().head(10)
print("Top 10 sản phẩm bán chạy nhất:")
print(top_products)
top_products.plot(kind='barh', title='Top 10 sản phẩm bán chạy nhất', xlabel='Số lượng bán')
plt.gca().invert_yaxis()
plt.show()

#3. Doanh thu theo tháng / quý
revenue_by_month = df_Superstore.groupby('Month')['Sales'].sum()
revenue_by_month.plot(kind='bar', title='Doanh thu theo tháng', ylabel='Doanh thu')
plt.show()

revenue_by_quarter = df_Superstore.groupby('Quarter')['Sales'].sum()
revenue_by_quarter.plot(kind='bar', title='Doanh thu theo quý', ylabel='Doanh thu')
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
plt.show()

#7. Sản phẩm thường được mua cùng nhau
from itertools import combinations
from collections import Counter

# Gom các sản phẩm theo từng đơn hàng
orders = df_Superstore.groupby('Order ID')['Product Name'].apply(list)

# Tạo các cặp sản phẩm
combo_counter = Counter()
for order in orders:
    combos = combinations(sorted(set(order)), 2)
    combo_counter.update(combos)

# Hiển thị top 10 cặp được mua cùng nhau nhiều nhất
print("Top 10 cặp sản phẩm được mua cùng nhau:")
print(combo_counter.most_common(10))

#8. Khu vực nào mua nhiều?
if 'Region' in df_Superstore.columns:
    df_Superstore['Region'].value_counts().plot(kind='bar', title='Số đơn hàng theo khu vực')
    plt.xlabel('Region')
    plt.ylabel('Số đơn hàng')
    plt.show()
else:
    print("Không có thông tin khu vực trong dữ liệu.")


