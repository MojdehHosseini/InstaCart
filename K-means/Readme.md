# Customer Segmentation using K-Means Clustering

## 📌 Project Overview
This project applies **K-Means clustering** to segment customers based on their shopping behavior. The segmentation helps identify customer groups such as **loyal customers, frequent buyers, occasional shoppers, and potential churners**. These insights can be used for **personalized marketing, customer targeting, and anomaly detection**.

## 📂 Dataset Information
The dataset contains transaction details, including:
- **Product details** (`product_id`, `product_name`, `aisle_id`, `department_id`, etc.).
- **Order details** (`order_id`, `user_id`, `order_number`, `order_hour_of_day`, `days_since_prior_order`).
- **Customer behavior metrics** (number of orders, frequency, reorder patterns, etc.).

## 🚀 Features Used for Segmentation
The following customer-based features were derived from the dataset:

| Feature Name | Description |
|-------------|-------------|
| `total_orders` | Total number of orders placed by the user |
| `total_products_ordered` | Total number of products purchased by the user |
| `average_cart_size` | Average number of products per order |
| `reorder_ratio` | Percentage of products reordered by the user |
| `days_since_last_order` | Days since the customer's last order |
| `order_frequency` | Average time between orders |
| `most_frequent_department` | The department from which the user buys the most |
| `most_frequent_aisle` | The aisle where the user buys the most products |
| `order_hour_preference` | Most frequent order time |

## 🛠 Installation & Setup
### **1. Install Required Libraries**
Ensure you have Python and the necessary libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### **2. Clone the Repository**
```bash
git clone https://github.com/your-repository/customer-segmentation.git
cd customer-segmentation
```

### **3. Run the Segmentation Script**
```bash
python segment_customers.py
```

## 🔍 Steps in Customer Segmentation
### **1️⃣ Data Preprocessing**
- Handling missing values.
- Selecting relevant numerical features.
- Standardizing data using `StandardScaler`.

### **2️⃣ Finding the Optimal Number of Clusters**
Using the **Elbow Method** to determine the best value of `k`:
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o', linestyle='-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()
```

### **3️⃣ Applying K-Means Clustering**
```python
optimal_k = 3  # Determined using the elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_df["Cluster"] = kmeans.fit_predict(X_scaled)
```

### **4️⃣ Visualizing the Segments**
#### **Scatter Plot of Two Features**
```python
import seaborn as sns

sns.scatterplot(x=customer_df["total_orders"], y=customer_df["total_products_ordered"], hue=customer_df["Cluster"], palette="viridis")
```

#### **PCA for Reducing Dimensions and Plotting in 2D**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
customer_df["PCA1"], customer_df["PCA2"] = X_pca[:, 0], X_pca[:, 1]

sns.scatterplot(x=customer_df["PCA1"], y=customer_df["PCA2"], hue=customer_df["Cluster"], palette="viridis")
```


## 👨‍💻 Author
- **Mojdeh Hosseini**
- **Contact:** mojdeh.h.hosseini@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/mojdeh-haghighat-hosseini/



