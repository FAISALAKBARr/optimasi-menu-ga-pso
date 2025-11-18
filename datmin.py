"""
DATA MINING PROJECT: SAMPLE SUPERSTORE ANALYSIS (REVISED VERSION)
===================================================================
BAGIAN CLUSTERING DIPERBAIKI: Transaction-Based Clustering (Bukan RFM)

Kelompok:
- MOCHAMAD FAISAL AKBAR (L0122094)
- JASSON FRANKLYN WANG (L0122081)
- FARRAS ARKAN WARDANA (L0123052)

Program Studi Informatika
Fakultas Teknologi Informasi dan Sains Data
Universitas Sebelas Maret
2025
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Clustering Libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Classification Libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("DATA MINING: SAMPLE SUPERSTORE ANALYSIS (REVISED)")
print("="*80)
print()

# ============================================================================
# LOAD DATASET
# ============================================================================
print("[1] LOADING DATASET...")
print("-" * 80)

try:
    df = pd.read_csv('/kaggle/input/sample-supermarket-dataset/SampleSuperstore.csv', encoding='latin-1')
    print("✓ Dataset loaded from Kaggle input")
except:
    try:
        df = pd.read_csv('SampleSuperstore.csv', encoding='latin-1')
        print("✓ Dataset loaded from local file")
    except:
        print("✗ Error: Dataset not found. Please upload 'SampleSuperstore.csv'")
        exit()

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print()

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())
print()

# ============================================================================
# DATA EXPLORATION
# ============================================================================
print("\n[2] DATA EXPLORATION")
print("-" * 80)

print("\nDataset Info:")
print(df.info())
print()

print("\nMissing Values:")
print(df.isnull().sum())
print()

print("\nNumerical Columns Statistics:")
print(df.describe())
print()

print("\nCategorical Columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts().head())

# ============================================================================
# PART 1: TRANSACTION PATTERN CLUSTERING USING K-MEANS
# ============================================================================
print("\n" + "="*80)
print("PART 1: TRANSACTION PATTERN CLUSTERING (K-MEANS)")
print("="*80)
print("\nCATATAN: Clustering dilakukan pada TRANSAKSI, bukan Customer")
print("Tujuan: Mengidentifikasi pola transaksi untuk optimalisasi strategi")
print("="*80)

# ============================================================================
# 1.1 FEATURE ENGINEERING FOR CLUSTERING
# ============================================================================
print("\n[1.1] FEATURE ENGINEERING FOR TRANSACTION CLUSTERING...")
print("-" * 80)

# Buat copy dataset untuk clustering
df_cluster = df.copy()

# Remove rows with missing critical values
df_cluster = df_cluster.dropna(subset=['Sales', 'Profit', 'Quantity', 'Discount'])

# Create additional features
df_cluster['Profit_Margin'] = (df_cluster['Profit'] / df_cluster['Sales']) * 100
df_cluster['Profit_Margin'] = df_cluster['Profit_Margin'].replace([np.inf, -np.inf], 0)
df_cluster['Price_Per_Item'] = df_cluster['Sales'] / df_cluster['Quantity']
df_cluster['Is_Discounted'] = (df_cluster['Discount'] > 0).astype(int)

print(f"Total transactions for clustering: {len(df_cluster)}")
print("\nNew features created:")
print("  - Profit_Margin: (Profit/Sales) * 100")
print("  - Price_Per_Item: Sales/Quantity")
print("  - Is_Discounted: 1 if Discount > 0, else 0")
print()

# ============================================================================
# 1.2 SELECT AND ENCODE FEATURES
# ============================================================================
print("\n[1.2] SELECTING AND ENCODING FEATURES...")
print("-" * 80)

# Numerical features
numerical_features = ['Sales', 'Quantity', 'Discount', 'Profit', 
                     'Profit_Margin', 'Price_Per_Item']

# Categorical features to encode
categorical_features = ['Segment', 'Category', 'Region']

# Encode categorical variables
label_encoders = {}
for col in categorical_features:
    if col in df_cluster.columns:
        le = LabelEncoder()
        df_cluster[f'{col}_Encoded'] = le.fit_transform(df_cluster[col])
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} categories")

print()

# Combine all features for clustering
clustering_features = numerical_features + [f'{col}_Encoded' for col in categorical_features]
X_cluster = df_cluster[clustering_features].copy()

print(f"Total features for clustering: {len(clustering_features)}")
print(f"Features: {clustering_features}")
print()

print("\nClustering data sample:")
print(X_cluster.head())
print()

# ============================================================================
# 1.3 FEATURE VISUALIZATION BEFORE CLUSTERING
# ============================================================================
print("\n[1.3] VISUALIZING FEATURE DISTRIBUTIONS...")
print("-" * 80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature Distributions for Transaction Clustering', fontsize=16, fontweight='bold')

# Sales distribution
axes[0, 0].hist(df_cluster['Sales'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Sales Distribution')
axes[0, 0].set_xlabel('Sales ($)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].axvline(df_cluster['Sales'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ${df_cluster["Sales"].mean():.2f}')
axes[0, 0].legend()

# Profit distribution
axes[0, 1].hist(df_cluster['Profit'], bins=50, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Profit Distribution')
axes[0, 1].set_xlabel('Profit ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df_cluster['Profit'].mean(), color='red', linestyle='--',
                   label=f'Mean: ${df_cluster["Profit"].mean():.2f}')
axes[0, 1].legend()

# Discount distribution
axes[0, 2].hist(df_cluster['Discount'], bins=30, color='salmon', edgecolor='black')
axes[0, 2].set_title('Discount Distribution')
axes[0, 2].set_xlabel('Discount')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].axvline(df_cluster['Discount'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df_cluster["Discount"].mean():.2f}')
axes[0, 2].legend()

# Quantity distribution
axes[1, 0].hist(df_cluster['Quantity'], bins=30, color='plum', edgecolor='black')
axes[1, 0].set_title('Quantity Distribution')
axes[1, 0].set_xlabel('Quantity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(df_cluster['Quantity'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df_cluster["Quantity"].mean():.1f}')
axes[1, 0].legend()

# Profit Margin distribution
profit_margin_clean = df_cluster['Profit_Margin'].replace([np.inf, -np.inf], np.nan).dropna()
axes[1, 1].hist(profit_margin_clean, bins=50, color='gold', edgecolor='black')
axes[1, 1].set_title('Profit Margin Distribution')
axes[1, 1].set_xlabel('Profit Margin (%)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(profit_margin_clean.mean(), color='red', linestyle='--',
                   label=f'Mean: {profit_margin_clean.mean():.1f}%')
axes[1, 1].legend()

# Segment distribution
segment_counts = df_cluster['Segment'].value_counts()
axes[1, 2].bar(segment_counts.index, segment_counts.values, color=['#ff9999', '#66b3ff', '#99ff99'])
axes[1, 2].set_title('Transaction Count by Segment')
axes[1, 2].set_xlabel('Segment')
axes[1, 2].set_ylabel('Count')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('feature_distribution_clustering.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_distribution_clustering.png")
plt.show()

# ============================================================================
# 1.4 DATA SCALING
# ============================================================================
print("\n[1.4] DATA SCALING...")
print("-" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("Data has been scaled using StandardScaler")
print(f"Scaled data shape: {X_scaled.shape}")
print()

# ============================================================================
# 1.5 ELBOW METHOD - DETERMINING OPTIMAL K
# ============================================================================
print("\n[1.5] DETERMINING OPTIMAL K USING ELBOW METHOD...")
print("-" * 80)

inertias = []
silhouette_scores = []
calinski_scores = []
davies_bouldin_scores = []
K_range = range(2, 11)

print("Testing K values from 2 to 10...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    calinski_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))
    
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, "
          f"Silhouette={silhouette_scores[-1]:.4f}, "
          f"Calinski-Harabasz={calinski_scores[-1]:.2f}, "
          f"Davies-Bouldin={davies_bouldin_scores[-1]:.4f}")

print()

# Plot evaluation metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Cluster Evaluation Metrics', fontsize=16, fontweight='bold')

# Inertia (WCSS)
axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (K)')
axes[0, 0].set_ylabel('Inertia (WCSS)')
axes[0, 0].set_title('Elbow Method - Inertia')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(K_range)

# Silhouette Score (higher is better)
axes[0, 1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (K)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score (Higher = Better)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(K_range)

# Calinski-Harabasz Score (higher is better)
axes[1, 0].plot(K_range, calinski_scores, 'go-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Clusters (K)')
axes[1, 0].set_ylabel('Calinski-Harabasz Score')
axes[1, 0].set_title('Calinski-Harabasz Score (Higher = Better)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks(K_range)

# Davies-Bouldin Score (lower is better)
axes[1, 1].plot(K_range, davies_bouldin_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of Clusters (K)')
axes[1, 1].set_ylabel('Davies-Bouldin Score')
axes[1, 1].set_title('Davies-Bouldin Score (Lower = Better)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(K_range)

plt.tight_layout()
plt.savefig('elbow_method_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: elbow_method_evaluation.png")
plt.show()

# Determine optimal K
optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
optimal_k_calinski = K_range[np.argmax(calinski_scores)]
optimal_k_davies = K_range[np.argmin(davies_bouldin_scores)]

print(f"\nOptimal K recommendations:")
print(f"  - Based on Silhouette Score: K={optimal_k_silhouette}")
print(f"  - Based on Calinski-Harabasz: K={optimal_k_calinski}")
print(f"  - Based on Davies-Bouldin: K={optimal_k_davies}")

# Choose optimal K (using Silhouette as primary metric)
optimal_k = optimal_k_silhouette
print(f"\n✓ Selected optimal K: {optimal_k}")
print()

# ============================================================================
# 1.6 FINAL K-MEANS CLUSTERING
# ============================================================================
print("\n[1.6] FINAL K-MEANS CLUSTERING...")
print("-" * 80)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
df_cluster['Cluster'] = kmeans_final.fit_predict(X_scaled)

print(f"✓ K-Means clustering completed with K={optimal_k}")
print(f"  - Final Inertia: {kmeans_final.inertia_:.2f}")
print(f"  - Silhouette Score: {silhouette_score(X_scaled, df_cluster['Cluster']):.4f}")
print(f"  - Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, df_cluster['Cluster']):.2f}")
print(f"  - Davies-Bouldin Score: {davies_bouldin_score(X_scaled, df_cluster['Cluster']):.4f}")
print()

# Cluster distribution
print("Cluster Distribution:")
cluster_counts = df_cluster['Cluster'].value_counts().sort_index()
for cluster, count in cluster_counts.items():
    percentage = (count / len(df_cluster)) * 100
    print(f"  Cluster {cluster}: {count} transactions ({percentage:.2f}%)")
print()

# ============================================================================
# 1.7 CLUSTER PROFILING & CHARACTERIZATION
# ============================================================================
print("\n[1.7] CLUSTER PROFILING & CHARACTERIZATION...")
print("-" * 80)

# Calculate statistics for each cluster
cluster_profile = df_cluster.groupby('Cluster').agg({
    'Sales': ['mean', 'median', 'std', 'min', 'max'],
    'Profit': ['mean', 'median', 'std'],
    'Quantity': ['mean', 'median'],
    'Discount': ['mean', 'median'],
    'Profit_Margin': ['mean', 'median']
}).round(2)

print("\nDetailed Cluster Statistics:")
print(cluster_profile)
print()

# Characterize each cluster
cluster_characteristics = {}
for cluster in range(optimal_k):
    cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
    
    avg_sales = cluster_data['Sales'].mean()
    avg_profit = cluster_data['Profit'].mean()
    avg_discount = cluster_data['Discount'].mean()
    avg_quantity = cluster_data['Quantity'].mean()
    avg_margin = cluster_data['Profit_Margin'].mean()
    
    # Determine cluster name based on characteristics
    if avg_profit > df_cluster['Profit'].quantile(0.75) and avg_sales > df_cluster['Sales'].quantile(0.75):
        name = "High-Value Profitable"
        desc = "Transaksi dengan nilai jual tinggi dan profit besar"
        strategy = "Prioritaskan layanan premium, minimalisir diskon"
    elif avg_discount > df_cluster['Discount'].quantile(0.75):
        name = "High-Discount Transactions"
        desc = "Transaksi dengan diskon tinggi, profit cenderung rendah"
        strategy = "Review kebijakan diskon, evaluasi margin"
    elif avg_profit < df_cluster['Profit'].quantile(0.25):
        name = "Low-Profit Transactions"
        desc = "Transaksi dengan profit rendah atau rugi"
        strategy = "Analisis penyebab kerugian, optimasi pricing"
    elif avg_sales < df_cluster['Sales'].quantile(0.25):
        name = "Small-Value Transactions"
        desc = "Transaksi nilai kecil, volume rendah"
        strategy = "Bundle produk, cross-selling"
    else:
        name = "Standard Transactions"
        desc = "Transaksi normal dengan performa standar"
        strategy = "Maintain service quality, upselling"
    
    cluster_characteristics[cluster] = {
        'name': name,
        'description': desc,
        'strategy': strategy,
        'avg_sales': avg_sales,
        'avg_profit': avg_profit,
        'avg_discount': avg_discount,
        'avg_quantity': avg_quantity,
        'avg_margin': avg_margin
    }
    
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster}: {name}")
    print(f"{'='*70}")
    print(f"Deskripsi: {desc}")
    print(f"\nKarakteristik Rata-rata:")
    print(f"  • Sales: ${avg_sales:,.2f}")
    print(f"  • Profit: ${avg_profit:,.2f}")
    print(f"  • Discount: {avg_discount:.2f} ({avg_discount*100:.1f}%)")
    print(f"  • Quantity: {avg_quantity:.1f} items")
    print(f"  • Profit Margin: {avg_margin:.2f}%")
    print(f"\nStrategi Bisnis:")
    print(f"  ➤ {strategy}")
    
    # Top categories and segments in this cluster
    top_categories = cluster_data['Category'].value_counts().head(3)
    top_segments = cluster_data['Segment'].value_counts().head(3)
    
    print(f"\nTop 3 Categories:")
    for cat, count in top_categories.items():
        pct = (count/len(cluster_data))*100
        print(f"  • {cat}: {count} ({pct:.1f}%)")
    
    print(f"\nTop 3 Segments:")
    for seg, count in top_segments.items():
        pct = (count/len(cluster_data))*100
        print(f"  • {seg}: {count} ({pct:.1f}%)")

# Add cluster names to dataframe
df_cluster['Cluster_Name'] = df_cluster['Cluster'].map(lambda x: cluster_characteristics[x]['name'])

# ============================================================================
# 1.8 COMPREHENSIVE CLUSTER VISUALIZATION
# ============================================================================
print("\n\n[1.8] COMPREHENSIVE CLUSTER VISUALIZATION...")
print("-" * 80)

fig = plt.figure(figsize=(20, 14))

# 1. 3D Scatter: Sales vs Profit vs Discount
ax1 = fig.add_subplot(3, 3, 1, projection='3d')
scatter = ax1.scatter(df_cluster['Sales'], df_cluster['Profit'], df_cluster['Discount'],
                     c=df_cluster['Cluster'], cmap='viridis', s=30, alpha=0.6)
ax1.set_xlabel('Sales ($)')
ax1.set_ylabel('Profit ($)')
ax1.set_zlabel('Discount')
ax1.set_title('3D: Sales vs Profit vs Discount', fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Cluster', shrink=0.5)

# 2. Sales vs Profit
ax2 = fig.add_subplot(3, 3, 2)
for cluster in range(optimal_k):
    cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
    ax2.scatter(cluster_data['Sales'], cluster_data['Profit'],
               label=f'C{cluster}: {cluster_characteristics[cluster]["name"]}', alpha=0.6, s=30)
ax2.set_xlabel('Sales ($)')
ax2.set_ylabel('Profit ($)')
ax2.set_title('Sales vs Profit by Cluster', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# 3. Discount vs Profit
ax3 = fig.add_subplot(3, 3, 3)
for cluster in range(optimal_k):
    cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
    ax3.scatter(cluster_data['Discount'], cluster_data['Profit'],
               label=f'Cluster {cluster}', alpha=0.6, s=30)
ax3.set_xlabel('Discount')
ax3.set_ylabel('Profit ($)')
ax3.set_title('Discount vs Profit', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Cluster Distribution Pie Chart
ax4 = fig.add_subplot(3, 3, 4)
cluster_sizes = df_cluster['Cluster'].value_counts().sort_index()
colors = plt.cm.viridis(np.linspace(0, 1, optimal_k))
wedges, texts, autotexts = ax4.pie(cluster_sizes, labels=[f'C{i}' for i in range(optimal_k)],
                                    autopct='%1.1f%%', startangle=90, colors=colors)
ax4.set_title('Cluster Distribution', fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 5. Average metrics by cluster (normalized)
ax5 = fig.add_subplot(3, 3, 5)
metrics = ['Sales', 'Profit', 'Discount', 'Quantity']
cluster_means = df_cluster.groupby('Cluster')[metrics].mean()
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
x = np.arange(len(cluster_means_norm))
width = 0.2
for i, metric in enumerate(metrics):
    ax5.bar(x + i*width, cluster_means_norm[metric], width, label=metric, alpha=0.8)
ax5.set_xlabel('Cluster')
ax5.set_ylabel('Normalized Value')
ax5.set_title('Average Metrics by Cluster (Normalized)', fontweight='bold')
ax5.set_xticks(x + width * 1.5)
ax5.set_xticklabels([f'C{i}' for i in range(optimal_k)])
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3, axis='y')

# 6. Profit Margin distribution by cluster
ax6 = fig.add_subplot(3, 3, 6)
profit_margin_data = [df_cluster[df_cluster['Cluster']==c]['Profit_Margin'].dropna() 
                      for c in range(optimal_k)]
bp = ax6.boxplot(profit_margin_data, labels=[f'C{i}' for i in range(optimal_k)], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax6.set_xlabel('Cluster')
ax6.set_ylabel('Profit Margin (%)')
ax6.set_title('Profit Margin Distribution by Cluster', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Category distribution by cluster
ax7 = fig.add_subplot(3, 3, 7)
category_cluster = pd.crosstab(df_cluster['Category'], df_cluster['Cluster'], normalize='columns') * 100
category_cluster.plot(kind='bar', ax=ax7, stacked=True, colormap='viridis', alpha=0.8)
ax7.set_xlabel('Category')
ax7.set_ylabel('Percentage (%)')
ax7.set_title('Category Distribution by Cluster', fontweight='bold')
ax7.legend(title='Cluster', labels=[f'C{i}' for i in range(optimal_k)], fontsize=8)
ax7.tick_params(axis='x', rotation=45)
ax7.grid(True, alpha=0.3, axis='y')

# 8. Segment distribution by cluster
ax8 = fig.add_subplot(3, 3, 8)
segment_cluster = pd.crosstab(df_cluster['Segment'], df_cluster['Cluster'], normalize='columns') * 100
segment_cluster.plot(kind='bar', ax=ax8, stacked=True, colormap='plasma', alpha=0.8)
ax8.set_xlabel('Segment')
ax8.set_ylabel('Percentage (%)')
ax8.set_title('Segment Distribution by Cluster', fontweight='bold')
ax8.legend(title='Cluster', labels=[f'C{i}' for i in range(optimal_k)], fontsize=8)
ax8.tick_params(axis='x', rotation=45)
ax8.grid(True, alpha=0.3, axis='y')

# 9. Heatmap of cluster centroids
ax9 = fig.add_subplot(3, 3, 9)
centroids_original = scaler.inverse_transform(kmeans_final.cluster_centers_)
centroids_df = pd.DataFrame(centroids_original, columns=clustering_features)
# Select key features for heatmap
key_features = ['Sales', 'Profit', 'Quantity', 'Discount', 'Profit_Margin', 'Price_Per_Item']
centroids_key = centroids_df[key_features]
centroids_norm = (centroids_key - centroids_key.min()) / (centroids_key.max() - centroids_key.min())
sns.heatmap(centroids_norm.T, annot=True, fmt='.2f', cmap='YlOrRd', 
            cbar_kws={'label': 'Normalized Value'}, ax=ax9)
ax9.set_xlabel('Cluster')
ax9.set_ylabel('Feature')
ax9.set_title('Cluster Centroids Heatmap', fontweight='bold')
ax9.set_xticklabels([f'C{i}' for i in range(optimal_k)])

plt.tight_layout()
plt.savefig('transaction_cluster_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: transaction_cluster_visualization.png")
plt.show()

# ============================================================================
# 1.9 BUSINESS INSIGHTS FROM CLUSTERING
# ============================================================================
print("\n\n[1.9] BUSINESS INSIGHTS FROM CLUSTERING...")
print("-" * 80)

print("\n" + "="*70)
print("REKOMENDASI STRATEGIS BERDASARKAN CLUSTER")
print("="*70)

for cluster in range(optimal_k):
    char = cluster_characteristics[cluster]
    cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
    count = len(cluster_data)
    pct = (count/len(df_cluster))*100
    total_sales = cluster_data['Sales'].sum()
    total_profit = cluster_data['Profit'].sum()
    
    print(f"\n{cluster+1}. CLUSTER {cluster}: {char['name']}")
    print(f"   {'─'*65}")
    print(f"   Jumlah Transaksi: {count:,} ({pct:.1f}% dari total)")
    print(f"   Total Sales: ${total_sales:,.2f}")
    print(f"   Total Profit: ${total_profit:,.2f}")
    print(f"   \n   ✓ Karakteristik: {char['description']}")
    print(f"   ✓ Strategi: {char['strategy']}")

# Save clustering results
df_cluster.to_csv('transaction_clustering_results.csv', index=False)
print("\n✓ Saved: transaction_clustering_results.csv")

print("\n" + "="*80)
print("PART 1 COMPLETED: Transaction Pattern Clustering")
print("="*80)

# ============================================================================
# PART 2: PROFITABILITY CLASSIFICATION USING DECISION TREE
# ============================================================================
print("\n\n" + "="*80)
print("PART 2: PROFITABILITY CLASSIFICATION (DECISION TREE)")
print("="*80)

# ============================================================================
# 2.1 FEATURE ENGINEERING FOR CLASSIFICATION
# ============================================================================
print("\n[2.1] FEATURE ENGINEERING FOR CLASSIFICATION...")
print("-" * 80)

# Use original dataframe
df_class = df.copy()

# Create target variable
df_class['IsProfitable'] = (df_class['Profit'] > 0).astype(int)
df_class['IsProfitable_Label'] = df_class['IsProfitable'].map({1: 'Untung', 0: 'Rugi'})

print(f"Target variable created: IsProfitable")
print(f"  - Untung (Profit > 0): {(df_class['IsProfitable'] == 1).sum()} transactions ({(df_class['IsProfitable'] == 1).sum()/len(df_class)*100:.2f}%)")
print(f"  - Rugi (Profit <= 0): {(df_class['IsProfitable'] == 0).sum()} transactions ({(df_class['IsProfitable'] == 0).sum()/len(df_class)*100:.2f}%)")
print()

# Select features
feature_columns = ['Segment', 'Region', 'Category', 'Sub-Category', 
                   'Sales', 'Quantity', 'Discount']

available_features = [col for col in feature_columns if col in df_class.columns]
print(f"Selected features: {available_features}")
print()

# ============================================================================
# 2.2 DATA PREPROCESSING
# ============================================================================
print("\n[2.2] DATA PREPROCESSING...")
print("-" * 80)

df_classification = df_class[available_features + ['IsProfitable', 'IsProfitable_Label', 'Profit']].copy()

# Handle missing values
print("Checking for missing values...")
missing_values = df_classification.isnull().sum()
if missing_values.sum() > 0:
    print("Missing values found:")
    print(missing_values[missing_values > 0])
    df_classification = df_classification.dropna()
    print(f"Rows after dropping missing values: {len(df_classification)}")
else:
    print("✓ No missing values found")
print()

# Encode categorical variables
print("Encoding categorical variables...")
le_dict = {}
categorical_features = df_classification.select_dtypes(include=['object']).columns
categorical_features = [col for col in categorical_features if col not in ['IsProfitable_Label']]

for col in categorical_features:
    le = LabelEncoder()
    df_classification[col + '_Encoded'] = le.fit_transform(df_classification[col])
    le_dict[col] = le
    print(f"  ✓ {col}: {len(le.classes_)} unique values encoded")

print()

# Prepare X and y
encoded_features = [col for col in available_features if col not in categorical_features] + \
                   [col + '_Encoded' for col in categorical_features if col in available_features]

X = df_classification[encoded_features]
y = df_classification['IsProfitable']

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Features used: {list(X.columns)}")
print()

# ============================================================================
# 2.3 TRAIN-TEST SPLIT
# ============================================================================
print("\n[2.3] TRAIN-TEST SPLIT (80/20)...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print()
print("Class distribution in training set:")
train_dist = y_train.value_counts()
for label, count in train_dist.items():
    label_name = 'Untung' if label == 1 else 'Rugi'
    print(f"  {label_name}: {count} ({count/len(y_train)*100:.2f}%)")
print()
print("Class distribution in test set:")
test_dist = y_test.value_counts()
for label, count in test_dist.items():
    label_name = 'Untung' if label == 1 else 'Rugi'
    print(f"  {label_name}: {count} ({count/len(y_test)*100:.2f}%)")
print()

# ============================================================================
# 2.4 DECISION TREE MODELING
# ============================================================================
print("\n[2.4] BUILDING DECISION TREE MODEL...")
print("-" * 80)

# Build model
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=42,
    criterion='entropy'
)

print("Model parameters:")
print(f"  - max_depth: 5")
print(f"  - min_samples_split: 100")
print(f"  - min_samples_leaf: 50")
print(f"  - criterion: entropy")
print()

print("Training Decision Tree model...")
dt_model.fit(X_train, y_train)
print("✓ Model training completed")
print()

# Make predictions
y_pred_train = dt_model.predict(X_train)
y_pred_test = dt_model.predict(X_test)

# ============================================================================
# 2.5 MODEL EVALUATION
# ============================================================================
print("\n[2.5] MODEL EVALUATION...")
print("-" * 80)

# Training performance
train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train, zero_division=0)
train_recall = recall_score(y_train, y_pred_train, zero_division=0)
train_f1 = f1_score(y_train, y_pred_train, zero_division=0)

print("TRAINING SET PERFORMANCE:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")
print()

# Test performance
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, zero_division=0)
test_recall = recall_score(y_test, y_pred_test, zero_division=0)
test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

print("TEST SET PERFORMANCE:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("CONFUSION MATRIX:")
print(f"                Predicted")
print(f"              Rugi  Untung")
print(f"Actual Rugi    {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"       Untung  {cm[1,0]:4d}   {cm[1,1]:4d}")
print()

# Detailed report
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_test, 
                          target_names=['Rugi', 'Untung'], zero_division=0))

# ============================================================================
# 2.6 VISUALIZATION: MODEL PERFORMANCE
# ============================================================================
print("\n[2.6] VISUALIZING MODEL PERFORMANCE...")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Rugi', 'Untung'],
            yticklabels=['Rugi', 'Untung'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Label', fontweight='bold')
axes[0].set_ylabel('True Label', fontweight='bold')
axes[0].set_title('Confusion Matrix', fontweight='bold', fontsize=14)

# Add percentage annotations
for i in range(2):
    for j in range(2):
        pct = cm[i,j] / cm.sum() * 100
        axes[0].text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')

# Performance metrics comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_scores = [train_accuracy, train_precision, train_recall, train_f1]
test_scores = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[1].bar(x - width/2, train_scores, width, label='Training', alpha=0.8, color='skyblue')
bars2 = axes[1].bar(x + width/2, test_scores, width, label='Testing', alpha=0.8, color='lightcoral')

axes[1].set_xlabel('Metrics', fontweight='bold')
axes[1].set_ylabel('Score', fontweight='bold')
axes[1].set_title('Model Performance Metrics', fontweight='bold', fontsize=14)
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics)
axes[1].legend()
axes[1].set_ylim([0, 1.1])
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('decision_tree_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: decision_tree_performance.png")
plt.show()

# ============================================================================
# 2.7 FEATURE IMPORTANCE
# ============================================================================
print("\n[2.7] FEATURE IMPORTANCE ANALYSIS...")
print("-" * 80)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance Ranking:")
print(feature_importance)
print()

print("Top 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")
print()

# Visualize
plt.figure(figsize=(12, 6))
top_n = min(10, len(feature_importance))
colors = plt.cm.viridis(feature_importance.head(top_n)['Importance'] / feature_importance['Importance'].max())
bars = plt.barh(range(top_n), feature_importance.head(top_n)['Importance'], color=colors)
plt.yticks(range(top_n), feature_importance.head(top_n)['Feature'])
plt.xlabel('Importance Score', fontweight='bold')
plt.ylabel('Feature', fontweight='bold')
plt.title(f'Top {top_n} Feature Importance', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (idx, row) in enumerate(feature_importance.head(top_n).iterrows()):
    plt.text(row['Importance'] + 0.005, i, f"{row['Importance']:.4f}", 
            va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.show()

# ============================================================================
# 2.8 DECISION TREE VISUALIZATION
# ============================================================================
print("\n[2.8] VISUALIZING DECISION TREE STRUCTURE...")
print("-" * 80)

plt.figure(figsize=(25, 15))
plot_tree(dt_model, 
          feature_names=X.columns,
          class_names=['Rugi', 'Untung'],
          filled=True,
          rounded=True,
          fontsize=10,
          proportion=True,
          impurity=True)
plt.title('Decision Tree Structure - Profitability Classification', 
         fontweight='bold', fontsize=18, pad=20)
plt.tight_layout()
plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
print("✓ Saved: decision_tree_structure.png")
plt.show()

# ============================================================================
# 2.9 BUSINESS INSIGHTS
# ============================================================================
print("\n[2.9] EXTRACTING BUSINESS INSIGHTS...")
print("-" * 80)

# Add predictions to dataframe
df_classification['Predicted'] = dt_model.predict(X)
df_classification['Correct'] = (df_classification['Predicted'] == df_classification['IsProfitable'])

# Insight 1: Discount Impact
print("\n1. IMPACT OF DISCOUNT ON PROFITABILITY")
print("   " + "─"*65)
discount_ranges = pd.cut(df_classification['Discount'], 
                         bins=[-0.01, 0.01, 0.1, 0.2, 0.3, 1.0],
                         labels=['No Discount', '0-10%', '10-20%', '20-30%', '>30%'])
discount_profit = df_classification.groupby(discount_ranges)['IsProfitable'].agg(['mean', 'count'])
discount_profit.columns = ['Profit_Rate', 'Count']
discount_profit['Profit_Rate_Pct'] = discount_profit['Profit_Rate'] * 100

print(discount_profit)
print()

# Insight 2: Category Performance
print("2. PROFITABILITY BY CATEGORY")
print("   " + "─"*65)
category_profit = df_classification.groupby('Category').agg({
    'IsProfitable': 'mean',
    'Profit': ['mean', 'sum', 'count']
}).round(2)
category_profit.columns = ['Profit_Rate', 'Avg_Profit', 'Total_Profit', 'Trans_Count']
category_profit['Profit_Rate_Pct'] = category_profit['Profit_Rate'] * 100
category_profit = category_profit.sort_values('Profit_Rate_Pct', ascending=False)
print(category_profit)
print()

# Insight 3: Segment Performance
print("3. PROFITABILITY BY SEGMENT")
print("   " + "─"*65)
segment_profit = df_classification.groupby('Segment').agg({
    'IsProfitable': 'mean',
    'Profit': ['mean', 'sum', 'count']
}).round(2)
segment_profit.columns = ['Profit_Rate', 'Avg_Profit', 'Total_Profit', 'Trans_Count']
segment_profit['Profit_Rate_Pct'] = segment_profit['Profit_Rate'] * 100
segment_profit = segment_profit.sort_values('Profit_Rate_Pct', ascending=False)
print(segment_profit)
print()

# Insight 4: Misclassification Analysis
print("4. MISCLASSIFICATION ANALYSIS")
print("   " + "─"*65)
misclassified = df_classification[~df_classification['Correct']]
print(f"Total misclassified: {len(misclassified)} ({len(misclassified)/len(df_classification)*100:.2f}%)")
print()
print("Misclassification by Category:")
misclass_category = misclassified['Category'].value_counts()
for cat, count in misclass_category.items():
    pct = (count/len(misclassified))*100
    print(f"  • {cat}: {count} ({pct:.1f}%)")
print()

# ============================================================================
# 2.10 BUSINESS INSIGHTS VISUALIZATION
# ============================================================================
print("\n[2.10] VISUALIZING BUSINESS INSIGHTS...")
print("-" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Discount vs Profitability
ax1 = axes[0, 0]
bars = ax1.bar(range(len(discount_profit)), discount_profit['Profit_Rate_Pct'], 
              color='coral', alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(discount_profit)))
ax1.set_xticklabels(discount_profit.index, rotation=45, ha='right')
ax1.set_xlabel('Discount Range', fontweight='bold')
ax1.set_ylabel('Profitability Rate (%)', fontweight='bold')
ax1.set_title('Profitability Rate by Discount Range', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
for i, (idx, row) in enumerate(discount_profit.iterrows()):
    ax1.text(i, row['Profit_Rate_Pct'] + 1, 
            f"{row['Profit_Rate_Pct']:.1f}%\n(n={row['Count']})", 
            ha='center', fontsize=9, fontweight='bold')

# 2. Category Profitability
ax2 = axes[0, 1]
bars = ax2.barh(range(len(category_profit)), category_profit['Profit_Rate_Pct'], 
               color='lightblue', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(category_profit)))
ax2.set_yticklabels(category_profit.index)
ax2.set_xlabel('Profitability Rate (%)', fontweight='bold')
ax2.set_ylabel('Category', fontweight='bold')
ax2.set_title('Profitability Rate by Category', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(category_profit.iterrows()):
    ax2.text(row['Profit_Rate_Pct'] + 1, i, f"{row['Profit_Rate_Pct']:.1f}%", 
            va='center', fontsize=9, fontweight='bold')

# 3. Segment Profitability
ax3 = axes[1, 0]
bars = ax3.barh(range(len(segment_profit)), segment_profit['Profit_Rate_Pct'],
               color='lightgreen', alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(segment_profit)))
ax3.set_yticklabels(segment_profit.index)
ax3.set_xlabel('Profitability Rate (%)', fontweight='bold')
ax3.set_ylabel('Segment', fontweight='bold')
ax3.set_title('Profitability Rate by Segment', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3, axis='x')
for i, (idx, row) in enumerate(segment_profit.iterrows()):
    ax3.text(row['Profit_Rate_Pct'] + 1, i, f"{row['Profit_Rate_Pct']:.1f}%", 
            va='center', fontsize=9, fontweight='bold')

# 4. Prediction Distribution
ax4 = axes[1, 1]
actual_counts = df_classification['IsProfitable'].value_counts().sort_index()
predicted_counts = df_classification['Predicted'].value_counts().sort_index()
x_pos = np.arange(2)
width = 0.35
bars1 = ax4.bar(x_pos - width/2, actual_counts.values, width, label='Actual', alpha=0.8, color='steelblue')
bars2 = ax4.bar(x_pos + width/2, predicted_counts.values, width, label='Predicted', alpha=0.8, color='orange')
ax4.set_xlabel('Class', fontweight='bold')
ax4.set_ylabel('Count', fontweight='bold')
ax4.set_title('Actual vs Predicted Distribution', fontweight='bold', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['Rugi', 'Untung'])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('business_insights_classification.png', dpi=300, bbox_inches='tight')
print("✓ Saved: business_insights_classification.png")
plt.show()

# Save results
df_classification.to_csv('profitability_classification_results.csv', index=False)
print("\n✓ Saved: profitability_classification_results.csv")

print("\n" + "="*80)
print("PART 2 COMPLETED: Profitability Classification")
print("="*80)

# ============================================================================
# FINAL COMPREHENSIVE REPORT
# ============================================================================
print("\n\n" + "="*80)
print("FINAL COMPREHENSIVE REPORT")
print("="*80)

print("\n" + "┌" + "─"*78 + "┐")
print("│" + " "*20 + "PART 1: TRANSACTION CLUSTERING" + " "*28 + "│")
print("└" + "─"*78 + "┘")

print(f"\n✓ Analysis Method: K-Means Clustering on Transaction Patterns")
print(f"✓ Total Transactions Analyzed: {len(df_cluster):,}")
print(f"✓ Number of Clusters: {optimal_k}")
print(f"✓ Clustering Quality Metrics:")
print(f"    • Silhouette Score: {silhouette_score(X_scaled, df_cluster['Cluster']):.4f}")
print(f"    • Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, df_cluster['Cluster']):.2f}")
print(f"    • Davies-Bouldin Score: {davies_bouldin_score(X_scaled, df_cluster['Cluster']):.4f}")

print(f"\n📊 Cluster Summary:")
for cluster in range(optimal_k):
    char = cluster_characteristics[cluster]
    count = len(df_cluster[df_cluster['Cluster'] == cluster])
    pct = (count/len(df_cluster))*100
    total_profit = df_cluster[df_cluster['Cluster'] == cluster]['Profit'].sum()
    
    print(f"\n  {cluster+1}. {char['name']}")
    print(f"     • Transactions: {count:,} ({pct:.1f}%)")
    print(f"     • Avg Sales: ${char['avg_sales']:,.2f}")
    print(f"     • Avg Profit: ${char['avg_profit']:,.2f}")
    print(f"     • Total Profit: ${total_profit:,.2f}")
    print(f"     • Strategy: {char['strategy']}")

print("\n" + "┌" + "─"*78 + "┐")
print("│" + " "*18 + "PART 2: PROFITABILITY CLASSIFICATION" + " "*24 + "│")
print("└" + "─"*78 + "┘")

print(f"\n✓ Model: Decision Tree (CART with Entropy)")
print(f"✓ Total Transactions: {len(df_classification):,}")
print(f"✓ Training Samples: {len(X_train):,} | Test Samples: {len(X_test):,}")
print(f"\n📊 Model Performance (Test Set):")
print(f"    • Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"    • Precision: {test_precision:.4f}")
print(f"    • Recall: {test_recall:.4f}")
print(f"    • F1-Score: {test_f1:.4f}")

print(f"\n🔑 Top 3 Important Features:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"    {idx+1}. {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")

print("\n" + "┌" + "─"*78 + "┐")
print("│" + " "*22 + "KEY BUSINESS RECOMMENDATIONS" + " "*27 + "│")
print("└" + "─"*78 + "┘")

print("\n📈 FROM CLUSTERING ANALYSIS:")
print("   1. Focus on High-Value Profitable transactions - maintain quality")
print("   2. Review discount policies for High-Discount cluster")
print("   3. Investigate causes of Low-Profit transactions")
print("   4. Optimize pricing strategy for Small-Value transactions")

print("\n📊 FROM CLASSIFICATION ANALYSIS:")
print("   1. HIGH DISCOUNT = HIGH RISK:")
print("      • Discounts >30% significantly reduce profitability")
print(f"      • Recommendation: Cap discounts at 20% for most categories")
print()
print("   2. CATEGORY-SPECIFIC STRATEGIES:")
most_profitable_cat = category_profit.index[0]
least_profitable_cat = category_profit.index[-1]
print(f"      • {most_profitable_cat}: Most profitable ({category_profit.loc[most_profitable_cat, 'Profit_Rate_Pct']:.1f}%)")
print(f"      • {least_profitable_cat}: Needs review ({category_profit.loc[least_profitable_cat, 'Profit_Rate_Pct']:.1f}%)")
print()
print("   3. SEGMENT-BASED APPROACH:")
best_segment = segment_profit.index[0]
print(f"      • Focus marketing on {best_segment} segment")
print(f"      • Profitability rate: {segment_profit.loc[best_segment, 'Profit_Rate_Pct']:.1f}%")

print("\n" + "┌" + "─"*78 + "┐")
print("│" + " "*28 + "OUTPUT FILES GENERATED" + " "*29 + "│")
print("└" + "─"*78 + "┘")

output_files = [
    "1. feature_distribution_clustering.png - Feature distributions",
    "2. elbow_method_evaluation.png - K optimization metrics",
    "3. transaction_cluster_visualization.png - Comprehensive cluster analysis",
    "4. transaction_clustering_results.csv - Clustering data",
    "5. decision_tree_performance.png - Model performance",
    "6. feature_importance.png - Important features",
    "7. decision_tree_structure.png - Tree visualization",
    "8. business_insights_classification.png - Business insights",
    "9. profitability_classification_results.csv - Classification data"
]

for file in output_files:
    print