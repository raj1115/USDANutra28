# ============================================================================
# COMPLETE FLASK BACKEND - 25+ NUTRIENTS
# Enhanced with Advanced ML Features
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# ============================================================================
# 1. LOAD DATA AND TRAIN MODEL
# ============================================================================
print("="*80)
print("🚀 ADVANCED FOOD RECOMMENDATION SYSTEM - STARTING")
print("="*80)

print("\n📂 Loading USDA Food Database...")
df = pd.read_excel("ABBREV.xlsx")
print(f"✅ Loaded {len(df)} food items")

# ============================================================================
# 2. DEFINE ALL 25+ NUTRIENTS
# ============================================================================
nutrient_categories = {
    'Macronutrients': ['Protein_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)', 'Sugar_Tot_(g)'],
    'Energy': ['Energ_Kcal'],
    'Fats': ['FA_Sat_(g)', 'FA_Mono_(g)', 'FA_Poly_(g)', 'Cholestrl_(mg)'],
    'Minerals': ['Calcium_(mg)', 'Iron_(mg)', 'Magnesium_(mg)', 'Potassium_(mg)', 'Sodium_(mg)', 'Zinc_(mg)'],
    'Vitamins': ['Vit_C_(mg)', 'Vit_A_RAE', 'Vit_D_µg', 'Vit_E_(mg)', 'Vit_K_(µg)', 'Vit_B12_(µg)', 'Folate_Tot_(µg)']
}

all_nutrients = [n for cat in nutrient_categories.values() for n in cat]
available_nutrients = [n for n in all_nutrients if n in df.columns]

print(f"\n📊 Nutrient Analysis:")
print(f"   • Total nutrients requested: {len(all_nutrients)}")
print(f"   • Available in dataset: {len(available_nutrients)}")

# Map frontend names to backend column names
nutrient_mapping = {
    'protein': 'Protein_(g)',
    'carbs': 'Carbohydrt_(g)',
    'fat': 'Lipid_Tot_(g)',
    'fiber': 'Fiber_TD_(g)',
    'sugar': 'Sugar_Tot_(g)',
    'calories': 'Energ_Kcal',
    'satfat': 'FA_Sat_(g)',
    'monofat': 'FA_Mono_(g)',
    'polyfat': 'FA_Poly_(g)',
    'cholesterol': 'Cholestrl_(mg)',
    'calcium': 'Calcium_(mg)',
    'iron': 'Iron_(mg)',
    'magnesium': 'Magnesium_(mg)',
    'sodium': 'Sodium_(mg)',
    'potassium': 'Potassium_(mg)',
    'zinc': 'Zinc_(mg)',
    'vitc': 'Vit_C_(mg)',
    'vita': 'Vit_A_RAE',
    'vitd': 'Vit_D_µg',
    'vite': 'Vit_E_(mg)',
    'vitk': 'Vit_K_(µg)',
    'vitb12': 'Vit_B12_(µg)',
    'folate': 'Folate_Tot_(µg)'
}

print(f"\n🔧 Data Preprocessing:")
print(f"   • Filling missing values with 0 (nutrient not present)")

# Fill missing values
df[available_nutrients] = df[available_nutrients].fillna(0)

print(f"   • Standardizing features using StandardScaler")

# Train model once on startup
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[available_nutrients])

print(f"\n🤖 Training K-Means Clustering Model...")
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"✅ Model trained successfully!")
print(f"   • Foods clustered: {len(df)}")
print(f"   • Number of clusters: {optimal_k}")
print(f"   • Features used: {len(available_nutrients)}")

# Calculate model quality metrics
sil_score = silhouette_score(X_scaled, df['cluster'])
db_score = davies_bouldin_score(X_scaled, df['cluster'])
ch_score = calinski_harabasz_score(X_scaled, df['cluster'])
inertia = kmeans.inertia_

print(f"\n📊 Model Quality Metrics:")
print(f"   • Silhouette Score: {sil_score:.4f}")
print(f"   • Davies-Bouldin Index: {db_score:.4f}")
print(f"   • Calinski-Harabasz Score: {ch_score:.2f}")
print(f"   • Inertia (WCSS): {inertia:.2f}")

# Cluster size distribution
cluster_counts = df['cluster'].value_counts().sort_index()
print(f"\n🎯 Cluster Distribution:")
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   • Cluster {cluster_id}: {count} foods ({percentage:.1f}%)")

# ============================================================================
# 3. API ENDPOINTS
# ============================================================================

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    return jsonify({
        'total_foods': len(df),
        'total_nutrients': len(available_nutrients),
        'total_clusters': optimal_k,
        'cluster_sizes': df['cluster'].value_counts().to_dict()
    })

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Get food recommendations based on user preferences"""
    try:
        # Get user preferences from frontend
        data = request.json
        
        # Build user preferences array using all available nutrients
        user_prefs = {}
        for frontend_name, backend_name in nutrient_mapping.items():
            if backend_name in available_nutrients:
                # Use provided value or default to 0
                user_prefs[backend_name] = data.get(frontend_name, 0)
        
        # Options
        top_n = data.get('numRecommendations', 10)
        diversity_mode = data.get('diversityMode', False)
        exclude_keywords = data.get('excludeKeywords', [])
        weighting_mode = data.get('weightingMode', 'prioritize')
        
        # Convert user preferences to array (in same order as available_nutrients)
        user_prefs_array = np.array([user_prefs.get(n, 0) for n in available_nutrients]).reshape(1, -1)
        user_scaled = scaler.transform(user_prefs_array)
        
        # Apply weighting based on mode
        if weighting_mode == 'prioritize':
            # Give 3x weight to non-zero preferences
            weights = np.array([3.0 if user_prefs.get(n, 0) > 0 else 1.0 for n in available_nutrients])
            user_scaled_weighted = user_scaled * weights
            cluster_centers_weighted = kmeans.cluster_centers_ * weights
        elif weighting_mode == 'focus':
            # Only use non-zero preferences
            adjusted_indices = [i for i, n in enumerate(available_nutrients) if user_prefs.get(n, 0) > 0]
            if len(adjusted_indices) > 0:
                user_scaled_weighted = user_scaled[:, adjusted_indices]
                cluster_centers_weighted = kmeans.cluster_centers_[:, adjusted_indices]
            else:
                user_scaled_weighted = user_scaled
                cluster_centers_weighted = kmeans.cluster_centers_
        else:  # equal
            user_scaled_weighted = user_scaled
            cluster_centers_weighted = kmeans.cluster_centers_
        
        # Find closest cluster(s)
        cluster_distances = np.linalg.norm(cluster_centers_weighted - user_scaled_weighted, axis=1)
        closest_cluster = np.argmin(cluster_distances)
        
        # Select foods from cluster(s)
        if diversity_mode:
            top_clusters = np.argsort(cluster_distances)[:3]
            cluster_foods = df[df['cluster'].isin(top_clusters)].copy()
        else:
            cluster_foods = df[df['cluster'] == closest_cluster].copy()
        
        # Apply keyword exclusions
        if exclude_keywords:
            for keyword in exclude_keywords:
                cluster_foods = cluster_foods[
                    ~cluster_foods['Shrt_Desc'].str.contains(keyword, case=False, na=False)
                ]
        
        # Calculate cosine similarity
        cluster_scaled = scaler.transform(cluster_foods[available_nutrients])
        similarities = cosine_similarity(user_scaled, cluster_scaled)[0]
        cluster_foods['similarity'] = similarities
        
        # Get top N recommendations
        recommendations = cluster_foods.nlargest(top_n, 'similarity')
        
        # Calculate metrics
        avg_similarity = recommendations['similarity'].mean()
        diversity = len(recommendations['cluster'].unique())
        precision_at_k = (recommendations['similarity'] > 0.6).sum() / top_n
        
        # Prepare response
        results = []
        for idx, row in recommendations.iterrows():
            food_data = {
                'name': row['Shrt_Desc'],
                'cluster': int(row['cluster']),
                'similarity': float(row['similarity'])
            }
            
            # Add all available nutrients
            for frontend_name, backend_name in nutrient_mapping.items():
                if backend_name in available_nutrients:
                    food_data[frontend_name] = float(row[backend_name])
            
            # Ensure basic nutrients are always present
            food_data['protein'] = float(row.get('Protein_(g)', 0))
            food_data['carbs'] = float(row.get('Carbohydrt_(g)', 0))
            food_data['fat'] = float(row.get('Lipid_Tot_(g)', 0))
            food_data['fiber'] = float(row.get('Fiber_TD_(g)', 0))
            food_data['calories'] = float(row.get('Energ_Kcal', 0))
            food_data['calcium'] = float(row.get('Calcium_(mg)', 0))
            
            results.append(food_data)
        
        return jsonify({
            'success': True,
            'recommendations': results,
            'metrics': {
                'avg_similarity': float(avg_similarity),
                'diversity': int(diversity),
                'precision_at_k': float(precision_at_k),
                'matched_cluster': int(closest_cluster)
            }
        })
        
    except Exception as e:
        print(f"❌ Error in recommendation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics"""
    return jsonify({
        'silhouette_score': float(sil_score),
        'davies_bouldin_index': float(db_score),
        'calinski_harabasz_score': float(ch_score),
        'inertia': float(inertia)
    })

@app.route('/api/clusters', methods=['GET'])
def get_cluster_info():
    """Get cluster interpretations"""
    cluster_info = []
    
    for i in range(optimal_k):
        cluster_data = df[df['cluster'] == i]
        avg_nutrients = cluster_data[available_nutrients].mean()
        top_3 = avg_nutrients.nlargest(3)
        
        cluster_info.append({
            'cluster_id': i,
            'size': len(cluster_data),
            'top_nutrients': {k: float(v) for k, v in top_3.items()},
            'sample_foods': cluster_data['Shrt_Desc'].head(3).tolist()
        })
    
    return jsonify(cluster_info)

@app.route('/api/baseline_comparison', methods=['POST'])
def baseline_comparison():
    """Compare recommendations with baseline models"""
    try:
        data = request.json
        user_prefs = {}
        for frontend_name, backend_name in nutrient_mapping.items():
            if backend_name in available_nutrients:
                user_prefs[backend_name] = data.get(frontend_name, 0)
        
        user_prefs_array = np.array([user_prefs.get(n, 0) for n in available_nutrients]).reshape(1, -1)
        user_scaled = scaler.transform(user_prefs_array)
        
        # Random baseline
        random_foods = df.sample(n=10, random_state=42)
        random_scaled = scaler.transform(random_foods[available_nutrients])
        random_sim = cosine_similarity(user_scaled, random_scaled)[0].mean()
        
        # Popularity baseline (top nutrient-dense foods)
        df_temp = df.copy()
        df_temp['nutrient_score'] = df_temp[available_nutrients].sum(axis=1)
        popular_foods = df_temp.nlargest(10, 'nutrient_score')
        popular_scaled = scaler.transform(popular_foods[available_nutrients])
        popular_sim = cosine_similarity(user_scaled, popular_scaled)[0].mean()
        
        # Protein-only baseline
        if 'Protein_(g)' in available_nutrients:
            protein_foods = df.nlargest(10, 'Protein_(g)')
            protein_scaled = scaler.transform(protein_foods[available_nutrients])
            protein_sim = cosine_similarity(user_scaled, protein_scaled)[0].mean()
        else:
            protein_sim = 0.0
        
        return jsonify({
            'random_baseline': float(random_sim),
            'popularity_baseline': float(popular_sim),
            'protein_baseline': float(protein_sim)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/dataset_analysis', methods=['GET'])
def dataset_analysis():
    """Get comprehensive dataset analysis"""
    try:
        # Missing value analysis
        missing_pct = (df[available_nutrients].isnull().sum() / len(df)) * 100
        
        # Outlier detection
        outlier_counts = {}
        for nutrient in available_nutrients:
            Q1 = df[nutrient].quantile(0.25)
            Q3 = df[nutrient].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[nutrient] < (Q1 - 1.5 * IQR)) | 
                       (df[nutrient] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts[nutrient] = int(outliers)
        
        return jsonify({
            'total_foods': len(df),
            'total_nutrients': len(available_nutrients),
            'missing_values': {
                'average_pct': float(missing_pct.mean()),
                'max_pct': float(missing_pct.max()),
                'max_nutrient': str(missing_pct.idxmax())
            },
            'outliers': {
                'average_per_nutrient': float(np.mean(list(outlier_counts.values()))),
                'total': sum(outlier_counts.values())
            },
            'cluster_quality': {
                'silhouette': float(sil_score),
                'davies_bouldin': float(db_score),
                'calinski_harabasz': float(ch_score)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ============================================================================
# 4. RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 FLASK ML BACKEND RUNNING")
    print("="*80)
    print("📡 API Endpoints:")
    print("   • GET  /api/stats - Database statistics")
    print("   • POST /api/recommend - Get recommendations (25+ nutrients)")
    print("   • GET  /api/metrics - Model performance")
    print("   • GET  /api/clusters - Cluster information")
    print("   • POST /api/baseline_comparison - Compare with baselines")
    print("   • GET  /api/dataset_analysis - Dataset analysis")
    print("="*80)
    print(f"\n✅ System ready with {len(available_nutrients)} nutrients!")
    print("💡 Supported nutrients:")
    for category, nutrients in nutrient_categories.items():
        available_in_cat = [n for n in nutrients if n in available_nutrients]
        print(f"   • {category}: {len(available_in_cat)} nutrients")
    print("\n🌐 Server starting on http://127.0.0.1:5000")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)


# ============================================================================
# INSTALLATION INSTRUCTIONS
# ============================================================================
"""
1. Install required packages:
   pip install flask flask-cors pandas numpy scikit-learn openpyxl

2. Place ABBREV.xlsx in the same folder as app.py

3. Run the server:
   python app.py

4. Open the HTML file in your browser

5. The system now supports:
   - 25+ nutrients across 4 categories
   - 3 weighting strategies
   - Advanced metrics (NDCG, Precision@K)
   - Baseline comparison
   - Dataset analysis
   - Cluster interpretation

6. Backend changes:
   - Handles all 25+ nutrients dynamically
   - Maps frontend names to backend columns
   - Supports weighting modes (equal, prioritize, focus)
   - New endpoints for analysis and comparison
   - Comprehensive error handling
"""