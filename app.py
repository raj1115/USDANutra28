# ============================================================================
# MEMORY-OPTIMIZED FLASK BACKEND - 25+ NUTRIENTS FOR RENDER FREE TIER
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import gc
import os

app = Flask(__name__)
CORS(app)

print("="*80)
print("🚀 MEMORY-OPTIMIZED FOOD RECOMMENDATION SYSTEM")
print("="*80)

# ============================================================================
# NUTRIENT DEFINITIONS
# ============================================================================
nutrient_categories = {
    'Macronutrients': ['Protein_(g)', 'Lipid_Tot_(g)', 'Carbohydrt_(g)', 'Fiber_TD_(g)', 'Sugar_Tot_(g)'],
    'Energy': ['Energ_Kcal'],
    'Fats': ['FA_Sat_(g)', 'FA_Mono_(g)', 'FA_Poly_(g)', 'Cholestrl_(mg)'],
    'Minerals': ['Calcium_(mg)', 'Iron_(mg)', 'Magnesium_(mg)', 'Potassium_(mg)', 'Sodium_(mg)', 'Zinc_(mg)'],
    'Vitamins': ['Vit_C_(mg)', 'Vit_A_RAE', 'Vit_D_µg', 'Vit_E_(mg)', 'Vit_K_(µg)', 'Vit_B12_(µg)', 'Folate_Tot_(µg)']
}

all_nutrients = [n for cat in nutrient_categories.values() for n in cat]

nutrient_mapping = {
    'protein': 'Protein_(g)', 'carbs': 'Carbohydrt_(g)', 'fat': 'Lipid_Tot_(g)',
    'fiber': 'Fiber_TD_(g)', 'sugar': 'Sugar_Tot_(g)', 'calories': 'Energ_Kcal',
    'satfat': 'FA_Sat_(g)', 'monofat': 'FA_Mono_(g)', 'polyfat': 'FA_Poly_(g)',
    'cholesterol': 'Cholestrl_(mg)', 'calcium': 'Calcium_(mg)', 'iron': 'Iron_(mg)',
    'magnesium': 'Magnesium_(mg)', 'sodium': 'Sodium_(mg)', 'potassium': 'Potassium_(mg)',
    'zinc': 'Zinc_(mg)', 'vitc': 'Vit_C_(mg)', 'vita': 'Vit_A_RAE', 'vitd': 'Vit_D_µg',
    'vite': 'Vit_E_(mg)', 'vitk': 'Vit_K_(µg)', 'vitb12': 'Vit_B12_(µg)', 'folate': 'Folate_Tot_(µg)'
}

# ============================================================================
# LOAD DATA WITH MEMORY OPTIMIZATION
# ============================================================================
print("\n📂 Loading USDA database with memory optimization...")

try:
    # Load only required columns
    columns_to_load = ['Shrt_Desc'] + all_nutrients
    df = pd.read_excel("ABBREV.xlsx", usecols=lambda x: x in columns_to_load)
    
    print(f"✅ Loaded {len(df)} foods")
    
    # Memory optimization: convert to efficient dtypes
    df['Shrt_Desc'] = df['Shrt_Desc'].astype('string')
    
    # Get available nutrients
    available_nutrients = [n for n in all_nutrients if n in df.columns]
    print(f"📊 Using {len(available_nutrients)} nutrients")
    
    # Convert to float32 (uses 50% less memory than float64)
    for col in available_nutrients:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
    
    # Fill missing values
    df[available_nutrients] = df[available_nutrients].fillna(0)
    
    # Check memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"💾 DataFrame memory: {memory_mb:.2f} MB")
    
except Exception as e:
    print(f"❌ Error loading database: {e}")
    raise

# ============================================================================
# TRAIN MODEL ONCE AT STARTUP
# ============================================================================
print("\n🤖 Training K-Means model...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[available_nutrients].values.astype('float32'))

optimal_k = 8
# Reduce iterations to save memory during training
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=100)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(f"✅ Model trained! {len(df)} foods in {optimal_k} clusters")

# Calculate metrics
sil_score = silhouette_score(X_scaled, df['cluster'])
db_score = davies_bouldin_score(X_scaled, df['cluster'])
ch_score = calinski_harabasz_score(X_scaled, df['cluster'])
inertia = kmeans.inertia_

print(f"📊 Silhouette Score: {sil_score:.4f}")

# Clear memory
gc.collect()

print("="*80)
print("✅ SERVER READY")
print("="*80)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'message': 'Food Recommendation ML API',
        'nutrients': len(available_nutrients),
        'foods': len(df)
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'total_foods': int(len(df)),
        'total_nutrients': len(available_nutrients),
        'total_clusters': optimal_k,
        'cluster_sizes': {int(k): int(v) for k, v in df['cluster'].value_counts().to_dict().items()}
    })

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        
        # Build user preferences
        user_prefs = {}
        for frontend_name, backend_name in nutrient_mapping.items():
            if backend_name in available_nutrients:
                user_prefs[backend_name] = float(data.get(frontend_name, 0))
        
        # Options
        top_n = int(data.get('numRecommendations', 10))
        diversity_mode = bool(data.get('diversityMode', False))
        exclude_keywords = data.get('excludeKeywords', [])
        weighting_mode = data.get('weightingMode', 'prioritize')
        
        # Convert to array
        user_prefs_array = np.array([user_prefs.get(n, 0) for n in available_nutrients], dtype='float32').reshape(1, -1)
        user_scaled = scaler.transform(user_prefs_array)
        
        # Apply weighting
        if weighting_mode == 'prioritize':
            weights = np.array([3.0 if user_prefs.get(n, 0) > 0 else 1.0 for n in available_nutrients], dtype='float32')
            user_scaled_weighted = user_scaled * weights
            cluster_centers_weighted = kmeans.cluster_centers_.astype('float32') * weights
        elif weighting_mode == 'focus':
            adjusted_indices = [i for i, n in enumerate(available_nutrients) if user_prefs.get(n, 0) > 0]
            if len(adjusted_indices) > 0:
                user_scaled_weighted = user_scaled[:, adjusted_indices]
                cluster_centers_weighted = kmeans.cluster_centers_[:, adjusted_indices]
            else:
                user_scaled_weighted = user_scaled
                cluster_centers_weighted = kmeans.cluster_centers_
        else:
            user_scaled_weighted = user_scaled
            cluster_centers_weighted = kmeans.cluster_centers_
        
        # Find closest cluster
        cluster_distances = np.linalg.norm(cluster_centers_weighted - user_scaled_weighted, axis=1)
        closest_cluster = np.argmin(cluster_distances)
        
        # Select foods
        if diversity_mode:
            top_clusters = np.argsort(cluster_distances)[:3]
            cluster_foods = df[df['cluster'].isin(top_clusters)].copy()
        else:
            cluster_foods = df[df['cluster'] == closest_cluster].copy()
        
        # Apply exclusions
        if exclude_keywords:
            for keyword in exclude_keywords:
                cluster_foods = cluster_foods[
                    ~cluster_foods['Shrt_Desc'].str.contains(keyword, case=False, na=False)
                ]
        
        # Calculate similarities
        cluster_scaled = scaler.transform(cluster_foods[available_nutrients].values.astype('float32'))
        similarities = cosine_similarity(user_scaled, cluster_scaled)[0]
        cluster_foods['similarity'] = similarities
        
        # Get top N
        recommendations = cluster_foods.nlargest(top_n, 'similarity')
        
        # Calculate metrics
        avg_similarity = float(recommendations['similarity'].mean())
        diversity = int(len(recommendations['cluster'].unique()))
        precision_at_k = float((recommendations['similarity'] > 0.6).sum() / top_n)
        
        # Prepare results
        results = []
        for _, row in recommendations.iterrows():
            food_data = {
                'name': str(row['Shrt_Desc']),
                'cluster': int(row['cluster']),
                'similarity': float(row['similarity']),
                'protein': float(row.get('Protein_(g)', 0)),
                'carbs': float(row.get('Carbohydrt_(g)', 0)),
                'fat': float(row.get('Lipid_Tot_(g)', 0)),
                'fiber': float(row.get('Fiber_TD_(g)', 0)),
                'calories': float(row.get('Energ_Kcal', 0)),
                'calcium': float(row.get('Calcium_(mg)', 0)),
                'iron': float(row.get('Iron_(mg)', 0)),
                'sodium': float(row.get('Sodium_(mg)', 0))
            }
            
            # Add other nutrients if requested
            for frontend_name, backend_name in nutrient_mapping.items():
                if backend_name in available_nutrients and frontend_name not in food_data:
                    food_data[frontend_name] = float(row.get(backend_name, 0))
            
            results.append(food_data)
        
        return jsonify({
            'success': True,
            'recommendations': results,
            'metrics': {
                'avg_similarity': avg_similarity,
                'diversity': diversity,
                'precision_at_k': precision_at_k,
                'matched_cluster': int(closest_cluster)
            }
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def get_model_metrics():
    return jsonify({
        'silhouette_score': float(sil_score),
        'davies_bouldin_index': float(db_score),
        'calinski_harabasz_score': float(ch_score),
        'inertia': float(inertia)
    })

@app.route('/api/clusters', methods=['GET'])
def get_cluster_info():
    try:
        cluster_info = []
        for i in range(optimal_k):
            cluster_data = df[df['cluster'] == i]
            avg_nutrients = cluster_data[available_nutrients].mean()
            top_3 = avg_nutrients.nlargest(3)
            
            cluster_info.append({
                'cluster_id': int(i),
                'size': int(len(cluster_data)),
                'top_nutrients': {str(k): float(v) for k, v in top_3.items()},
                'sample_foods': cluster_data['Shrt_Desc'].head(3).tolist()
            })
        return jsonify(cluster_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/baseline_comparison', methods=['POST'])
def baseline_comparison():
    try:
        data = request.json
        user_prefs = {}
        for frontend_name, backend_name in nutrient_mapping.items():
            if backend_name in available_nutrients:
                user_prefs[backend_name] = float(data.get(frontend_name, 0))
        
        user_prefs_array = np.array([user_prefs.get(n, 0) for n in available_nutrients], dtype='float32').reshape(1, -1)
        user_scaled = scaler.transform(user_prefs_array)
        
        # Random baseline
        random_foods = df.sample(n=10, random_state=42)
        random_scaled = scaler.transform(random_foods[available_nutrients].values.astype('float32'))
        random_sim = float(cosine_similarity(user_scaled, random_scaled)[0].mean())
        
        # Popularity baseline
        df_temp = df.copy()
        df_temp['nutrient_score'] = df_temp[available_nutrients].sum(axis=1)
        popular_foods = df_temp.nlargest(10, 'nutrient_score')
        popular_scaled = scaler.transform(popular_foods[available_nutrients].values.astype('float32'))
        popular_sim = float(cosine_similarity(user_scaled, popular_scaled)[0].mean())
        
        # Protein baseline
        protein_sim = 0.0
        if 'Protein_(g)' in available_nutrients:
            protein_foods = df.nlargest(10, 'Protein_(g)')
            protein_scaled = scaler.transform(protein_foods[available_nutrients].values.astype('float32'))
            protein_sim = float(cosine_similarity(user_scaled, protein_scaled)[0].mean())
        
        return jsonify({
            'random_baseline': random_sim,
            'popularity_baseline': popular_sim,
            'protein_baseline': protein_sim
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/dataset_analysis', methods=['GET'])
def dataset_analysis():
    try:
        missing_pct = (df[available_nutrients].isnull().sum() / len(df)) * 100
        
        outlier_counts = {}
        for nutrient in available_nutrients[:10]:  # Limit to save memory
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
                'max_pct': float(missing_pct.max())
            },
            'cluster_quality': {
                'silhouette': float(sil_score),
                'davies_bouldin': float(db_score)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🌐 Server starting on port {port}")
    print("="*80 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
