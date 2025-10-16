
import re, os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# =======================
# 🔧 CONFIGURATION
# =======================
TRAIN_PATH = '/dataset/train.csv'
TEST_PATH = '/dataset/test.csv'
OUTPUT_PATH = 'test_out.csv'

# Try import lightgbm
try:
    import lightgbm as lgb
    LGB_INSTALLED = True
except:
    LGB_INSTALLED = False
    print("⚠️ LightGBM not installed. Install with: pip install lightgbm")


def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    den[den == 0] = 1e-8
    return 100.0 * np.mean(num / den)


def extract_quantity_info(text):
    """Extract pack size, unit size, and total quantity - CRITICAL for price prediction"""
    text_lower = text.lower()
    
    # Extract pack/count
    pack = 1
    pack_patterns = [
        r'(\d+)\s*(?:pack|pk|ct|count|piece|pcs|box)',
        r'pack\s*of\s*(\d+)',
        r'(\d+)\s*x\s*\d+',  # "12 x 16oz" format
        r'(\d+)\s*-\s*pack',
    ]
    for pattern in pack_patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                pack = int(match.group(1))
                break
            except:
                pass
    
    # Extract unit size and normalize to oz equivalent
    unit_size_oz = 0
    
    # Fluid ounces
    match = re.search(r'(\d+\.?\d*)\s*(?:fl\.?\s*)?oz\b', text_lower)
    if match:
        unit_size_oz = float(match.group(1))
    
    # Milliliters (convert to oz)
    if unit_size_oz == 0:
        match = re.search(r'(\d+\.?\d*)\s*(?:ml|milliliter)', text_lower)
        if match:
            unit_size_oz = float(match.group(1)) * 0.033814
    
    # Liters (convert to oz)
    if unit_size_oz == 0:
        match = re.search(r'(\d+\.?\d*)\s*(?:l|liter)(?!i)', text_lower)
        if match:
            unit_size_oz = float(match.group(1)) * 33.814
    
    # Grams (convert to oz)
    if unit_size_oz == 0:
        match = re.search(r'(\d+\.?\d*)\s*(?:g|gram)\b', text_lower)
        if match:
            unit_size_oz = float(match.group(1)) * 0.035274
    
    # Kilograms (convert to oz)
    if unit_size_oz == 0:
        match = re.search(r'(\d+\.?\d*)\s*(?:kg|kilogram)', text_lower)
        if match:
            unit_size_oz = float(match.group(1)) * 35.274
    
    # Pounds (convert to oz)
    if unit_size_oz == 0:
        match = re.search(r'(\d+\.?\d*)\s*(?:lb|pound)', text_lower)
        if match:
            unit_size_oz = float(match.group(1)) * 16
    
    # Total quantity
    total_quantity_oz = pack * unit_size_oz
    
    return pack, unit_size_oz, total_quantity_oz


def extract_numeric_features(texts):
    """Extract comprehensive numeric features from text"""
    features = []
    
    for text in texts.astype(str):
        text_lower = text.lower()
        feat = {}
        
        # Quantity features (MOST IMPORTANT)
        pack, unit_size, total_qty = extract_quantity_info(text)
        feat['pack_size'] = pack
        feat['unit_size_oz'] = unit_size
        feat['total_quantity_oz'] = total_qty
        feat['log_pack_size'] = np.log1p(pack)
        feat['log_unit_size'] = np.log1p(unit_size)
        feat['log_total_qty'] = np.log1p(total_qty)
        
        # Basic text features
        feat['length'] = len(text)
        feat['word_count'] = len(text.split())
        feat['upper_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        feat['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        
        # Extract all numbers
        numbers = re.findall(r'\d+\.?\d*', text)
        feat['num_count'] = len(numbers)
        num_list = [float(n) for n in numbers if n]
        feat['num_sum'] = sum(num_list) if num_list else 0
        feat['num_max'] = max(num_list) if num_list else 0
        feat['num_min'] = min(num_list) if num_list else 0
        feat['num_avg'] = np.mean(num_list) if num_list else 0
        
        # Category indicators
        categories = {
            'food': r'\b(food|snack|candy|chocolate|cereal|cookie|chip|cracker|nut|seed|bar|granola)\b',
            'beverage': r'\b(water|juice|soda|drink|beverage|coffee|tea|milk|energy)\b',
            'health': r'\b(vitamin|supplement|protein|medicine|health|wellness|probiotic)\b',
            'beauty': r'\b(shampoo|soap|lotion|cream|beauty|cosmetic|makeup|conditioner)\b',
            'cleaning': r'\b(detergent|cleaner|soap|wipes|spray|disinfect|laundry|dish)\b',
            'baby': r'\b(baby|infant|diaper|formula|wipes|toddler)\b',
            'pet': r'\b(pet|dog|cat|animal|treat|puppy|kitten)\b',
            'paper': r'\b(paper|tissue|towel|napkin|toilet)\b',
        }
        
        for cat, pattern in categories.items():
            feat[f'cat_{cat}'] = 1 if re.search(pattern, text_lower) else 0
        
        # Price quality indicators
        feat['has_premium'] = 1 if re.search(r'\b(premium|deluxe|organic|natural|gourmet|artisan|craft)\b', text_lower) else 0
        feat['has_value'] = 1 if re.search(r'\b(value|economy|basic|budget|great value)\b', text_lower) else 0
        feat['has_bulk'] = 1 if re.search(r'\b(bulk|wholesale|family|mega|super|jumbo|xl)\b', text_lower) else 0
        
        # Container type
        feat['is_bottle'] = 1 if re.search(r'\b(bottle|btl)\b', text_lower) else 0
        feat['is_can'] = 1 if re.search(r'\b(can|cans)\b', text_lower) else 0
        feat['is_bag'] = 1 if re.search(r'\b(bag|pouch)\b', text_lower) else 0
        feat['is_box'] = 1 if re.search(r'\b(box|carton)\b', text_lower) else 0
        
        features.append(list(feat.values()))
    
    return np.array(features, dtype=np.float32)


def extract_brand_features(df_train, df_test, top_k=800):
    """Extract brand with target encoding and smoothing"""
    
    def get_brand(text):
        if not isinstance(text, str) or len(text) < 3:
            return 'unknown'
        
        # Method 1: First capitalized word or phrase before comma/dash
        seg = re.split(r'[,:()\-]', text.strip())[0].strip()
        
        # Look for capitalized words
        cap_words = re.findall(r'[A-Z][a-z]+|[A-Z]+(?![a-z])', seg)
        if cap_words:
            brand = ' '.join(cap_words[:2])
            if len(brand) >= 2:
                return brand
        
        # Method 2: First 1-2 words
        words = text.split()[:2]
        return ' '.join(words).strip() if words else 'unknown'
    
    train_brands = df_train['catalog_content'].fillna('').apply(get_brand)
    test_brands = df_test['catalog_content'].fillna('').apply(get_brand)
    
    # Keep only frequent brands
    brand_counts = train_brands.value_counts()
    top_brands = set(brand_counts.head(top_k).index)
    
    train_brands = train_brands.apply(lambda x: x if x in top_brands else 'other')
    test_brands = test_brands.apply(lambda x: x if x in top_brands else 'other')
    
    # Target encoding with smoothing
    y = np.log1p(df_train['price'].values)
    brand_stats = pd.DataFrame({
        'brand': train_brands,
        'price': y
    }).groupby('brand')['price'].agg(['mean', 'std', 'count'])
    
    global_mean = y.mean()
    global_std = y.std()
    
    # Bayesian smoothing
    min_samples = 10
    brand_stats['smooth_mean'] = (
        (brand_stats['mean'] * brand_stats['count'] + global_mean * min_samples) /
        (brand_stats['count'] + min_samples)
    )
    
    brand_stats['smooth_std'] = (
        (brand_stats['std'].fillna(global_std) * brand_stats['count'] + global_std * min_samples) /
        (brand_stats['count'] + min_samples)
    )
    
    brand_mean_dict = brand_stats['smooth_mean'].to_dict()
    brand_std_dict = brand_stats['smooth_std'].to_dict()
    
    # Create features
    train_feat_mean = train_brands.map(lambda x: brand_mean_dict.get(x, global_mean)).values.reshape(-1, 1)
    test_feat_mean = test_brands.map(lambda x: brand_mean_dict.get(x, global_mean)).values.reshape(-1, 1)
    
    train_feat_std = train_brands.map(lambda x: brand_std_dict.get(x, global_std)).values.reshape(-1, 1)
    test_feat_std = test_brands.map(lambda x: brand_std_dict.get(x, global_std)).values.reshape(-1, 1)
    
    # Brand frequency (log-scaled)
    freq_dict = brand_counts.to_dict()
    train_freq = train_brands.map(lambda x: np.log1p(freq_dict.get(x, 0))).values.reshape(-1, 1)
    test_freq = test_brands.map(lambda x: np.log1p(freq_dict.get(x, 0))).values.reshape(-1, 1)
    
    return (
        np.hstack([train_feat_mean, train_feat_std, train_freq]),
        np.hstack([test_feat_mean, test_feat_std, test_freq])
    )


def create_text_features(train_texts, test_texts, max_features=15000):
    """Create TF-IDF text features with dimensionality reduction"""
    
    # Character n-grams (capture patterns like "12oz", "pack")
    char_vec = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=5000,
        min_df=2,
        sublinear_tf=True
    )
    
    # Word n-grams
    word_vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        token_pattern=r'\b[a-zA-Z0-9]{2,}\b'
    )
    
    print("  📝 Fitting character n-grams...")
    X_char_train = char_vec.fit_transform(train_texts)
    X_char_test = char_vec.transform(test_texts)
    
    print("  📝 Fitting word n-grams...")
    X_word_train = word_vec.fit_transform(train_texts)
    X_word_test = word_vec.transform(test_texts)
    
    # Dimensionality reduction
    print("  📉 Applying SVD...")
    svd_char = TruncatedSVD(n_components=80, random_state=42)
    svd_word = TruncatedSVD(n_components=180, random_state=42)
    
    X_char_train_svd = svd_char.fit_transform(X_char_train)
    X_char_test_svd = svd_char.transform(X_char_test)
    
    X_word_train_svd = svd_word.fit_transform(X_word_train)
    X_word_test_svd = svd_word.transform(X_word_test)
    
    print(f"  ✅ Explained variance: char={svd_char.explained_variance_ratio_.sum():.2%}, word={svd_word.explained_variance_ratio_.sum():.2%}")
    
    return (
        np.hstack([X_char_train_svd, X_word_train_svd]),
        np.hstack([X_char_test_svd, X_word_test_svd])
    )


# ===============================
# MAIN PIPELINE
# ===============================
print("="*60)
print("🚀 ENHANCED PRICE PREDICTION PIPELINE")
print("="*60)

print("\n📥 Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Normalize column names
train.columns = [c.strip() for c in train.columns]
test.columns = [c.strip() for c in test.columns]

# Handle catalog_content column
for df in [train, test]:
    if 'catalog_content' not in df.columns:
        for c in df.columns:
            if 'catalog' in c.lower():
                df.rename(columns={c: 'catalog_content'}, inplace=True)
                break

train['catalog_content'] = train['catalog_content'].fillna('').astype(str)
test['catalog_content'] = test['catalog_content'].fillna('').astype(str)

print(f"📊 Train shape: {train.shape}, Test shape: {test.shape}")

# Target variable
y = train['price'].astype(float).values
y_log = np.log1p(y)

print(f"\n💰 Price statistics:")
print(f"   Min: ${y.min():.2f}")
print(f"   Max: ${y.max():.2f}")
print(f"   Mean: ${y.mean():.2f}")
print(f"   Median: ${np.median(y):.2f}")

# ===============================
# FEATURE ENGINEERING
# ===============================
print("\n" + "="*60)
print("🔧 FEATURE ENGINEERING")
print("="*60)

print("\n1️⃣ Extracting numeric & quantity features...")
X_numeric_train = extract_numeric_features(train['catalog_content'])
X_numeric_test = extract_numeric_features(test['catalog_content'])
print(f"   ✅ Created {X_numeric_train.shape[1]} numeric features")

# Check feature quality
print("\n🔍 Top feature correlations with price:")
for i in range(min(6, X_numeric_train.shape[1])):
    corr = np.corrcoef(X_numeric_train[:, i], y_log)[0, 1]
    if abs(corr) > 0.01:
        print(f"   Feature {i}: {corr:.3f}")

print("\n2️⃣ Extracting brand features...")
X_brand_train, X_brand_test = extract_brand_features(train, test, top_k=800)
print(f"   ✅ Created {X_brand_train.shape[1]} brand features")

print("\n3️⃣ Creating text features...")
X_text_train, X_text_test = create_text_features(
    train['catalog_content'].values,
    test['catalog_content'].values,
    max_features=15000
)
print(f"   ✅ Created {X_text_train.shape[1]} text features")

# Combine all features
X_train_full = np.hstack([X_numeric_train, X_brand_train, X_text_train])
X_test_full = np.hstack([X_numeric_test, X_brand_test, X_text_test])

print(f"\n✅ Final feature shape: {X_train_full.shape}")

# ===============================
# FAST SINGLE SPLIT TRAINING
# ===============================
print("\n" + "="*60)
print("🎯 MODEL TRAINING")
print("="*60)

print("\n📊 Splitting data (85% train, 15% validation)...")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_log, test_size=0.15, random_state=42
)

print(f"   Train: {X_tr.shape[0]} samples")
print(f"   Val: {X_val.shape[0]} samples")

# Initialize models
base_models = []

if LGB_INSTALLED:
    print("\n✅ Using LightGBM (optimized for SMAPE)")
    
    # Model 1: Focus on accuracy
    base_models.append(('lgb1', lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        num_leaves=100,
        learning_rate=0.05,
        n_estimators=1000,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )))
    
    # Model 2: Focus on generalization
    base_models.append(('lgb2', lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        num_leaves=150,
        learning_rate=0.03,
        n_estimators=1200,
        min_child_samples=15,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.15,
        reg_lambda=0.15,
        random_state=123,
        verbose=-1,
        n_jobs=-1
    )))
    
    # Model 3: Different tree structure
    base_models.append(('lgb3', lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        num_leaves=80,
        learning_rate=0.04,
        n_estimators=1500,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=456,
        verbose=-1,
        n_jobs=-1
    )))
else:
    print("\n⚠️ LightGBM not found, using sklearn models")
    base_models.append(('rf', RandomForestRegressor(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )))

# Add ensemble diversity
base_models.extend([
    ('etr', ExtraTreesRegressor(
        n_estimators=200,
        max_depth=30,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )),
    ('ridge', Ridge(alpha=3.0, random_state=42))
])

# Train all base models
print(f"\n🔹 Training {len(base_models)} base models...")
val_predictions = []
test_predictions = []

for i, (name, model) in enumerate(base_models, 1):
    print(f"\n[{i}/{len(base_models)}] Training {name}...", end=' ')
    model.fit(X_tr, y_tr)
    
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test_full)
    
    val_predictions.append(val_pred)
    test_predictions.append(test_pred)
    
    val_smape = smape(np.expm1(y_val), np.expm1(val_pred))
    val_mae = mean_absolute_error(np.expm1(y_val), np.expm1(val_pred))
    print(f"✓")
    print(f"        Val SMAPE: {val_smape:.2f}% | Val MAE: ${val_mae:.2f}")

# Stack predictions
val_predictions = np.column_stack(val_predictions)
test_predictions = np.column_stack(test_predictions)

# Simple ensemble average
ensemble_val = val_predictions.mean(axis=1)
ensemble_smape = smape(np.expm1(y_val), np.expm1(ensemble_val))
print(f"\n📈 Simple Ensemble Average Val SMAPE: {ensemble_smape:.2f}%")

# ===============================
# META-MODEL (STACKING)
# ===============================
print("\n🎯 Training meta-model (stacking)...")

# Try different meta-models
meta_models = [
    ('ridge_1', Ridge(alpha=1.0, random_state=42)),
    ('ridge_5', Ridge(alpha=5.0, random_state=42)),
    ('ridge_10', Ridge(alpha=10.0, random_state=42)),
]

best_meta_smape = float('inf')
best_meta_model = None
best_meta_name = None

for meta_name, meta_model in meta_models:
    meta_model.fit(val_predictions, y_val)
    meta_val_pred = meta_model.predict(val_predictions)
    meta_smape_score = smape(np.expm1(y_val), np.expm1(meta_val_pred))
    
    if meta_smape_score < best_meta_smape:
        best_meta_smape = meta_smape_score
        best_meta_model = meta_model
        best_meta_name = meta_name

print(f"   Best meta-model: {best_meta_name}")
print(f"   📊 Meta-model Val SMAPE: {best_meta_smape:.2f}%")

# Choose best approach
if best_meta_smape < ensemble_smape:
    print(f"\n✅ Using meta-model (improvement: {ensemble_smape - best_meta_smape:.2f}%)")
    final_test_pred_log = best_meta_model.predict(test_predictions)
    final_val_smape = best_meta_smape
else:
    print(f"\n✅ Using simple ensemble average")
    final_test_pred_log = test_predictions.mean(axis=1)
    final_val_smape = ensemble_smape

final_test_pred = np.expm1(final_test_pred_log)

# ===============================
# POST-PROCESSING
# ===============================
print("\n" + "="*60)
print("🔧 POST-PROCESSING")
print("="*60)

# Bias correction
if best_meta_smape < ensemble_smape:
    val_pred_final = best_meta_model.predict(val_predictions)
else:
    val_pred_final = val_predictions.mean(axis=1)

val_residual_ratio = np.expm1(y_val).mean() / np.expm1(val_pred_final).mean()
print(f"\n📊 Validation mean price: ${np.expm1(y_val).mean():.2f}")
print(f"📊 Predicted mean price: ${np.expm1(val_pred_final).mean():.2f}")
print(f"📊 Residual ratio: {val_residual_ratio:.4f}")

# Apply small correction if systematic bias exists
if 0.98 < val_residual_ratio < 1.02:
    final_test_pred = final_test_pred * val_residual_ratio
    print(f"✅ Applied bias correction: {val_residual_ratio:.4f}")
else:
    print(f"⚠️ No correction applied (ratio outside safe range)")

# Ensure positive predictions
final_test_pred = np.maximum(final_test_pred, 0.01)

# Round to 2 decimals
final_test_pred = np.round(final_test_pred, 2)

# ===============================
# SAVE RESULTS
# ===============================
print("\n" + "="*60)
print("💾 SAVING RESULTS")
print("="*60)

output = pd.DataFrame({
    'sample_id': test['sample_id'],
    'price': final_test_pred
})

output.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Predictions saved to: {OUTPUT_PATH}")
print(f"\n📊 Final Validation SMAPE: {final_val_smape:.2f}%")
print(f"\n📊 Prediction Statistics:")
print(f"   Min: ${final_test_pred.min():.2f}")
print(f"   Max: ${final_test_pred.max():.2f}")
print(f"   Mean: ${final_test_pred.mean():.2f}")
print(f"   Median: ${np.median(final_test_pred):.2f}")

print("\n" + "="*60)
print("✅ PIPELINE COMPLETE!")
print("="*60)