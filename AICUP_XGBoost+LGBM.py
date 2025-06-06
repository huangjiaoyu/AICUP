# Import Libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib

# Load & Preprocess Training Data
BASE_DIR = '/kaggle/input/table-tennis-aicup'  
train_info = pd.read_csv(os.path.join(BASE_DIR, '39_Training_Dataset/39_Training_Dataset/train_info.csv'))

train_info['gender'] = train_info['gender'].map({1: 1, 2: 0})
train_info['hand'] = train_info['hold racket handed'].map({1: 1, 2: 0})
train_info['years'] = train_info['play years']
train_info['level'] = train_info['level']

# Extract Features Function
def extract_features(seq):
    features = {}
    names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    for i in range(6):
        x = seq[:, i]
        prefix = names[i]
        features[f'{prefix}_mean'] = np.mean(x)
        features[f'{prefix}_std'] = np.std(x)
        features[f'{prefix}_max'] = np.max(x)
        features[f'{prefix}_min'] = np.min(x)
        features[f'{prefix}_median'] = np.median(x)
        features[f'{prefix}_range'] = np.max(x) - np.min(x)
    return features
    
# Build Feature DataFrame
X = []
y_gender, y_hand, y_years, y_level = [], [], [], []
data_path = os.path.join(BASE_DIR, '39_Training_Dataset/39_Training_Dataset/train_data')
# 確認 train_data 資料夾中有哪些檔案
for _, row in tqdm(train_info.iterrows(), total=len(train_info)):
    uid = row['unique_id']
    path = os.path.join(data_path, f"{uid}.txt")
    try:
        data = np.loadtxt(path)  # 改這裡：用空格分隔
        if data.shape[1] == 6 and data.shape[0] >= 10:
            X.append(extract_features(data))
            y_gender.append(row['gender'])
            y_hand.append(row['hand'])
            y_years.append(row['years'])
            y_level.append(row['level'])
    except Exception as e:
        print(f"ERROR loading {uid}: {e}")

X_df = pd.DataFrame(X)
X_df['gender'] = y_gender
X_df['hand'] = y_hand
X_df['years'] = y_years
X_df['level'] = y_level
print(X_df.shape)

# 任務設定與 Cross-Validation + 投票集成
TARGETS = {
    'gender': {'type': 'binary'},
    'hand': {'type': 'binary'},
    'years': {'type': 'multiclass', 'classes': [0, 1, 2]},
    'level': {'type': 'multiclass', 'classes': [2, 3, 4, 5]},
}
LABEL_ENCODINGS = {
    'level': {2: 0, 3: 1, 4: 2, 5: 3},
    'level_inv': {0: 2, 1: 3, 2: 4, 3: 5}
}

N_SPLITS = 5
SEED = 42
scores = {k: [] for k in TARGETS}

for task, info in TARGETS.items():
    print(f"\n=== Task: {task.upper()} ===")
    
    y = X_df[task]
    X = X_df.drop(columns=list(TARGETS))

    # 如果是 level 要先 encode 類別
    if task == 'level':
        y = y.map(LABEL_ENCODINGS['level'])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic' if info['type']=='binary' else 'multi:softprob',
            num_class=len(info['classes']) if info['type']=='multiclass' else None,
            eval_metric='logloss', use_label_encoder=False, random_state=SEED
        ).fit(X_tr, y_tr)

        lgbm_model = lgb.LGBMClassifier(
            objective='binary' if info['type']=='binary' else 'multiclass',
            num_class=len(info['classes']) if info['type']=='multiclass' else None,
            verbosity=-1,  # ✅ 關閉所有訊息            
            random_state=SEED
        ).fit(X_tr, y_tr)

        prob = (xgb_model.predict_proba(X_va) + lgbm_model.predict_proba(X_va)) / 2
        auc = roc_auc_score(y_va, prob[:, 1] if info['type']=='binary' else prob,
                            multi_class='ovr' if info['type']=='multiclass' else None,
                            average='micro' if info['type']=='multiclass' else None)
        scores[task].append(auc)
        print(f"Fold {fold+1}: ROC AUC = {auc:.4f}")

print("\n=== Final Ensemble Scores ===")
final_score = np.mean([np.mean(v) for v in scores.values()])
print(f"Leaderboard Avg Score: {final_score:.4f}")

# Grid Search Function
def run_grid_search(task_name, y, X, task_type, num_class=None):
    model = lgb.LGBMClassifier(objective='binary' if task_type=='binary' else 'multiclass',
                               num_class=num_class if task_type=='multiclass' else None,
                               random_state=42)
    scoring = 'roc_auc' if task_type=='binary' else 'roc_auc_ovr_weighted'
    param_grid = {
        'num_leaves': [15, 31],
        'learning_rate': [0.01, 0.05],
        'n_estimators': [100, 200],
    }
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=3, verbose=1, n_jobs=-1)
    grid.fit(X, y)
    print(f"Best for {task_name}: {grid.best_params_} | Score = {grid.best_score_:.4f}")
    return grid.best_estimator_
    
# 預測測試集與產出 submission.csv
test_info = pd.read_csv(os.path.join(BASE_DIR, '39_Test_Dataset/39_Test_Dataset/test_info.csv'))
test_data_dir = os.path.join(BASE_DIR, '39_Test_Dataset/39_Test_Dataset/test_data')
uids, test_feats = [], []

# 載入並處理 test data 特徵
for _, row in tqdm(test_info.iterrows(), total=len(test_info)):
    try:
        uid = row['unique_id']
        data = np.loadtxt(os.path.join(test_data_dir, f"{uid}.txt"))  # 不加 delimiter
        if data.shape[1] == 6 and data.shape[0] >= 10:
            feats = extract_features(data)
            uids.append(uid)
            test_feats.append(feats)
    except Exception as e:
        print(f"❌ ERROR loading {uid}: {e}")

X_test = pd.DataFrame(test_feats)

# 強制特徵順序對齊訓練資料
X = X_df.drop(columns=['gender', 'hand', 'years', 'level'])
X_test = X_test.reindex(columns=X.columns)
assert list(X_test.columns) == list(X.columns), "❌ Feature mismatch with train set!"

submission = pd.DataFrame({'unique_id': uids})

output_map = {
    'gender': ['gender'],
    'hand': ['hold racket handed'],
    'years': ['play years_0', 'play years_1', 'play years_2'],
    'level': ['level_2', 'level_3', 'level_4', 'level_5']
}

# LEVEL label encoding map（跟 CV 階段一致）
LEVEL_ENC = {2: 0, 3: 1, 4: 2, 5: 3}

for task, info in TARGETS.items():
    print(f"\n🔮 Predicting task: {task.upper()}")
    
    y = X_df[task]
    if task == 'level':
        y = y.map(LEVEL_ENC)

    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic' if info['type']=='binary' else 'multi:softprob',
        num_class=len(info['classes']) if info['type']=='multiclass' else None,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=SEED
    ).fit(X, y)

    joblib.dump(xgb_model, f'{task}_xgb_model.pkl')

    # Train LightGBM
    lgbm_model = lgb.LGBMClassifier(
        objective='binary' if info['type']=='binary' else 'multiclass',
        num_class=len(info['classes']) if info['type']=='multiclass' else None,
        verbosity=-1,
        random_state=SEED
    ).fit(X, y)

    joblib.dump(lgbm_model, f'{task}_lgbm_model.pkl')

    # Predict & ensemble
    probs = (xgb_model.predict_proba(X_test) + lgbm_model.predict_proba(X_test)) / 2

    # 寫入 submission.csv 結構
    if info['type'] == 'binary':
        submission[output_map[task][0]] = probs[:, 1]
    else:
        for i, col in enumerate(output_map[task]):
            submission[col] = probs[:, i]

# 儲存提交檔案
submission.to_csv('submission.csv', index=False, encoding='utf-8', float_format='%.4f')
print("✅ submission.csv 已成功產出，模型也已保存 .pkl 檔案")