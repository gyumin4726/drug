import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, AllChem

import warnings
warnings.filterwarnings('ignore')


print("데이터 로딩 중...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

print(f"훈련 데이터 : {train.shape}")
print(f"테스트 데이터 : {test.shape}")


def get_molecule_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * 2232

        basic_descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumAliphaticRings(mol),
            Lipinski.NumAromaticHeterocycles(mol),
            Lipinski.NumSaturatedHeterocycles(mol),
            Lipinski.NumAliphaticHeterocycles(mol),
            Descriptors.HeavyAtomCount(mol),
            Descriptors.RingCount(mol),
            Descriptors.NOCount(mol),
            Descriptors.NHOHCount(mol),
            Descriptors.NumRadicalElectrons(mol),
        ]

        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan_features = [int(bit) for bit in morgan_fp.ToBitString()]

        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_features = [int(bit) for bit in maccs_fp.ToBitString()]

        all_features = basic_descriptors + morgan_features + maccs_features

        return all_features
    except:
        return [0] * 2232


print("분자 특성 추출 중...")
train['features'] = train['Canonical_Smiles'].apply(get_molecule_descriptors)

X_train_list = train['features'].tolist()
feature_lengths = [len(x) for x in X_train_list]

if len(set(feature_lengths)) != 1:
    max_length = max(feature_lengths)
    X_train_list = [x + [0] * (max_length - len(x)) for x in X_train_list]

X_train = np.array(X_train_list)
y_train = train['Inhibition'].values

test['features'] = test['Canonical_Smiles'].apply(get_molecule_descriptors)
X_test_list = test['features'].tolist()
feature_lengths = [len(x) for x in X_test_list]

if len(set(feature_lengths)) != 1:
    max_length = max(feature_lengths)
    X_test_list = [x + [0] * (max_length - len(x)) for x in X_test_list]

if X_train.shape[1] != len(X_test_list[0]):
    diff = abs(X_train.shape[1] - len(X_test_list[0]))
    if X_train.shape[1] > len(X_test_list[0]):
        X_test_list = [x + [0] * diff for x in X_test_list]
    else:
        X_train = np.array([x.tolist() + [0] * diff for x in X_train])

X_test = np.array(X_test_list)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)


def normalized_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse / (np.max(y_true) - np.min(y_true))

def pearson_correlation(y_true, y_pred):
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return np.clip(corr, 0, 1)

def competition_score(y_true, y_pred):
    nrmse = min(normalized_rmse(y_true, y_pred), 1)
    pearson = pearson_correlation(y_true, y_pred)
    return 0.5 * (1 - nrmse) + 0.5 * pearson

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_nrmse = normalized_rmse(y_val, y_val_pred)
    val_pearson = pearson_correlation(y_val, y_val_pred)
    val_score = competition_score(y_val, y_val_pred)
    print(f"검증 NRMSE: {val_nrmse:.4f}")
    print(f"검증 Pearson: {val_pearson:.4f}")
    print(f"검증 점수: {val_score:.4f}")
    return model, val_score


models = {
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, gamma=0,
        reg_alpha=0.1, reg_lambda=1, random_state=42
    ),
    "LightGBM": lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.8,
        random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42
    )
}

best_score = -np.inf
best_model_name = None
trained_models = {}

for name, model in models.items():
    print(f"\n{name} 모델 학습 중...")
    trained_model, val_score = train_and_evaluate_model(
        model, X_train_final, y_train_final, X_val, y_val
    )
    trained_models[name] = trained_model
    if val_score > best_score:
        best_score = val_score
        best_model_name = name

print(f"\n최고 성능 모델: {best_model_name}, 검증 점수: {best_score:.4f}")


final_model = models[best_model_name]
final_model.fit(X_train_scaled, y_train)

test_preds = final_model.predict(X_test_scaled)

submission['Inhibition'] = test_preds
submission.to_csv('improved_submission.csv', index=False)
print("예측 결과 저장: improved_submission.csv")