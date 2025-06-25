import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
from datetime import datetime
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from rdkit.Chem import MACCSkeys, Lipinski
warnings.filterwarnings('ignore')

# 상수 정의
MORGAN_DIM = 1024
TORSION_DIM = 1024
MACCS_DESIRED_DIM = 1024
TOTAL_FP_DIM = MORGAN_DIM + TORSION_DIM + MACCS_DESIRED_DIM
CACHE_DIR = 'fingerprint_cache'

class FingerprintCache:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}
        
    def _get_cache_path(self, smiles):
        # 간단한 해시 함수를 사용하여 캐시 파일 이름 생성
        import hashlib
        return os.path.join(self.cache_dir, hashlib.md5(smiles.encode()).hexdigest() + '.pkl')
        
    def get(self, smiles):
        if smiles in self.cache:
            return self.cache[smiles]
            
        cache_path = self._get_cache_path(smiles)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                fp = pickle.load(f)
                self.cache[smiles] = fp
                return fp
        return None
        
    def put(self, smiles, fingerprint):
        self.cache[smiles] = fingerprint
        cache_path = self._get_cache_path(smiles)
        with open(cache_path, 'wb') as f:
            pickle.dump(fingerprint, f)

# 전역 캐시 객체 생성
fingerprint_cache = FingerprintCache()

class FingerprintPredictor(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        fingerprint_size = TOTAL_FP_DIM
        self.fingerprint_encoder = FingerprintResNet(fingerprint_size, hidden_dim)
        
        # 최종 예측 레이어 - 드롭아웃 비율 증가 및 레이어 구조 수정
        self.final_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.1),  # 모멘텀 값 낮춤
            nn.ReLU(),
            nn.Dropout(0.5),  # 드롭아웃 비율 증가
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, 64),
            nn.BatchNorm1d(64, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, fingerprints):
        x = self.fingerprint_encoder(fingerprints)
        return self.final_predictor(x)

# Residual Network for Fingerprints
class FingerprintResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim)
        
        self.final_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.final_norm(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
    def forward(self, x):
        return torch.relu(x + self.block(x))

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem import rdMolDescriptors

# 전역에 한 번 생성해두면 효율적
morgan_generator = GetMorganGenerator(radius=2, fpSize=1024)
torsion_generator = GetTopologicalTorsionGenerator(fpSize=1024)

def get_molecular_fingerprint(smiles):
    # 캐시된 fingerprint가 있는지 확인
    cached_fp = fingerprint_cache.get(smiles)
    if cached_fp is not None:
        return cached_fp

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(TOTAL_FP_DIM)

        # Morgan Fingerprint (ECFP)
        morgan_fp = morgan_generator.GetFingerprint(mol)
        morgan_array = np.array(morgan_fp)
        
        # Topological Torsion Fingerprint
        torsion_fp = torsion_generator.GetFingerprint(mol)
        torsion_array = np.array(torsion_fp)
        
        # MACCS Keys
        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        maccs_array = np.array(maccs_fp)
        maccs_padded = np.pad(maccs_array, (0, MACCS_DESIRED_DIM - len(maccs_array)), 'constant')
        
        # 모든 fingerprint 결합
        combined_fp = np.concatenate([morgan_array, torsion_array, maccs_padded])
        
        # 캐시에 저장
        fingerprint_cache.put(smiles, combined_fp)
        
        return combined_fp
    except:
        zero_fp = np.zeros(TOTAL_FP_DIM)
        fingerprint_cache.put(smiles, zero_fp)
        return zero_fp

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels=None):
        self.smiles_list = smiles_list
        self.labels = labels
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        # 분자 지문 추출
        fingerprint = torch.FloatTensor(get_molecular_fingerprint(smiles))
        
        if self.labels is not None:
            label = torch.FloatTensor([self.labels[idx]])
            return fingerprint, label
        
        return fingerprint

def train_epoch(model, train_loader, optimizer, criterion, device, label_normalizer):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        fingerprints, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(fingerprints)
        mse_loss = criterion(outputs, labels)
        mse_loss.backward()
        optimizer.step()
        
        # 원래 스케일로 변환하여 RMSE 계산
        denorm_outputs = torch.from_numpy(label_normalizer.inverse_transform(outputs.cpu().detach().numpy())).to(device)
        denorm_labels = torch.from_numpy(label_normalizer.inverse_transform(labels.cpu().detach().numpy())).to(device)
        rmse_loss = torch.sqrt(torch.mean((denorm_outputs - denorm_labels) ** 2))
        total_loss += rmse_loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device, label_normalizer):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            fingerprints, labels = [b.to(device) for b in batch]
            outputs = model(fingerprints)
            
            # 원래 스케일로 변환하여 RMSE 계산
            denorm_outputs = torch.from_numpy(label_normalizer.inverse_transform(outputs.cpu().numpy())).to(device)
            denorm_labels = torch.from_numpy(label_normalizer.inverse_transform(labels.cpu().numpy())).to(device)
            rmse_loss = torch.sqrt(torch.mean((denorm_outputs - denorm_labels) ** 2))
            total_loss += rmse_loss.item()
    
    return total_loss / len(val_loader)

class LabelNormalizer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, labels):
        self.scaler.fit(labels.reshape(-1, 1))
        
    def transform(self, labels):
        return self.scaler.transform(labels.reshape(-1, 1)).flatten()
        
    def inverse_transform(self, normalized_labels):
        return self.scaler.inverse_transform(normalized_labels.reshape(-1, 1)).flatten()
        
    def save(self, path):
        joblib.dump(self.scaler, path)
        
    def load(self, path):
        self.scaler = joblib.load(path)

def train_ml_models(X_train, y_train):
    print("\nML 모델 학습 중...")
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
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"{name} 모델 학습 중...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def get_ml_predictions(models, X_test):
    print("ML 모델로 예측 중...")
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    return predictions

def ensemble_predictions(dl_preds, ml_preds, weights=None):
    print("앙상블 예측 생성 중...")
    if weights is None:
        # 기본 가중치: 딥러닝(0.4), XGBoost(0.2), LightGBM(0.2), GradientBoosting(0.2)
        weights = {
            "DeepLearning": 0.4,
            "XGBoost": 0.2,
            "LightGBM": 0.2,
            "GradientBoosting": 0.2
        }
    
    final_pred = weights["DeepLearning"] * dl_preds
    for model_name, preds in ml_preds.items():
        final_pred += weights[model_name] * preds
    
    return final_pred

def main():
    # 데이터 로드
    print("데이터 로딩 중...")
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # Label Normalization
    label_normalizer = LabelNormalizer()
    label_normalizer.fit(train_data['Inhibition'].values)
    normalized_labels = label_normalizer.transform(train_data['Inhibition'].values)
    
    # 학습/검증 데이터 분할
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        train_data['Canonical_Smiles'].values,
        normalized_labels,
        test_size=0.2,
        random_state=42
    )
    
    # 라벨 정규화 스케일러 저장
    label_normalizer.save('label_scaler.pkl')
    
    # 데이터셋 생성
    train_dataset = MoleculeDataset(train_smiles, train_labels)
    val_dataset = MoleculeDataset(val_smiles, val_labels)
    test_dataset = MoleculeDataset(test_data['Canonical_Smiles'].values)
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 초기화
    model = FingerprintPredictor().to(device)
    
    # 모델 정보 출력
    print("\n=== Fingerprint Model Configuration ===")
    print(f"- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 학습 설정
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # 딥러닝 모델 학습
    n_epochs = 50
    best_val_loss = float('inf')
    
    print(f"\n=== 딥러닝 모델 학습 시작 (총 {n_epochs} 에폭) ===")
    print("Note: RMSE는 원래 스케일로 표시됩니다.")
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, label_normalizer)
        val_loss = validate(model, val_loader, criterion, device, label_normalizer)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train RMSE: {train_loss:.4f}")
        print(f"Val RMSE: {val_loss:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"*** 새로운 최적 모델 저장! (Val RMSE: {val_loss:.4f}) ***")
            
        print("-" * 50)
    
    # 딥러닝 모델 예측
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    dl_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            fingerprints = batch.to(device)
            outputs = model(fingerprints)
            dl_predictions.extend(outputs.cpu().numpy().flatten())
    
    dl_predictions = label_normalizer.inverse_transform(np.array(dl_predictions))
    
    # ML 모델용 특성 추출
    print("\nML 모델용 특성 추출 중...")
    train_features = []
    test_features = []
    
    for smiles in train_data['Canonical_Smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            train_features.append([0] * 18)  # 기본 특성의 개수
            continue
            
        features = [
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
        train_features.append(features)
    
    for smiles in test_data['Canonical_Smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            test_features.append([0] * 18)
            continue
            
        features = [
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
        test_features.append(features)
    
    X_train = np.array(train_features)
    X_test = np.array(test_features)
    y_train = train_data['Inhibition'].values
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ML 모델 학습 및 예측
    ml_models = train_ml_models(X_train_scaled, y_train)
    ml_predictions = get_ml_predictions(ml_models, X_test_scaled)
    
    # 앙상블 예측
    final_predictions = ensemble_predictions(dl_predictions, ml_predictions)
    
    # 결과 저장
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    submit_filename = f'ensemble_submit_{current_datetime}.csv'
    
    submit = pd.read_csv('sample_submission.csv')
    submit['Inhibition'] = final_predictions
    submit.to_csv(submit_filename, index=False)
    print(f"\n최종 예측 결과가 {submit_filename}에 저장되었습니다.")

if __name__ == "__main__":
    main()