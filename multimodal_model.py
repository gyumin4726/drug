import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from transformers import RobertaModel, RobertaTokenizer  # 트랜스포머 제거
from sklearn.metrics import mean_squared_error
from datetime import datetime
import random
import os

# RDKit 경고 메시지 비활성화
RDLogger.DisableLog('rdApp.*')

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels=None):
        self.smiles = smiles_list
        self.labels = labels
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        mol = Chem.MolFromSmiles(smiles)
        
        # 1. 분자 특성 추출
        features = torch.tensor([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
        ], dtype=torch.float32)
        
        # 2. Morgan Fingerprint 생성
        fingerprint = torch.tensor(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        ).float()
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return features, fingerprint, label
        return features, fingerprint

class MultiModalMoleculeModel(nn.Module):
    def __init__(self, n_molecule_features=6):
        super().__init__()
        
        # 1. 분자 특성 처리
        self.molecule_mlp = nn.Sequential(
            nn.Linear(n_molecule_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # 2. Fingerprint 처리
        self.fingerprint_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 각 모델의 가중치 파라미터 (2개로 축소)
        self.model_weights = nn.Parameter(torch.ones(2))
        
        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(512, 512),  # 256 * 2 (분자특성 + Fingerprint)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, molecule_features, fingerprint):
        # 1. 분자 특성 처리
        molecule_output = self.molecule_mlp(molecule_features)
        
        # 2. Fingerprint 처리
        fingerprint_output = self.fingerprint_mlp(fingerprint)
        
        # 가중치 정규화
        weights = F.softmax(self.model_weights, dim=0)
        
        # 특성 융합
        combined_features = torch.cat([
            molecule_output * weights[0],
            fingerprint_output * weights[1]
        ], dim=-1)
        
        # 최종 예측
        output = self.fusion(combined_features)
        return output.squeeze()

def calculate_metrics(predictions, labels):
    # RMSE 계산
    rmse = torch.sqrt(F.mse_loss(predictions, labels))
    
    # 상관계수 계산
    vx = predictions - torch.mean(predictions)
    vy = labels - torch.mean(labels)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    
    # 대회 평가 지표 계산
    score = 0.5 * (1 - min(rmse.item(), 1)) + 0.5 * corr.item()
    
    return {
        'rmse': rmse.item(),
        'correlation': corr.item(),
        'score': score
    }

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_metrics = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            features, fingerprint, labels = batch
            # 데이터를 GPU로 이동
            features = features.to(device)
            fingerprint = fingerprint.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features, fingerprint)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 검증
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                features, fingerprint, labels = batch
                # 데이터를 GPU로 이동
                features = features.to(device)
                fingerprint = fingerprint.to(device)
                labels = labels.to(device)
                
                outputs = model(features, fingerprint)
                val_loss += criterion(outputs, labels).item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 메트릭 계산
        val_loss /= len(val_loader)
        correlation = np.corrcoef(all_preds, all_labels)[0, 1]
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        
        # 모델 가중치 출력
        weights = F.softmax(model.model_weights, dim=0)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {total_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Correlation: {correlation:.4f}')
        print(f'RMSE: {rmse:.4f}')
        print(f'Model Weights - Features: {weights[0]:.3f}, Fingerprint: {weights[1]:.3f}\n')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'val_loss': val_loss,
                'correlation': correlation,
                'rmse': rmse,
                'weights': {
                    'features': weights[0].item(),
                    'fingerprint': weights[1].item()
                }
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_metrics': best_metrics,
                'epoch': epoch
            }, 'best_model.pth')
    
    return best_metrics

def set_seed(seed=42):
    """시드 고정 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    # 시드 고정 (재현 가능한 결과를 위해)
    set_seed(42)
    print("Random seed fixed to 42 for reproducibility")
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터 로드
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # 학습/검증 데이터 분할
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        train_data['Canonical_Smiles'].values,
        train_data['Inhibition'].values,
        test_size=0.2,
        random_state=42
    )
    
    # 데이터셋 생성
    train_dataset = MoleculeDataset(train_smiles, train_labels)
    val_dataset = MoleculeDataset(val_smiles, val_labels)
    test_dataset = MoleculeDataset(test_data['Canonical_Smiles'].values)
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 모델 초기화 및 GPU 이동
    model = MultiModalMoleculeModel().to(device)
    
    # 학습
    print("Starting training...")
    best_metrics = train_model(model, train_loader, val_loader, device)  # device 전달
    print("Training completed!")
    print("Best validation metrics:", best_metrics)
    
    # 테스트 데이터 예측
    print("Generating predictions...")
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            features, fingerprint = batch
            # 데이터를 GPU로 이동
            features = features.to(device)
            fingerprint = fingerprint.to(device)
            
            outputs = model(features, fingerprint)
            predictions.extend(outputs.cpu().numpy())
    
    # 제출 파일 생성 (날짜+시각 포함)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'submission_{current_time}.csv'
    
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'Inhibition': predictions
    })
    submission.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

if __name__ == "__main__":
    main()