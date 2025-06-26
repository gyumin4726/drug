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
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import mean_squared_error

# RDKit 경고 메시지 비활성화
RDLogger.DisableLog('rdApp.*')

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels=None, tokenizer=None):
        self.smiles = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        
        # 1. SMILES 텍스트 처리
        encoded = self.tokenizer(smiles, 
                               padding='max_length',
                               max_length=128,
                               truncation=True,
                               return_tensors='pt')
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # 2. 분자 특성 추출
        mol = Chem.MolFromSmiles(smiles)
        features = torch.tensor([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
        ], dtype=torch.float32)
        
        # 3. Morgan Fingerprint 생성
        fingerprint = torch.tensor(
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        ).float()
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return {'input_ids': input_ids, 
                   'attention_mask': attention_mask}, features, fingerprint, label
        return {'input_ids': input_ids, 
                'attention_mask': attention_mask}, features, fingerprint

class MultiModalMoleculeModel(nn.Module):
    def __init__(self, n_molecule_features=6):
        super().__init__()
        
        # 1. Transformer 기반 SMILES 처리
        self.language_model = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.smiles_projection = nn.Linear(768, 256)
        
        # 2. 분자 특성 처리
        self.molecule_mlp = nn.Sequential(
            nn.Linear(n_molecule_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # 3. Fingerprint 처리
        self.fingerprint_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 각 모델의 가중치 파라미터
        self.model_weights = nn.Parameter(torch.ones(3))
        
        # 융합 레이어
        self.fusion = nn.Sequential(
            nn.Linear(768, 512),  # 256 * 3 (SMILES + 분자특성 + Fingerprint)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, smiles_encoded, molecule_features, fingerprint):
        # SMILES 처리 - 입력 형식 수정
        smiles_output = self.language_model(
            input_ids=smiles_encoded['input_ids'],
            attention_mask=smiles_encoded['attention_mask']
        ).last_hidden_state[:, 0, :]  # CLS 토큰
        smiles_features = self.smiles_projection(smiles_output)
        
        # 2. 분자 특성 처리
        molecule_output = self.molecule_mlp(molecule_features)
        
        # 3. Fingerprint 처리
        fingerprint_output = self.fingerprint_mlp(fingerprint)
        
        # 가중치 정규화
        weights = F.softmax(self.model_weights, dim=0)
        
        # 특성 융합
        combined_features = torch.cat([
            smiles_features * weights[0],
            molecule_output * weights[1],
            fingerprint_output * weights[2]
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
            encoded, features, fingerprint, labels = batch
            # 데이터를 GPU로 이동
            encoded = {k: v.to(device) for k, v in encoded.items()}
            features = features.to(device)
            fingerprint = fingerprint.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(encoded, features, fingerprint)
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
                encoded, features, fingerprint, labels = batch
                # 데이터를 GPU로 이동
                encoded = {k: v.to(device) for k, v in encoded.items()}
                features = features.to(device)
                fingerprint = fingerprint.to(device)
                labels = labels.to(device)
                
                outputs = model(encoded, features, fingerprint)
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
        print(f'Model Weights - SMILES: {weights[0]:.3f}, Features: {weights[1]:.3f}, Fingerprint: {weights[2]:.3f}\n')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'val_loss': val_loss,
                'correlation': correlation,
                'rmse': rmse,
                'weights': {
                    'smiles': weights[0].item(),
                    'features': weights[1].item(),
                    'fingerprint': weights[2].item()
                }
            }
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_metrics': best_metrics,
                'epoch': epoch
            }, 'best_model.pth')
    
    return best_metrics

def main():
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터 로드
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # SMILES 토크나이저 초기화
    tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    
    # 학습/검증 데이터 분할
    train_smiles, val_smiles, train_labels, val_labels = train_test_split(
        train_data['Canonical_Smiles'].values,
        train_data['Inhibition'].values,
        test_size=0.2,
        random_state=42
    )
    
    # 데이터셋 생성
    train_dataset = MoleculeDataset(train_smiles, train_labels, tokenizer)
    val_dataset = MoleculeDataset(val_smiles, val_labels, tokenizer)
    test_dataset = MoleculeDataset(test_data['Canonical_Smiles'].values, tokenizer=tokenizer)
    
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
            encoded, features, fingerprint = batch
            # 데이터를 GPU로 이동
            encoded = {k: v.to(device) for k, v in encoded.items()}
            features = features.to(device)
            fingerprint = fingerprint.to(device)
            
            outputs = model(encoded, features, fingerprint)
            predictions.extend(outputs.cpu().numpy())
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'Inhibition': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()
