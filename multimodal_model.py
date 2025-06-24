import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class MultiModalMoleculePredictor(nn.Module):
    def __init__(self, smiles_embedding_dim=768, rdkit_feature_dim=12, hidden_dim=512,
                 use_smiles=True, use_rdkit=True, use_fingerprint=True):
        super().__init__()
        
        self.use_smiles = use_smiles
        self.use_rdkit = use_rdkit
        self.use_fingerprint = use_fingerprint
        
        # 활성화된 모델 수 계산
        active_models = sum([use_smiles, use_rdkit, use_fingerprint])
        if active_models == 0:
            raise ValueError("At least one model must be activated")
        
        # SMILES 문자열을 처리하는 트랜스포머 (개선된 버전)
        if self.use_smiles:
            self.smiles_encoder = RobertaModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
            # Attention-based pooling 추가
            self.smiles_attention = nn.MultiheadAttention(smiles_embedding_dim, num_heads=8, batch_first=True)
            self.smiles_projection = nn.Sequential(
                nn.Linear(smiles_embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # RDKit 특성을 처리하는 개선된 신경망
        if self.use_rdkit:
            self.rdkit_encoder = nn.Sequential(
                nn.Linear(rdkit_feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
        
        # 분자 지문을 처리하는 개선된 신경망 (Residual Connections)
        if self.use_fingerprint:
            fingerprint_size = 3072
            self.fingerprint_encoder = FingerprintResNet(fingerprint_size, hidden_dim)
        
        # Cross-attention 메커니즘 추가
        if active_models > 1:
            self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # 특성 결합 및 최종 예측 (개선된 버전)
        combined_dim = hidden_dim * active_models
        self.final_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, smiles_input_ids, attention_mask, rdkit_features, fingerprints):
        features_list = []
        
        # SMILES 인코딩 (개선된 attention pooling)
        if self.use_smiles:
            smiles_hidden = self.smiles_encoder(
                input_ids=smiles_input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
            
            # Attention pooling 적용
            # CLS token과 다른 토큰들 간의 attention
            cls_token = smiles_hidden[:, 0:1, :]  # [batch_size, 1, hidden_dim]
            attn_output, _ = self.smiles_attention(cls_token, smiles_hidden, smiles_hidden)
            smiles_output = attn_output.squeeze(1)  # [batch_size, hidden_dim]
            
            smiles_features = self.smiles_projection(smiles_output)
            features_list.append(smiles_features)
        
        # RDKit 특성 인코딩
        if self.use_rdkit:
            rdkit_output = self.rdkit_encoder(rdkit_features)
            features_list.append(rdkit_output)
        
        # 분자 지문 인코딩
        if self.use_fingerprint:
            fingerprint_output = self.fingerprint_encoder(fingerprints)
            features_list.append(fingerprint_output)
        
        # Cross-attention 적용 (여러 모달리티가 있을 때)
        if len(features_list) > 1 and hasattr(self, 'cross_attention'):
            # 각 특성을 query로 사용하여 다른 특성들과 attention
            enhanced_features = []
            features_tensor = torch.stack(features_list, dim=1)  # [batch_size, num_modalities, hidden_dim]
            
            for i, feature in enumerate(features_list):
                query = feature.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                attn_output, _ = self.cross_attention(query, features_tensor, features_tensor)
                enhanced_feature = self.attention_norm(feature + attn_output.squeeze(1))
                enhanced_features.append(enhanced_feature)
            
            features_list = enhanced_features
        
        # 모든 특성 결합
        combined_features = torch.cat(features_list, dim=1)
        
        # 최종 예측
        return self.final_predictor(combined_features)

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

def clean_smiles(smiles):
    """
    SMILES 문자열에서 잘못된 표기법을 정리합니다.
    """
    # N['']를 N으로 변경 (잘못된 표기법 수정)
    smiles = smiles.replace("N['']", "N")
    
    # 기타 잘못된 표기법들 제거
    smiles = smiles.replace("['']", "")
    
    return smiles

def get_rdkit_features(smiles):
    try:
        # SMILES 문자열 정리
        smiles = clean_smiles(smiles)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(12)  # 확장된 특성 수
        
        return np.array([
            # 기본 분자 특성
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            # 추가 특성들
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCsp3(mol),
            Descriptors.BertzCT(mol),  # 분자 복잡도
            len(Chem.GetSymmSSSR(mol))  # 고리 개수
        ])
    except:
        return np.zeros(12)

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem import rdMolDescriptors

# 전역에 한 번 생성해두면 효율적
morgan_generator = GetMorganGenerator(radius=2, fpSize=1024)
torsion_generator = GetTopologicalTorsionGenerator(fpSize=1024)

def get_molecular_fingerprint(smiles):
    try:
        # SMILES 문자열 정리
        smiles = clean_smiles(smiles)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(3072)  # 1024 + 1024 + 1024 = 3072

        # Morgan Fingerprint (ECFP)
        morgan_fp = morgan_generator.GetFingerprint(mol)
        morgan_array = np.array(morgan_fp)
        
        # Topological Torsion Fingerprint (using new API)
        torsion_fp = torsion_generator.GetFingerprint(mol)
        torsion_array = np.array(torsion_fp)
        
        # MACCS Keys (167 bits -> 1024로 패딩)
        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        maccs_array = np.array(maccs_fp)
        maccs_padded = np.pad(maccs_array, (0, 1024 - len(maccs_array)), 'constant')
        
        # 모든 fingerprint 결합
        combined_fp = np.concatenate([morgan_array, torsion_array, maccs_padded])
        
        return combined_fp
    except:
        return np.zeros(3072)


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, labels=None):
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = RobertaTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        # SMILES 문자열 정리
        smiles = clean_smiles(smiles)
        
        # SMILES 토큰화
        tokens = self.tokenizer(
            smiles,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        # RDKit 특성 추출
        rdkit_features = torch.FloatTensor(get_rdkit_features(smiles))
        
        # 분자 지문 추출
        fingerprint = torch.FloatTensor(get_molecular_fingerprint(smiles))
        
        if self.labels is not None:
            label = torch.FloatTensor([self.labels[idx]])
            return input_ids, attention_mask, rdkit_features, fingerprint, label
        
        return input_ids, attention_mask, rdkit_features, fingerprint

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids, attention_mask, rdkit_features, fingerprints, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, rdkit_features, fingerprints)
        mse_loss = criterion(outputs, labels)
        mse_loss.backward()
        optimizer.step()
        
        # RMSE 계산 (평가용)
        rmse_loss = torch.sqrt(mse_loss)
        total_loss += rmse_loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, rdkit_features, fingerprints, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask, rdkit_features, fingerprints)
            mse_loss = criterion(outputs, labels)
            # RMSE 계산
            rmse_loss = torch.sqrt(mse_loss)
            total_loss += rmse_loss.item()
    
    return total_loss / len(val_loader)

def main():
    # 데이터 로드
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # 모델 설정 (튜닝된 버전)
    use_smiles = True       # SMILES 인코딩 사용 여부 (개선된 attention pooling)
    use_rdkit = True        # RDKit 특성 사용 여부 (확장된 12개 특성)
    use_fingerprint = True  # 분자 지문 사용 여부 (3가지 fingerprint 조합)
    
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
    
    # 데이터 로더 생성 (배치 크기 조정)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # 더 복잡한 모델이므로 배치 크기 감소
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 초기화
    model = MultiModalMoleculePredictor(
        use_smiles=use_smiles,
        use_rdkit=use_rdkit,
        use_fingerprint=use_fingerprint
    ).to(device)
    
    # 활성화된 모델 출력
    print("\n=== Tuned MultiModal Model Configuration ===")
    print(f"- SMILES encoding: {'enabled (with attention pooling)' if use_smiles else 'disabled'}")
    print(f"- RDKit features: {'enabled (12 enhanced features)' if use_rdkit else 'disabled'}")
    print(f"- Molecular fingerprints: {'enabled (3 combined fingerprints)' if use_fingerprint else 'disabled'}")
    print(f"- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # 학습 설정 (개선된 버전)
    criterion = nn.MSELoss()
    
    # 차별적 학습률 적용
    smiles_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'smiles_encoder' in name:
            smiles_params.append(param)
        else:
            other_params.append(param)
    
    # ChemBERTa는 더 낮은 학습률 적용
    param_groups = [
        {'params': smiles_params, 'lr': 1e-5},  # Pre-trained 모델은 낮은 학습률
        {'params': other_params, 'lr': 1e-3}   # 새로운 레이어는 높은 학습률
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 학습
    n_epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train RMSE: {train_loss:.4f}")
        print(f"Val RMSE: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # 테스트 데이터 예측
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, rdkit_features, fingerprints = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask, rdkit_features, fingerprints)
            predictions.extend(outputs.cpu().numpy().flatten())
    
    # 제출 파일 생성
    submit = pd.read_csv('sample_submission.csv')
    submit['Inhibition'] = predictions
    
    # 현재 날짜와 시각을 YYYYMMDD_HHMMSS 형식으로 파일명에 추가
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    submit_filename = f'multimodal_submit_{current_datetime}.csv'
    
    submit.to_csv(submit_filename, index=False)
    print(f"Predictions saved to {submit_filename}")

if __name__ == "__main__":
    main()