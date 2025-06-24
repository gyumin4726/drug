import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb

# 데이터 불러오기
train = pd.read_csv("train.csv")
train = train[['Canonical_Smiles', 'Inhibition']]

def extract_rdkit_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return [
                Descriptors.MolWt(mol),             # 분자량
                Descriptors.MolLogP(mol),           # LogP
                Descriptors.NumHAcceptors(mol),     # 수소 수용자 수
                Descriptors.NumHDonors(mol),        # 수소 기부자 수
                Descriptors.TPSA(mol),              # 극성 표면적
                Descriptors.NumRotatableBonds(mol), # 회전 가능한 결합 수
            ]
    except:
        pass
    return [0] * 6  # 실패 시 0으로 채움

# 특성 생성
train['Features'] = train['Canonical_Smiles'].apply(extract_rdkit_features)

# numpy 배열로 변환
train_x = np.stack(train['Features'].values)
train_y = train['Inhibition'].astype(float).values  # 안전하게 float 변환

# 모델 학습
model = xgb.XGBRegressor(random_state=42)
model.fit(train_x, train_y)

# 테스트셋 불러오기
test = pd.read_csv("test.csv")
test = test[['ID','Canonical_Smiles']]  # 예측에 필요한 컬럼만 유지

# 테스트셋 분자 특성 추출
test['Features'] = test['Canonical_Smiles'].apply(extract_rdkit_features)

# numpy 배열로 변환
test_x = np.stack(test['Features'].values)

# 예측 수행
test_y_pred = model.predict(test_x)

submit = pd.read_csv('./sample_submission.csv')
submit['Inhibition'] = test_y_pred
submit.head()

submit.to_csv('baseline_submit.csv',index=False)