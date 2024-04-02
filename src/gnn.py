import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import numpy as np


### Pre Processing

file_path = '../output/rnd/2013-7/1.csv'
data = pd.read_csv(file_path)

#print(data.head())

# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Check for any infinity or negative infinity values
infinite_values = (data == float('inf')).sum() + (data == float('-inf')).sum()

# Initialize StandardScaler
scaler = StandardScaler()

# Select features to scale - for demonstration we'll scale 'CPU usage [%]' and 'Memory usage [KB]'
features_to_scale = ['CPU usage [%]', 'Memory usage [KB]']

# Perform scaling on the features
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

# Let's check the first few rows of the dataframe after scaling
scaled_data_head = data.head()

#print(missing_values, infinite_values, scaled_data_head)

# For the additional preprocessing steps we will perform the following:
# 1. Convert the timestamp from UNIX time to a readable date-time format
# 2. Extract time features from the timestamp
# 3. Further feature engineering if needed

# Convert UNIX timestamp to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp [ms]'], unit='ms')

# Extract time features
data['Hour'] = data['Timestamp'].dt.hour
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek

# Drop the original timestamp column to avoid redundancy
data.drop('Timestamp [ms]', axis=1, inplace=True)

# For simplicity, we will not perform further complex feature engineering or outlier detection
# as it requires domain expertise and detailed analysis. 
# The final data.head() will reflect the current preprocessing state.
final_data_head = data.head()
final_data_head


### Node & Edge

# 예를 들어, pandas DataFrame `data`에는 전처리된 시계열 데이터가 있습니다.
# 이 데이터에서 노드 리스트와 에지 리스트를 생성합니다.

# 노드를 생성합니다. 이 경우 데이터의 각 행이 노드가 됩니다.
nodes = data.to_dict('records')  # 각 타임스탬프의 워크로드 특성을 포함하는 노드

# 연속된 타임스탬프를 에지로 연결합니다.
edges = [(i, i+1) for i in range(len(data)-1)]

# 노드 특성 선택: 예를 들어 CPU 사용량, 메모리 사용량, 시간 등의 특성을 선택합니다.
# 'nodes' 변수는 딕셔너리의 리스트이므로, NumPy 배열로 변환하는 과정이 필요합니다.
node_features = np.array([[node['CPU usage [%]'], node['Memory usage [KB]'], node['Hour']] for node in nodes])

# 데이터 객체 생성
x = torch.tensor(node_features, dtype=torch.float)  # 노드 특성
edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)  # 에지 인덱스
y = torch.tensor(data['CPU usage [%]'].values, dtype=torch.float)  # 타겟

# Data 객체 생성. 타겟이 있다면 y도 포함됩니다.
graph_data = Data(x=x, edge_index=edge_index, y=y)

print(graph_data)


### GNN Architecture

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)  # 예시로, 16은 임의로 설정한 히든 유닛 수입니다.
        self.conv2 = GCNConv(16, num_classes)  # 최종 클래스 또는 회귀 출력 수에 따라 조정됩니다.

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 그래프 컨볼루션 레이어
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # 최종 출력 레이어
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)  # 분류 문제를 위한 활성화 함수


# 모델, 손실 함수, 옵티마이저 초기화
model = GNN(num_node_features=3, num_classes=1)  # 회귀 문제의 경우 num_classes를 1로 설정합니다.
loss_func = torch.nn.MSELoss()  # 회귀 문제의 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 학습 루프
for epoch in range(200):  # 에포크 수는 실험을 통해 결정됩니다.
    optimizer.zero_grad()  # 기울기 초기화
    out = model(graph_data)  # 모델로부터 예측
    loss = loss_func(out, graph_data.y)  # 손실 계산
    loss.backward()  # 역전파
    optimizer.step()  # 최적화 단계

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


### 모델 평가

test_data = Data(x=x_test, edge_index=edge_index_test, y=y_test)

model.eval()
with torch.no_grad():
    predictions = model(test_data)
    test_loss = loss_func(predictions, test_data.y)

print(f'Test Loss: {test_loss.item()}')


### 모델 저장

# 모델 저장하기
torch.save(model.state_dict(), 'model.pth')

# 모델 로드하기
model = GNN(num_node_features=3, num_classes=1)  # 모델 재정의
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 평가 모드로 설정