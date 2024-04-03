import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import numpy as np

import matplotlib.pyplot as plt


# Todo
# 1. 데이터 전처리 검토: 이상치 제거 (Min - Max 값 plt로 확인, Cpu 사용률, 메모리 사용률 등)

def remove_outliers(data, threshold=3):
    for col in data.columns:
        if data[col].dtype == np.float64 or data[col].dtype == np.int64:
            mean = data[col].mean()
            std = data[col].std()
            data = data[(data[col] - mean).abs() <= threshold * std]
    return data


def remove_outliers_q(data, columns):
    for col in columns:
        lower_bound = data[col].quantile(0.01)
        upper_bound = data[col].quantile(0.99)
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data


def apply_scaling(data, columns):
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data


def pre_processing(file_path, train_size=0.8):
    data = pd.read_csv(file_path)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.median(), inplace=True)
    
    # 특성 스케일링
    features_to_scale = ['CPU usage [%]', 'Memory usage [KB]']
    data = remove_outliers_q(data, features_to_scale)
    data = apply_scaling(data, features_to_scale)
    #print(data)

    # 시간 관련 특성 추가
    data['Timestamp'] = pd.to_datetime(data['Timestamp [ms]'], unit='ms')
    data['Hour'] = data['Timestamp'].dt.hour
    data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
    data.drop('Timestamp [ms]', axis=1, inplace=True)

    # 데이터 분할
    total_length = len(data)
    split_point = int(total_length * train_size)

    train_data = data[:split_point]
    test_data = data[split_point:]

    print(train_data)

    return train_data, test_data


def create_nodes(data):
    node_features = np.array(data[['CPU usage [%]', 'Memory usage [KB]', 'Hour', 'DayOfWeek', 'CPU cores', 'Memory capacity provisioned [KB]']])
    x = torch.tensor(node_features, dtype=torch.float)
    return x


def create_edges(data):
    num_nodes = len(data)
    edges = [(i, i+1) for i in range(num_nodes-1)]
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    y = torch.tensor(data['CPU usage [%]'].values, dtype=torch.float)
    return edge_index, y


### Pre Processing

file_path = '../output/rnd/2013-7/1.csv'

train_data, test_data = pre_processing(file_path, train_size=0.8)


### Node & Edge

# 노드와 에지 생성
train_x = create_nodes(train_data)
train_edge_index, train_y = create_edges(train_data)

test_x = create_nodes(test_data)
test_edge_index, test_y = create_edges(test_data)

# Data 객체 생성
graph_data = Data(x=train_x, edge_index=train_edge_index, y=train_y)

print(graph_data.y.unsqueeze(1))


### GNN Architecture

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.conv2 = GCNConv(32, 16)
        self.bn2 = torch.nn.BatchNorm1d(16)
        self.conv3 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 그래프 컨볼루션 레이어
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        return x


# 모델, 손실 함수, 옵티마이저 초기화
model = GNN(num_node_features=6, num_classes=1)  # 회귀 문제의 경우 num_classes를 1로 설정합니다.
loss_func = torch.nn.MSELoss()  # 회귀 문제의 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# 학습 루프
for epoch in range(200):  # 에포크 수는 실험을 통해 결정됩니다.
    optimizer.zero_grad()  # 기울기 초기화
    out = model(graph_data)  # 모델로부터 예측
    loss = loss_func(out, graph_data.y.unsqueeze(1))  # 손실 계산
    loss.backward()  # 역전파
    optimizer.step()  # 최적화 단계

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


### 모델 평가

test_data = Data(x=test_x, edge_index=test_edge_index, y=test_y.unsqueeze(1))

model.eval()
with torch.no_grad():
    predictions = model(test_data)
    test_loss = loss_func(predictions, test_data.y)

print(f'Test Loss: {test_loss.item()}')


### 모델 저장

# 모델 저장하기
torch.save(model.state_dict(), 'model.pth')

# 모델 로드하기
model = GNN(num_node_features=6, num_classes=1)  # 모델 재정의
model.load_state_dict(torch.load('model.pth'))


### 예측값과 실제값 시각화

model.eval()  # 평가 모드로 설정
with torch.no_grad():
    predictions = model(test_data).squeeze()  # 예측값의 차원을 조정
    print(predictions)

#predictions = predictions.clamp(min=0)
#predictions = predictions.clamp(max=50)

# 실제 타겟 값
actuals = test_data.y.squeeze()  # 실제값의 차원을 조정

# 예측값과 실제값 비교 출력
for actual, predicted in zip(actuals[:10], predictions[:10]):  # 첫 10개의 결과만 출력
    print(f'Actual: {actual.item():.4f}, Predicted: {predicted.item():.4f}')

# 예측값과 실제값 시각화
plt.figure(figsize=(10, 5))
plt.plot(actuals.numpy(), label='Actual Values', color='blue')
plt.plot(predictions.numpy(), label='Predicted Values', color='red', linestyle='--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()