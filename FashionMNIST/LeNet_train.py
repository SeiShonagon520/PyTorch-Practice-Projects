import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 定义转换
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

# 数据集的一个加载
train_set = torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,transform=data_transform)
test_set = torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,transform=data_transform)

# 数据加载器创建
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

# 创建神经网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(16, 32, 3,padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(32, 64, 3,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1)
        )

# 前向传播
    def forward(self,x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output

# 超参数的设置
torch.manual_seed(20)
torch.cuda.manual_seed_all(20)
model = LeNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 50
batch_size = 100
final_losses = []

def train_model():
    count = 0
    total = 0
    correct = 0
    for epoch in range(epochs):
        for images , labels in train_loader:
            images,labels = images.to(device),labels.to(device)
            train = images.view(images.size(0), 1, 28, 28)

            #正向传播
            preds = model(train)

            #计算损失
            Loss = loss_fn(preds, labels)
            final_losses.append(Loss)

            #反向传播
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            count += 1
            total += labels.size(0)
            correct += labels.eq(preds.max(1)[1]).sum().item()
            accuracy  = 100 * correct / total
            if count % 600 == 0:
                print("Epoch: {}, Iteration: {}, Loss: {}, accuracy :{}".format(epoch, count, Loss.data, accuracy))

if __name__ == "__main__":
    train_model()
    torch.save(model.state_dict(), "mlenet.pth")
    print("Saved PyTorch Model State to mlenet.pth")









