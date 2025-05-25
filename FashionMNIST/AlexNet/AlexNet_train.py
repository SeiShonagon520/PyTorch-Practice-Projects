import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#定义数据转换，将图片转换为张量数据，并进行归一化处理。
data_transform = transforms.Compose(
    [
        transforms.RandomRotation(10),  # 随机旋转图片
        transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))#单通道数据
    ]
)

#数据集的一个加载
train_set = torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,transform=data_transform)#加载测试数据集
test_set = torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,transform=data_transform)#加载训练数据集

#数据加载器
train_loader = torch.utils.data.DataLoader(train_set,batch_size=100,shuffle=True)#测试数据加载器，批次大小为100，随机打乱数据
test_loader = torch.utils.data.DataLoader(test_set,batch_size=100,shuffle=False)#训练数据加载器，批次大小为100，不随机打乱数据

#AlexNet模型定义
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # AlexNet网络卷积结构
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1,1),#定义卷积层
            nn.ReLU(),#Relu激活函数
            nn.MaxPool2d(2, 2),#最大池化层

            nn.Conv2d(64, 128, 3, 1, 1),#定义卷积层
            nn.ReLU(),#Relu激活函数
            nn.MaxPool2d(2, 2),#最大池化层

            nn.Conv2d(128, 256, 3, 1,1),
            nn.ReLU(),#Relu激活函数
            nn.Conv2d(256, 256, 3, 1),
            nn.ReLU(),#Relu激活函数
            nn.Conv2d(256, 256, 3, 1),
            nn.ReLU(),#Relu激活函数
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),#多维张量展平，将多维张量展平成一维张量
            nn.Linear(256 * 1 * 1, 1024),#线性层，输入通道为256，输出通道为1024
            nn.ReLU(),#Relu激活函数
            nn.Dropout(0.5),#随机失活，失活概率为0.5
            nn.Linear(1024, 512),#线性层
            nn.ReLU(),#Relu激活函数
            nn.Dropout(0.5),#随机失活，失活概率为0.5
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)#对数概率分布，dim=1表示对输出通道进行归一化
        )

    def forward(self, x):#前向传播
        output = self.Conv(x)
        output = self.fc(output)
        return output

#超参数设置
torch.manual_seed(20)#固定cpu随机数种子
torch.cuda.manual_seed_all(20)#固定gpu随机数种子
model = AlexNet()#初始化模型
criterion = nn.CrossEntropyLoss()#定义损失函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#调用gpu进行训练
model.to(device)#将模型加载到gpu上
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#使用Adam优化器，学习率设置为
epochs = 30#设置训练轮数
batch_size = 100#设置训练批次大小

def train_model():
    #定义训练模型
    count = 0
    total = 0
    correct = 0

    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            train = images.view(images.size(0), 1, 28, 28)

            #前向传播
            preds = model(train)

            #计算损失
            Loss = criterion(preds, labels)

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
    torch.save(model.state_dict(), "alexnet.pth")



