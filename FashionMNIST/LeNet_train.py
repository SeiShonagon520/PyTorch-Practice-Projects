# 导入库
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 定义转换,将图片转换为张量数据，并进行归一化处理。
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))#单通道数据
    ]
)

# 数据集的一个加载
train_set = torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,transform=data_transform)#加载训练集
test_set = torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,transform=data_transform)#加载测试集

# 数据加载器创建
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)#使用训练数据集，批量大小为100，乱序，使用0个进程进行数据加载。
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)#使用测试数据集，批量大小为100，正序，使用0个进程进行数据加载。

# 创建神经网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),#定义卷积层，输入通道为1，输出通道为6，卷积核大小为5，填充为2
            nn.BatchNorm2d(6),#批处理标准化大小为6
            nn.ReLU(),#Relu激活函数
            nn.MaxPool2d(2, 2)#最大池化层，池化核大小为2，步长为2
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),#定义卷积层，输入通道为6，输出通道为16，卷积核大小为5
            nn.BatchNorm2d(16),#批处理标准化大小为16
            nn.ReLU(),#Relu激活函数
            nn.MaxPool2d(2, 2)#最大池化层，池化核大小为2，步长为2
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(16, 32, 3,padding=1),#定义卷积层，输入通道为16，输出通道为32，卷积核大小为3，填充为1
                nn.BatchNorm2d(32),#批处理标准化大小为32
                nn.ReLU(),#Relu激活函数
                nn.MaxPool2d(2, 2)#最大池化层，池化核大小为2，步长为2
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(32, 64, 3,padding=1),#定义卷积层，输入通道为32，输出通道为64，卷积核大小为3，填充为1
                nn.BatchNorm2d(64),#批处理标准化大小为64
                nn.ReLU(),#Relu激活函数
                nn.MaxPool2d(2, 2)#最大池化层，池化核大小为2，步长为2
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),#多维张量展平，将多维张量展平成一维张量
            nn.Linear(64 * 1 * 1, 120),#线性层，输入通道为64，输出通道为120
            nn.ReLU(),#Relu激活函数
            nn.Linear(120, 84),#线性层，输入通道为120，输出通道为84
            nn.ReLU(),#Relu激活函数
            nn.Linear(84, 10),#线性层，输入通道为84，输出通道为10
            nn.LogSoftmax(dim=1)#对数概率分布，dim=1表示对输出通道进行归一化
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
torch.manual_seed(20)#固定cpu随机数种子
torch.cuda.manual_seed_all(20)#固定gpu随机数种子
model = LeNet()#初始化卷积神经网络
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#调用gpu进行训练
model.to(device)#将模型加载到gpu上
loss_fn = nn.NLLLoss()#损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#使用Adam优化器，学习率设置为0.01
epochs = 50#训练轮数
batch_size = 100#训练批次的大小
final_losses = []#存储每个训练周期的平均损失

def train_model():#创建训练模型
    #初始化统计参数
    count = 0 #记录已处理的批次数量
    total = 0 #记录已处理的样本总数
    correct = 0 #记录预测正确的样本数
    for epoch in range(epochs):#外层循环遍历训练轮数
        for images , labels in train_loader:#内层循环遍历训练批次
            images,labels = images.to(device),labels.to(device)#将训练数据加载到gpu上
            train = images.view(images.size(0), 1, 28, 28)#将图片张量转换为四维张量数据

            #前向传播
            preds = model(train)

            #计算损失
            Loss = loss_fn(preds, labels)#计算预测值于真实标签的损失
            final_losses.append(Loss)#将损失添加到列表中

            #反向传播
            optimizer.zero_grad()#清空梯度缓存
            Loss.backward()#反向传播
            optimizer.step()#更新参数

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









