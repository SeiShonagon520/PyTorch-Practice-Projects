#导入库
import torch
import torch.nn as nn
from LeNet_train import LeNet,test_loader
def test_model(model_path = "mlenet.pth"):#创建测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#使用gpu运行
    model = LeNet()#初始化神经网络
    model.to(device)#将模型加载到gpu上
    model.load_state_dict(torch.load("mlenet.pth",weights_only=True))#加载模型权重
    model.eval()#切换到评估模式

    total = 0#测试数据总数
    correct = 0#测试数据正确数
    with torch.no_grad():#禁用梯度
        for images, labels in test_loader:#遍历测试数据
            total += labels.size(0)#测试数据总数
            images, labels = images.to(device), labels.to(device)#将测试数据加载到gpu上运行
            correct += labels.eq(model(images).max(1)[1]).sum().item()#测试数据正确数
        print("Accuracy of the network on the 10000 test images: {} %".format(100 * correct / total))

if __name__ == "__main__":
    test_model()
