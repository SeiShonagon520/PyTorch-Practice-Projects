import torch
import torch.nn as nn
from FashionMNIST.AlexNet.AlexNet_train import AlexNet, test_loader


def test_model(model_path = "alexnet.pth"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet()
    model.to(device)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()


    total = 0
    correct = 0

    with torch.no_grad():
        for images , labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            correct  += labels.eq(model(images).max(1)[1]).sum().item()

        print("Test Accuracy: {}".format(100 * correct / total))

if  __name__ == "__main__":
    test_model()
