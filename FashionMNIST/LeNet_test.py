import torch
import torch.nn as nn
from LeNet_train import LeNet,test_loader
def test_model(model_path = "mlenet.pth" ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet()
    model.to(device)
    model.load_state_dict(torch.load("mlenet.pth",weights_only=True))
    model.eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            total += labels.size(0)
            images, labels = images.to(device), labels.to(device)
            correct += labels.eq(model(images).max(1)[1]).sum().item()
        print("Accuracy of the network on the 10000 test images: {} %".format(100 * correct / total))

if __name__ == "__main__":
    test_model()
