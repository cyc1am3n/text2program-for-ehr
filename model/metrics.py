import torch

def get_accuracy(logit, label):
    _, predicted = torch.max(logit, dim=2)
    active_correct = (predicted == label)[~label.eq(-100)]
    try:
        accuracy = active_correct.sum().item()/active_correct.size(0)
    except:
        accuracy = 1.0 # None or 1.0?
    return accuracy