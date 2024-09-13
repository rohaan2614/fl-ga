import torch

def set_flatten_model_back(model, x_flattern):
    with torch.no_grad():
        start = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p_extract = x_flattern[start : (start + p.numel())]
            p.set_(p_extract.view(p.shape).clone())
            if p.grad is not None:
                p.grad.zero_()
            start += p.numel()
            
def get_flatten_model_param(model):
    with torch.no_grad():
        return torch.cat(
            [p.detach().view(-1) for p in model.parameters() if p.requires_grad]
        )
            
def accuracy(output, target):
    pred = output.data.max(1, keepdim=True)[1]
    return pred.eq(target.data.view_as(pred)).cpu().float().mean()

def local_update_fedavg(clients: list, server, local_steps):
    train_loss_sum, train_acc_sum = 0, 0
    for client in clients:
        train_loss, train_acc = client.train_k_step_fedavg(k=local_steps)
        train_loss_sum += train_loss
        train_acc_sum += train_acc
    return train_loss_sum / len(clients), train_acc_sum / len(clients)

def local_train_1_step(clients: list):
    train_loss_sum, train_acc_sum = 0, 0
    for client in clients:
        train_loss, train_acc = client.train_k_step_fedavg(k=1)
        train_loss_sum += train_loss
        train_acc_sum += train_acc
    return train_loss_sum / len(clients), train_acc_sum / len(clients)

def get_flatten_model_grad(model) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat(
            [p.grad.detach().view(-1) for p in model.parameters() if p.requires_grad]
        )
