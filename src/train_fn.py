from tqdm import tqdm
from .loss_fn import negative_feedback


def train_fn(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
    # for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def train_fn_irs(args, model, device, train_loader, optimizer, criterion, beta):
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
    # for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = negative_feedback(outputs[-1], target, beta, criterion, outputs[:-1])
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def train_fn_ovf(args, model, device, train_loader, optimizer, criterion, beta):
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        model.noise_backbone = args.var1 + 0.15
        ovf1 = model(data)
        model.noise_backbone = args.var1 + 0.1
        ovf2 = model(data)
        model.noise_backbone = args.var1 + 0.05
        ovf3 = model(data)
        model.noise_backbone = args.var1
        loss = negative_feedback(output, target, beta, criterion, [ovf1, ovf2, ovf3])
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def elbo_loss(output, target, kl_divergence, criterion, num_samples, kl_weight=1e-5):
    """计算ELBO损失"""
    likelihood_loss = criterion(output, target)
    kl_loss = kl_divergence / num_samples
    total_loss = likelihood_loss + kl_weight * kl_loss
    return total_loss, likelihood_loss, kl_loss

def train_fn_bnn(model, device, train_loader, optimizer, criterion, kl_weight=1e-5):
    """BNN训练函数"""
    model.train()
    running_loss = 0.0
    num_samples = len(train_loader.dataset)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(data)
        kl_div = model.kl_divergence()
        
        loss, likelihood_loss, kl_loss = elbo_loss(
            output, target, kl_div, criterion, num_samples, kl_weight
        )
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

