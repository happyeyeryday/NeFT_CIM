import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from config import Config
from load_dataset.load_dataset import fetch_dataloader
from models.resnet18_bnn import resnet18_bnn
from src.test_fn import test_fn


def elbo_loss(output, target, kl_divergence, criterion, num_samples, kl_weight=1e-5):
    """计算ELBO损失"""
    # 数据拟合损失 (负对数似然)
    likelihood_loss = criterion(output, target)
    
    # KL散度损失
    kl_loss = kl_divergence / num_samples  # 标准化KL散度
    
    # 总损失 = 负对数似然 + β * KL散度
    total_loss = likelihood_loss + kl_weight * kl_loss
    
    return total_loss, likelihood_loss, kl_loss


def train_fn_bnn(model, device, train_loader, optimizer, criterion, kl_weight=1e-5):
    """BNN训练函数"""
    model.train()
    running_loss = 0.0
    running_likelihood_loss = 0.0
    running_kl_loss = 0.0
    
    num_samples = len(train_loader.dataset)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        
        # 计算KL散度
        kl_div = model.kl_divergence()
        
        # 计算ELBO损失
        loss, likelihood_loss, kl_loss = elbo_loss(
            output, target, kl_div, criterion, num_samples, kl_weight
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 累积损失
        running_loss += loss.item()
        running_likelihood_loss += likelihood_loss.item()
        running_kl_loss += kl_loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_likelihood_loss = running_likelihood_loss / len(train_loader)
    epoch_kl_loss = running_kl_loss / len(train_loader)
    
    return epoch_loss, epoch_likelihood_loss, epoch_kl_loss


def test_fn_bnn_mc(model, device, test_loader, num_samples=10):
    """使用蒙特卡洛采样的BNN测试函数"""
    model.eval()
    correct = 0
    total = 0
    predictions_list = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 进行多次前向传播（蒙特卡洛采样）
            predictions = []
            for _ in range(num_samples):
                output = model(data)
                predictions.append(torch.softmax(output, dim=1))
            
            # 平均预测结果
            avg_prediction = torch.stack(predictions).mean(dim=0)
            predicted = torch.argmax(avg_prediction, dim=1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            predictions_list.append(avg_prediction.cpu())
    
    accuracy = 100.0 * correct / total
    return accuracy, predictions_list


def train_loop_bnn(model, device, criterion, optimizer, scheduler, train_loader, test_loader, epochs, save_path, kl_weight=1e-5):
    """BNN训练循环"""
    best_acc = 0.0
    best_ep = 0
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")
        
        # 训练
        train_loss, likelihood_loss, kl_loss = train_fn_bnn(
            model, device, train_loader, optimizer, criterion, kl_weight
        )
        
        # 测试
        acc, _ = test_fn_bnn_mc(model, device, test_loader, num_samples=10)
        
        print(f'Epoch {epoch+1}: Acc={acc:.2f}%, Train Loss={train_loss:.4f}, '
              f'Likelihood Loss={likelihood_loss:.4f}, KL Loss={kl_loss:.4f}')
        
        if acc > best_acc:
            best_acc = acc
            best_ep = epoch + 1
            torch.save(model.state_dict(), save_path)
        
        scheduler.step()
    
    print(f'Best accuracy: {best_acc:.2f}%')
    print(f'Best epoch: {best_ep}')


def train_part_bnn(args, train_loader, test_loader):
    """BNN训练部分"""
    save_path = f'check_points/resnet18_bnn_{args.dataset}_{args.mark}_{args.num}.pth'
    
    print(f'******BNN Train BEGIN!!!*****')
    print(f'ResNet-18 BNN')
    print(f'Save model at {save_path}')
    print(f'Epoch={Config.EPOCH}, lr={Config.LR}, batch size={Config.BATCH_SIZE}')
    print(f'Dataset={args.dataset}, KL weight={args.kl_weight}')
    
    # 根据数据集设置模型参数
    if args.dataset == 'mnist':
        in_channels, num_classes = 1, 10
    elif args.dataset == 'cifar10':
        in_channels, num_classes = 3, 10
    elif args.dataset == 'cifar100':
        in_channels, num_classes = 3, 100
    elif args.dataset == 'tiny':
        in_channels, num_classes = 3, 200
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 创建BNN模型
    net = resnet18_bnn(in_channels, num_classes)
    net = net.to(Config.DEVICE)
    
    # 设置优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCH)
    
    print(f'Optimizer: {optimizer}')
    
    # 开始训练
    train_loop_bnn(net, Config.DEVICE, criterion, optimizer, scheduler, 
                   train_loader, test_loader, Config.EPOCH, save_path, args.kl_weight)


def test_part_bnn(args, test_loader):
    """BNN测试部分"""
    save_path = f'check_points/resnet18_bnn_{args.dataset}_{args.mark}_{args.num}.pth'
    
    print(f'******BNN Test BEGIN!!!*****')
    print(f'ResNet-18 BNN')
    print(f'Load model from {save_path}')
    print(f'Dataset={args.dataset}')
    print(f'{Config.MC_times} times MC inference')
    
    # 根据数据集设置模型参数
    if args.dataset == 'mnist':
        in_channels, num_classes = 1, 10
    elif args.dataset == 'cifar10':
        in_channels, num_classes = 3, 10
    elif args.dataset == 'cifar100':
        in_channels, num_classes = 3, 100
    elif args.dataset == 'tiny':
        in_channels, num_classes = 3, 200
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 创建并加载模型
    net = resnet18_bnn(in_channels, num_classes)
    net.load_state_dict(torch.load(save_path))
    net = net.to(Config.DEVICE)
    
    # 进行蒙特卡洛测试
    accuracy, predictions = test_fn_bnn_mc(net, Config.DEVICE, test_loader, num_samples=Config.MC_times)
    
    print(f'BNN Test Accuracy: {accuracy:.2f}%')
    
    return accuracy

def extract_mean_weights(args):
    """从BNN模型中提取均值权重并保存为确定性模型"""
    bnn_save_path = f'check_points/resnet18_bnn_{args.dataset}_{args.mark}_{args.num}.pth'
    mean_save_path = f'check_points/res18_bnn_mean_{args.dataset}_{args.mark}_{args.num}.pth'
    
    print(f'******BNN Extract Mean Weights BEGIN!!!*****')
    print(f'Load BNN model from: {bnn_save_path}')
    print(f'Save mean weights to: {mean_save_path}')
    
    # 根据数据集设置模型参数
    if args.dataset == 'mnist':
        in_channels, num_classes = 1, 10
    elif args.dataset == 'cifar10':
        in_channels, num_classes = 3, 10
    elif args.dataset == 'cifar100':
        in_channels, num_classes = 3, 100
    elif args.dataset == 'tiny':
        in_channels, num_classes = 3, 200
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 创建BNN模型并加载权重
    bnn_model = resnet18_bnn(in_channels, num_classes)
    bnn_state_dict = torch.load(bnn_save_path, map_location='cpu')
    bnn_model.load_state_dict(bnn_state_dict)
    
    # 提取均值权重
    mean_state_dict = {}
    extracted_layers = 0
    
    for name, module in bnn_model.named_modules():
        if hasattr(module, 'weight_mu') and hasattr(module, 'weight_sigma'):
            # 贝叶斯层：提取权重均值
            mean_state_dict[name + '.weight'] = module.weight_mu.data.clone()
            extracted_layers += 1
            print(f"  Extracted weight mean from layer: {name}")
            
            if hasattr(module, 'bias_mu'):
                # 如果有偏置，也提取偏置均值
                mean_state_dict[name + '.bias'] = module.bias_mu.data.clone()
                print(f"  Extracted bias mean from layer: {name}")
                
        elif hasattr(module, 'weight'):
            # 普通层：直接复制权重
            mean_state_dict[name + '.weight'] = module.weight.data.clone()
            if hasattr(module, 'bias') and module.bias is not None:
                mean_state_dict[name + '.bias'] = module.bias.data.clone()
    
    print(f"  Total Bayesian layers processed: {extracted_layers}")
    
    # 保存均值权重
    torch.save(mean_state_dict, mean_save_path)
    print(f'Mean weights saved successfully!')
    print(f'******BNN Extract Mean Weights END!!!*****')
    
    return mean_save_path


def test_part_bnn_mean(args, test_loader):
    """使用提取的均值权重进行确定性测试（与其他方法相同的测试框架）"""
    # 这里需要使用标准ResNet18架构，加载均值权重
    from models.resnet18 import resnet18_test
    from config import s_factor
    
    mean_save_path = f'check_points/res18_bnn_mean_{args.dataset}_{args.mark}_{args.num}.pth'
    
    print(f'******BNN Mean Test BEGIN!!!*****')
    print(f'ResNet-18 with BNN mean weights')
    print(f'Load mean weights from: {mean_save_path}')
    print(f'Using standard noise testing framework')
    
    # 根据数据集设置参数
    if args.dataset == 'mnist':
        in_channels, num_classes = 1, 10
    elif args.dataset == 'cifar10':
        in_channels, num_classes = 3, 10
    elif args.dataset == 'cifar100':
        in_channels, num_classes = 3, 100
    elif args.dataset == 'tiny':
        in_channels, num_classes = 3, 200
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # 使用与其他方法相同的测试框架
    # 这需要与res18_main.py中的test_part保持一致
    device_name = 'RRAM1'  # 默认设备
    noise_variations = torch.arange(0, 0.4 + 0.05, 0.05).tolist()
    
    print(f'Noise variations: {noise_variations}')
    print(f'{Config.MC_times} times MC inference')
    
    std = []
    acc_list = []
    ci95 = []
    ci99 = []
    
    for noise_var in noise_variations:
        if noise_var == 0:
            # 无噪声测试
            net = resnet18_test(in_channels, num_classes, noise_var, s_factor[device_name]).to(Config.DEVICE)
            net.load_state_dict(torch.load(mean_save_path), strict=False)
            net.epoch_noise()
            accb = test_fn(net, Config.DEVICE, test_loader)
            acc_list.append(accb)
            std.append(0)
            ci95.append(0)
            ci99.append(0)
        else:
            # 有噪声的蒙特卡洛测试
            accb_loop = []
            for _ in range(Config.MC_times):
                net = resnet18_test(in_channels, num_classes, noise_var, s_factor[device_name]).to(Config.DEVICE)
                net.load_state_dict(torch.load(mean_save_path), strict=False)
                net.epoch_noise()
                accb = test_fn(net, Config.DEVICE, test_loader)
                accb_loop.append(accb)
            
            std.append(np.std(accb_loop, ddof=1))
            acc_list.append(np.mean(accb_loop))
            ci95.append((np.std(accb_loop, ddof=1) / np.sqrt(len(accb_loop))) * 1.96)
            ci99.append((np.std(accb_loop, ddof=1) / np.sqrt(len(accb_loop))) * 2.576)
    
    # 格式化结果
    AC = [float('{:.2f}'.format(i)) for i in acc_list]
    STD = [float('{:.2f}'.format(i)) for i in std]
    CI95 = [float('{:.2f}'.format(i)) for i in ci95]
    CI99 = [float('{:.2f}'.format(i)) for i in ci99]
    
    print(f'BNN Mean Weights Test Results:')
    print(f'accuracy: {AC}')
    print(f'std: {STD}')
    print(f'95% CI: {CI95}')
    print(f'99% CI: {CI99}')
    
    # 返回无噪声时的准确率作为主要结果
    return AC[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help="train / test", required=True)
    parser.add_argument('--dataset', type=str, help="mnist / cifar10 / cifar100 / tiny", required=True)
    parser.add_argument('--kl_weight', type=float, default=1e-5, help="KL divergence weight")
    parser.add_argument('--num', type=str, default='1', help="experiment number")
    parser.add_argument('--mark', type=float, default=1.0, help="model mark")
    
    args = parser.parse_args()
    
    def format_time(seconds):
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{days}d {hours}h {minutes}m {seconds:.2f}s"
    
    start_time = time.time()
    
    # 加载数据
    train_loader = fetch_dataloader(args.dataset, train=True)
    test_loader = fetch_dataloader(args.dataset, train=False)
    
    Config.LR = 1e-3  # BNN通常需要较小的学习率
    
    if args.mode == 'train':
        train_part_bnn(args, train_loader, test_loader)
    elif args.mode == 'test':
        test_part_bnn(args, test_loader)
    elif args.mode == 'extract_mean':
        # 真正的权重提取功能
        extract_mean_weights(args)
        print("BNN mean weights extraction completed successfully")
    elif args.mode == 'test_mean':
        # 使用均值权重进行标准测试
        test_part_bnn_mean(args, test_loader)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    
    end_time = time.time()
    print(f'Total time: {format_time(end_time - start_time)}')