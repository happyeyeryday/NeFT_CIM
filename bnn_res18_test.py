import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import sys
import threading
import signal
import psutil

def run_experiment(method, dataset, mode, var1=None, var2=None, kl_weight=None, num='1', mark='1.0'):
    """运行单个实验"""
    if method == 'bnn':
        cmd = ['python', 'main_bnn.py', '--mode', mode, '--dataset', dataset, 
               '--num', num, '--mark', mark]
        if kl_weight is not None:
            cmd.extend(['--kl_weight', str(kl_weight)])
    elif method == 'bnn_mean':
        # 特殊处理：BNN均值权重使用标准测试
        cmd = ['python', 'res18_main.py', '--mode', mode, '--type', 'base', 
               '--dataset', dataset, '--device', 'RRAM1', '--num', num, '--mark', mark]
        cmd.extend(['--bnn_mean', 'True'])
        if var1 is not None:
            cmd.extend(['--var1', str(var1)])
        if var2 is not None:
            cmd.extend(['--var2', str(var2)])
    else:
        cmd = ['python', 'res18_main.py', '--mode', mode, '--type', method, 
               '--dataset', dataset, '--device', 'RRAM1', '--num', num, '--mark', mark]
        if var1 is not None:
            cmd.extend(['--var1', str(var1)])
        if var2 is not None:
            cmd.extend(['--var2', str(var2)])
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    # 🔧 修复：为测试模式设置更短超时
    if mode in ['test', 'test_mean']:
        timeout = 1800  # 测试阶段30分钟超时（考虑MC推理）
        print(f"⏱️  Starting subprocess with timeout={timeout}s (5 minutes for test)...")
    else:
        timeout = 7200  # 训练阶段2小时超时
        print(f"⏱️  Starting subprocess with timeout={timeout}s (2 hours for train)...")
    
    try:
        # 使用Popen以便更好地控制进程
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        try:
            # 等待进程完成，带超时
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            print(f"⏰ Process timed out after {timeout} seconds, terminating...")
            
            # 强制终止进程及其子进程
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                
                # 等待进程终止
                process.wait(timeout=5)
            except:
                # 如果温和终止失败，强制杀死
                process.kill()
                process.wait()
            
            return None, "Timeout", -1
        
        # 添加详细的调试输出
        print(f"✅ Subprocess completed!")
        print(f"📊 Return code: {returncode}")
        print(f"📏 STDOUT length: {len(stdout) if stdout else 0} characters")
        print(f"📏 STDERR length: {len(stderr) if stderr else 0} characters")
        
        if returncode == 0:
            print(f"🎉 Command executed successfully!")
        else:
            print(f"❌ Command failed with return code: {returncode}")
        
        # 显示输出预览
        if stdout:
            print(f"📤 STDOUT preview (first 300 chars):")
            print(f"   {stdout[:300]}...")
        else:
            print(f"📤 STDOUT: Empty or None")
            
        if stderr:
            print(f"⚠️  STDERR preview (first 300 chars):")
            print(f"   {stderr[:300]}...")
        else:
            print(f"⚠️  STDERR: Empty or None")
        
        return stdout, stderr, returncode
        
    except Exception as e:
        print(f"💥 Unexpected error occurred: {str(e)}")
        print(f"💥 Error type: {type(e).__name__}")
        return None, f"Error: {str(e)}", -1


def parse_accuracy(output):
    """从输出中解析准确率"""
    if not output:
        return None
    
    lines = output.split('\n')
    
    # 尝试多种模式匹配准确率
    patterns = [
        r'accuracy[:\s=]+(\d+\.\d+)%?',  # "accuracy: 85.5%" 或 "accuracy = 85.5"
        r'acc[:\s=]+(\d+\.\d+)%?',      # "acc: 85.5%" 或 "acc = 85.5"  
        r'test accuracy[:\s=]+(\d+\.\d+)%?',  # "test accuracy: 85.5%"
        r'final accuracy[:\s=]+(\d+\.\d+)%?',  # "final accuracy: 85.5%"
    ]
    
    import re
    for line in lines:
        line_lower = line.lower()
        for pattern in patterns:
            match = re.search(pattern, line_lower)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    
    # 如果没有找到，尝试提取所有数字，选择最可能的准确率值
    for line in lines:
        if any(keyword in line.lower() for keyword in ['accuracy', 'acc', 'test']):
            numbers = re.findall(r'\d+\.\d+', line)
            for num_str in numbers:
                num = float(num_str)
                if 0 <= num <= 100:  # 准确率应该在0-100之间
                    return num
    
    return None


def compare_methods(dataset='cifar10', num_runs=1):
    """比较不同方法的性能"""
    # 从作者原论文获取的最优参数（可根据实际情况调整）
    methods = {
        'base': {'var1': 0.0, 'var2': 0.0},
        'ovf': {'var1': 0.1, 'var2': 0.05},
        'irs': {'var1': 0.1, 'var2': 0.05},  
        'bnn': {'kl_weight': 1e-5}  # 使用更标准的KL权重
    }
    
    results = {}
    
    for method_name, params in methods.items():
        print(f"\n{'='*50}")
        print(f"Running {method_name.upper()} method")
        print(f"{'='*50}")
        accuracies = []
        
        for run in range(num_runs):
            print(f"\nRun {run + 1}/{num_runs} for {method_name.upper()}")
            success = False
            
            # 统一的num参数 - 确保训练和测试使用相同值
            run_num = str(run + 1)
            
            # 训练阶段
            print(f"  Training {method_name}...")
            if method_name == 'bnn':
                stdout, stderr, returncode = run_experiment(
                    method_name, dataset, 'train', 
                    kl_weight=params['kl_weight'], 
                    num=run_num, mark='1.0'  # 使用统一的run_num
                )
            else:
                stdout, stderr, returncode = run_experiment(
                    method_name, dataset, 'train',
                    var1=params['var1'], var2=params['var2'], 
                    num=run_num, mark='1.0'  # 使用统一的run_num
                )
            
            # 修复：基于returncode判断成功
            if returncode == 0 and stderr != "Timeout":
                print(f"  ✓ Training completed for {method_name}")
                
                # BNN需要额外的权重提取步骤
                if method_name == 'bnn':
                    print(f"  Extracting mean weights from BNN...")
                    stdout_extract, stderr_extract, returncode_extract = run_experiment(
                        method_name, dataset, 'extract_mean',
                        num=run_num, mark='1.0'  # 使用统一的run_num
                    )
                    if returncode_extract == 0 and stderr_extract != "Timeout":
                        print(f"  ✓ Mean weights extracted successfully")
                    else:
                        print(f"  ✗ Failed to extract mean weights")
                        continue
                
                # 测试阶段 - 使用与训练相同的num参数
                print(f"  Testing {method_name}...")
                if method_name == 'bnn':
                    # BNN使用提取的均值权重进行标准测试
                    stdout, stderr, returncode = run_experiment(
                        'bnn', dataset, 'test_mean',
                        kl_weight=params['kl_weight'],  # 🔧 修复：使用kl_weight参数
                        num=run_num, mark='1.0'  # 🔧 关键修复：使用相同的run_num
                    )
                else:
                    # 其他方法正常测试 - 使用相同的num参数
                    stdout, stderr, returncode = run_experiment(
                        method_name, dataset, 'test',
                        var1=params['var1'], var2=params['var2'], 
                        num=run_num, mark='1.0'  # 🔧 关键修复：使用相同的run_num
                    )
                
                # 解析测试结果
                if returncode == 0 and stdout:
                    acc = parse_accuracy(stdout)
                    if acc is not None:
                        accuracies.append(acc)
                        print(f"  ✓ {method_name} Run {run+1} Accuracy: {acc:.2f}%")
                        success = True
                    else:
                        print(f"  ✗ Failed to parse accuracy from output")
                        print(f"  Output preview: {stdout[:200]}...")
                else:
                    print(f"  ✗ No output from test or test failed")
            else:
                print(f"  ✗ Training failed for {method_name}")
                print(f"  Return code: {returncode}")
                if stderr and stderr != "Timeout":
                    print(f"  Error: {stderr[:200]}...")
            
            if not success:
                print(f"  ⚠️  Run {run+1} failed, skipping...")
        
        # 保存结果
        if accuracies:
            results[method_name] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'values': accuracies
            }
            print(f"\n{method_name.upper()} Final Results:")
            print(f"  Mean Accuracy: {np.mean(accuracies):.2f}%")
            print(f"  Std Deviation: {np.std(accuracies):.2f}%")
            print(f"  All Runs: {accuracies}")
        else:
            print(f"\n⚠️  No successful runs for {method_name.upper()}")
    
    return results


def plot_comparison(results, dataset):
    """绘制比较结果"""
    if not results:
        print("No results to plot")
        return None
    
    methods = list(results.keys())
    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    
    # 设置颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(methods)]
    
    # 创建柱状图
    plt.figure(figsize=(12, 8))
    bars = plt.bar(methods, means, yerr=stds, capsize=5, alpha=0.8, color=colors)
    
    # 美化图表
    plt.title(f'Robustness Method Comparison on {dataset.upper()}', fontsize=18, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=16)
    plt.xlabel('Methods', fontsize=16)
    
    # 添加数值标签
    for i, (method, mean, std) in enumerate(zip(methods, means, stds)):
        plt.text(i, mean + std + 1, f'{mean:.2f}±{std:.2f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 添加网格和样式
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, max(means) + max(stds) + 10)
    
    # 调整x轴标签
    method_labels = [m.upper() for m in methods]
    plt.xticks(range(len(methods)), method_labels, fontsize=14)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{dataset}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'comparison_{dataset}.pdf', bbox_inches='tight')
    
    try:
        plt.show()
    except:
        print("Cannot display plot (no GUI), but saved to file")
    
    # 创建详细结果表
    df_data = []
    for method, result in results.items():
        df_data.append({
            'Method': method.upper(),
            'Mean Accuracy (%)': f"{result['mean']:.2f}",
            'Std Deviation (%)': f"{result['std']:.2f}",
            'Best Run (%)': f"{max(result['values']):.2f}",
            'Worst Run (%)': f"{min(result['values']):.2f}",
            'Number of Runs': len(result['values'])
        })
    
    df = pd.DataFrame(df_data)
    print("\n" + "="*80)
    print("DETAILED COMPARISON RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


def format_time(seconds):
    """格式化时间显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours}h {minutes}m {seconds:.1f}s"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare BNN vs Traditional Methods')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['mnist', 'cifar10', 'cifar100', 'tiny'],
                        help='Dataset to use for comparison')
    parser.add_argument('--runs', type=int, default=3, 
                        help='Number of runs for each method (default: 3)')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Timeout per experiment in seconds (default: 2 hours)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NEURAL NETWORK ROBUSTNESS COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Runs per method: {args.runs}")
    print(f"Methods: BASE, OVF, IRS, BNN")
    print(f"Timeout per run: {format_time(args.timeout)}")
    print("="*80)
    
    start_time = time.time()
    
    # 运行比较实验
    print("\nStarting comparison experiment...")
    results = compare_methods(args.dataset, args.runs)
    
    # 绘制和保存结果
    if results:
        print(f"\n✓ Experiment completed! Processing results...")
        df = plot_comparison(results, args.dataset)
        
        # 保存结果到文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f'comparison_results_{args.dataset}_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        
        print(f"\n📊 Results saved to:")
        print(f"  - {csv_filename}")
        print(f"  - comparison_{args.dataset}.png")
        print(f"  - comparison_{args.dataset}.pdf")
        
        # 打印最终总结
        print(f"\n🏆 FINAL RANKING (by mean accuracy):")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (method, result) in enumerate(sorted_results, 1):
            print(f"  {i}. {method.upper()}: {result['mean']:.2f}% ± {result['std']:.2f}%")
    else:
        print("\n❌ No results obtained. Please check your setup and try again.")
        print("\nTroubleshooting tips:")
        print("1. Ensure all required scripts (main_bnn.py, res18_main.py) are present")
        print("2. Check if models are properly saved after training")
        print("3. Verify dataset availability")
        print("4. Consider reducing the number of runs or using a smaller dataset")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total experiment time: {format_time(total_time)}")