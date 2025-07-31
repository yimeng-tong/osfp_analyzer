#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试依赖包是否正确安装
"""

def test_imports():
    """测试所需的包是否可以导入"""
    print("开始测试依赖包...")
    
    try:
        import pandas as pd
        print("✓ pandas 导入成功")
    except ImportError as e:
        print(f"✗ pandas 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy 导入成功")
    except ImportError as e:
        print(f"✗ numpy 导入失败: {e}")
        return False
    
    try:
        import openpyxl
        print("✓ openpyxl 导入成功")
    except ImportError as e:
        print(f"✗ openpyxl 导入失败: {e}")
        return False
    
    try:
        from datetime import datetime, timedelta
        print("✓ datetime 导入成功")
    except ImportError as e:
        print(f"✗ datetime 导入失败: {e}")
        return False
    
    print("所有依赖包测试完成！")
    return True

def create_simple_sample():
    """创建简单的示例数据"""
    try:
        import pandas as pd
        import numpy as np
        import os
        from datetime import datetime, timedelta
        
        print("开始创建示例数据...")
        
        # 创建示例目录
        sample_dir = "sample_data"
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            print(f"创建目录: {sample_dir}")
        
        # 生成简单的示例数据
        n_samples = 20
        
        # 1. SN映射数据
        sn_mapping_data = pd.DataFrame({
            'pcba_sn': [f'PCBA{i:05d}' for i in range(1, n_samples + 1)],
            'shell_sn': [f'SHELL{i:05d}' for i in range(1001, 1001 + n_samples)],
        })
        
        # 2. 测试结果数据
        test_results = []
        for i in range(n_samples):
            test_result = 'PASS' if np.random.random() > 0.3 else 'FAIL'
            test_results.append({
                'shell_sn': f'SHELL{i+1001:05d}',
                'test_result': test_result,
                'optical_power_output_dbm': np.random.normal(-2, 1) if test_result == 'PASS' else np.random.normal(-8, 2)
            })
        
        test_results_data = pd.DataFrame(test_results)
        
        # 3. 贴片数据
        smt_data = pd.DataFrame({
            'sn': [f'PCBA{i:05d}' for i in range(1, n_samples + 1)],
            'paste_volume_ul': np.random.normal(2.5, 0.3, n_samples),
            'placement_accuracy_um': np.random.normal(50, 10, n_samples),
        })
        
        # 保存文件
        datasets = {
            'sn_mapping_sample.xlsx': sn_mapping_data,
            'test_results_sample.xlsx': test_results_data,
            'smt_sample.xlsx': smt_data
        }
        
        for filename, data in datasets.items():
            filepath = os.path.join(sample_dir, filename)
            data.to_excel(filepath, index=False)
            print(f"✓ 已创建: {filepath}")
        
        print(f"\n✓ 所有示例文件已创建在 {sample_dir} 目录中")
        return True
        
    except Exception as e:
        print(f"✗ 创建示例数据时出错: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("800g OSFP 数据分析工具 - 依赖测试")
    print("=" * 50)
    
    # 测试依赖包
    if test_imports():
        print("\n" + "=" * 30)
        # 创建示例数据
        create_simple_sample()
    else:
        print("\n请先安装缺失的依赖包:")
        print("pip install pandas numpy openpyxl")
    
    print("\n" + "=" * 50)
    print("测试完成，按任意键继续...")
    input()  # 防止窗口立即关闭