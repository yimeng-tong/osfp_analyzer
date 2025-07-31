#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
800g OSFP光模块生产数据分析工具
AI驱动的根本原因分析(RCA)系统
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import requests
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 设置matplotlib后端
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
import logging
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

# 加载配置文件
def load_config():
    """加载配置文件"""
    config_path = 'config.json'
    default_config = {
        "app": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False,
            "max_content_length": 104857600
        },
        "openai": {
            "api_key": "",
            "api_url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "analysis": {
            "correlation_threshold": 0.5,
            "t_test_alpha": 0.05,
            "random_forest_n_estimators": 100,
            "test_size": 0.3,
            "random_state": 42
        },
        "column_mapping": {
            "sn_columns": ["sn", "SN", "序列号", "serial_number"],
            "pcba_sn_columns": ["pcba_sn", "PCBA_SN", "pcba序列号"],
            "shell_sn_columns": ["shell_sn", "SHELL_SN", "壳体sn", "shell序列号"],
            "test_result_columns": ["test_result", "TEST_RESULT", "测试结果", "result", "RESULT"]
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info("配置文件加载成功")
                return config
        else:
            logger.warning("配置文件不存在，使用默认配置")
            return default_config
    except Exception as e:
        logger.error(f"配置文件加载失败: {str(e)}")
        return default_config

# 加载配置
config = load_config()

# 配置Flask应用
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = config['app']['max_content_length']

# 创建必要的目录
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class DataAnalyzer:
    def __init__(self):
        self.data = {}
        self.analysis_results = {}
        self.column_mapping = config['column_mapping']
        
    def find_column(self, df, possible_names):
        """智能查找列名"""
        for col in df.columns:
            if col.lower().strip() in [name.lower() for name in possible_names]:
                return col
        return None
    
    def standardize_columns(self, df, section):
        """标准化列名"""
        df_copy = df.copy()
        
        if section == 'sn_mapping':
            # 查找PCBA SN列
            pcba_col = self.find_column(df, self.column_mapping['pcba_sn_columns'])
            if pcba_col:
                df_copy = df_copy.rename(columns={pcba_col: 'pcba_sn'})
            
            # 查找壳体SN列
            shell_col = self.find_column(df, self.column_mapping['shell_sn_columns'])
            if shell_col:
                df_copy = df_copy.rename(columns={shell_col: 'shell_sn'})
                
        elif section == 'test_results':
            # 查找壳体SN列
            shell_col = self.find_column(df, self.column_mapping['shell_sn_columns'])
            if shell_col:
                df_copy = df_copy.rename(columns={shell_col: 'shell_sn'})
            
            # 查找测试结果列
            result_col = self.find_column(df, self.column_mapping['test_result_columns'])
            if result_col:
                df_copy = df_copy.rename(columns={result_col: 'test_result'})
                
        else:  # 生产数据 (smt, wire_bonding, lens_coupling)
            # 查找SN列
            sn_col = self.find_column(df, self.column_mapping['sn_columns'])
            if sn_col:
                df_copy = df_copy.rename(columns={sn_col: 'sn'})
        
        return df_copy
    
    def load_excel_files(self, files_dict):
        """加载并整合Excel文件"""
        for section, files in files_dict.items():
            section_data = []
            for file_path in files:
                try:
                    # 尝试读取Excel文件
                    if file_path.endswith('.xlsx'):
                        df = pd.read_excel(file_path, engine='openpyxl')
                    else:
                        df = pd.read_excel(file_path)
                    
                    # 标准化列名
                    df = self.standardize_columns(df, section)
                    
                    # 数据清洗
                    df = self.clean_data(df, section)
                    
                    section_data.append(df)
                    logger.info(f"成功加载文件: {file_path}, 数据行数: {len(df)}")
                    
                except Exception as e:
                    logger.error(f"加载文件失败 {file_path}: {str(e)}")
                    continue
            
            if section_data:
                # 合并同一版块的多个文件
                self.data[section] = pd.concat(section_data, ignore_index=True)
                logger.info(f"版块 {section} 总数据行数: {len(self.data[section])}")
    
    def clean_data(self, df, section):
        """数据清洗"""
        df_copy = df.copy()
        
        # 移除完全空白的行
        df_copy = df_copy.dropna(how='all')
        
        # 移除重复行
        initial_rows = len(df_copy)
        df_copy = df_copy.drop_duplicates()
        if len(df_copy) < initial_rows:
            logger.info(f"移除了 {initial_rows - len(df_copy)} 行重复数据")
        
        # 数值列的异常值处理
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 使用IQR方法检测异常值
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 记录异常值数量
            outliers = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)]
            if len(outliers) > 0:
                logger.warning(f"列 {col} 发现 {len(outliers)} 个异常值")
        
        return df_copy
                
    def merge_data_by_sn(self):
        """通过SN码关联生产数据和测试数据"""
        if 'sn_mapping' not in self.data or 'test_results' not in self.data:
            logger.error("缺少SN映射数据或测试结果数据")
            return None
            
        # 获取SN映射关系
        sn_mapping = self.data['sn_mapping']
        test_data = self.data['test_results']
        
        # 检查必要的列是否存在
        required_mapping_cols = ['pcba_sn', 'shell_sn']
        required_test_cols = ['shell_sn']
        
        for col in required_mapping_cols:
            if col not in sn_mapping.columns:
                logger.error(f"SN映射数据缺少必要列: {col}")
                return None
                
        for col in required_test_cols:
            if col not in test_data.columns:
                logger.error(f"测试结果数据缺少必要列: {col}")
                return None
        
        logger.info(f"开始数据关联: SN映射{len(sn_mapping)}行, 测试数据{len(test_data)}行")
        
        # 从测试数据开始，通过shell_sn关联到mapping
        merged_data = pd.merge(
            test_data, sn_mapping,
            on='shell_sn',
            how='inner'
        )
        
        logger.info(f"测试数据与SN映射关联后: {len(merged_data)}行")
        
        # 通过PCBA SN关联生产数据
        for section in ['smt', 'wire_bonding', 'lens_coupling']:
            if section in self.data:
                production_data = self.data[section]
                
                if 'sn' not in production_data.columns:
                    logger.warning(f"{section}数据缺少SN列，跳过")
                    continue
                
                initial_rows = len(merged_data)
                merged_data = pd.merge(
                    merged_data, production_data,
                    left_on='pcba_sn', right_on='sn',
                    how='left', suffixes=('', f'_{section}')
                )
                
                logger.info(f"关联{section}数据后: {len(merged_data)}行")
        
        # 移除重复的SN列
        columns_to_drop = [col for col in merged_data.columns if col.endswith('_smt') or col.endswith('_wire_bonding') or col.endswith('_lens_coupling')]
        if 'sn' in merged_data.columns:
            columns_to_drop.append('sn')
        
        merged_data = merged_data.drop(columns=[col for col in columns_to_drop if col in merged_data.columns])
        
        logger.info(f"最终合并数据: {len(merged_data)}行, {len(merged_data.columns)}列")
        return merged_data
    
    def correlation_analysis(self, data):
        """相关性分析"""
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) < 2:
                logger.warning("数值列少于2列，无法进行相关性分析")
                return None, None
            
            # 移除常数列（方差为0的列）
            valid_columns = []
            for col in numeric_columns:
                if data[col].var() > 1e-10:  # 避免除零错误
                    valid_columns.append(col)
            
            if len(valid_columns) < 2:
                logger.warning("有效数值列少于2列，无法进行相关性分析")
                return None, None
            
            correlation_matrix = data[valid_columns].corr()
            
            # 生成热力图
            plt.figure(figsize=(max(12, len(valid_columns)), max(10, len(valid_columns))))
            
            # 创建mask来隐藏上三角
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(
                correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8}
            )
            
            plt.title('生产参数相关性矩阵 (Correlation Matrix)', fontsize=16, pad=20)
            plt.xlabel('参数', fontsize=12)
            plt.ylabel('参数', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            correlation_plot_path = os.path.join(app.config['RESULTS_FOLDER'], 'correlation_matrix.png')
            plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 识别强相关关系
            strong_correlations = []
            threshold = config['analysis']['correlation_threshold']
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        strong_correlations.append({
                            'var1': correlation_matrix.columns[i],
                            'var2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            logger.info(f"发现 {len(strong_correlations)} 对强相关关系 (|r| >= {threshold})")
            
            # 确保相关性数据是JSON可序列化的
            def convert_correlation_matrix(matrix):
                """转换相关性矩阵为JSON可序列化格式"""
                result = {}
                for col in matrix.columns:
                    result[col] = {}
                    for row in matrix.index:
                        value = matrix.loc[row, col]
                        if pd.isna(value):
                            result[col][row] = None
                        else:
                            result[col][row] = float(value)
                return result
            
            # 确保强相关关系数据是JSON可序列化的
            serializable_correlations = []
            for corr in strong_correlations:
                serializable_correlations.append({
                    'var1': str(corr['var1']),
                    'var2': str(corr['var2']),
                    'correlation': float(corr['correlation'])
                })
            
            return {
                'matrix': convert_correlation_matrix(correlation_matrix),
                'strong_correlations': serializable_correlations
            }, correlation_plot_path
            
        except Exception as e:
            logger.error(f"相关性分析失败: {str(e)}")
            return None, None
    
    def t_test_analysis(self, data, target_column):
        """T检验分析"""
        try:
            results = {}
            
            if target_column not in data.columns:
                logger.error(f"目标列 {target_column} 不存在")
                return results
            
            # 检查测试结果的值分布
            value_counts = data[target_column].value_counts()
            logger.info(f"测试结果分布: {value_counts.to_dict()}")
            
            # 更灵活的PASS/FAIL识别
            pass_values_possible = ['PASS', 'Pass', 'pass', '通过', 'OK', 'ok', '1', 1, True]
            fail_values_possible = ['FAIL', 'Fail', 'fail', '失败', 'NG', 'ng', '0', 0, False]
            
            pass_mask = data[target_column].isin(pass_values_possible)
            fail_mask = data[target_column].isin(fail_values_possible)
            
            pass_data = data[pass_mask]
            fail_data = data[fail_mask]
            
            if len(pass_data) == 0 or len(fail_data) == 0:
                logger.warning(f"PASS样本数: {len(pass_data)}, FAIL样本数: {len(fail_data)}")
                return results
            
            logger.info(f"T检验分析: PASS样本{len(pass_data)}个, FAIL样本{len(fail_data)}个")
            
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            alpha = config['analysis']['t_test_alpha']
            
            for col in numeric_columns:
                if col != target_column:
                    pass_values = pass_data[col].dropna()
                    fail_values = fail_data[col].dropna()
                    
                    if len(pass_values) >= 5 and len(fail_values) >= 5:  # 最小样本要求
                        try:
                            t_stat, p_value = ttest_ind(pass_values, fail_values)
                            
                            # 计算效应量 (Cohen's d)
                            pooled_std = np.sqrt(((len(pass_values) - 1) * pass_values.var() + 
                                                (len(fail_values) - 1) * fail_values.var()) / 
                                               (len(pass_values) + len(fail_values) - 2))
                            
                            cohens_d = (pass_values.mean() - fail_values.mean()) / pooled_std if pooled_std > 0 else 0
                            
                            results[col] = {
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': bool(p_value < alpha),  # 确保是Python布尔值
                                'pass_mean': float(pass_values.mean()),
                                'fail_mean': float(fail_values.mean()),
                                'pass_std': float(pass_values.std()),
                                'fail_std': float(fail_values.std()),
                                'pass_count': int(len(pass_values)),
                                'fail_count': int(len(fail_values)),
                                'cohens_d': float(cohens_d),
                                'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
                            }
                        except Exception as e:
                            logger.warning(f"列 {col} T检验失败: {str(e)}")
                            continue
                    else:
                        logger.warning(f"列 {col} 样本量不足: PASS={len(pass_values)}, FAIL={len(fail_values)}")
            
            logger.info(f"完成T检验分析，共分析 {len(results)} 个参数")
            return results
            
        except Exception as e:
            logger.error(f"T检验分析失败: {str(e)}")
            return {}
    
    def random_forest_analysis(self, data, target_column):
        """随机森林分析"""
        try:
            if target_column not in data.columns:
                logger.error(f"目标列 {target_column} 不存在")
                return None, None
            
            # 准备特征数据
            X = data.select_dtypes(include=[np.number]).copy()
            
            # 移除目标列（如果它是数值型）
            if target_column in X.columns:
                X = X.drop(columns=[target_column])
            
            if len(X.columns) == 0:
                logger.warning("没有可用的数值特征进行随机森林分析")
                return None, None
            
            # 处理缺失值
            X = X.fillna(X.mean())
            
            # 编码目标变量
            y = data[target_column]
            pass_values_possible = ['PASS', 'Pass', 'pass', '通过', 'OK', 'ok', '1', 1, True]
            y_encoded = y.isin(pass_values_possible).astype(int)
            
            # 检查类别平衡
            class_counts = pd.Series(y_encoded).value_counts()
            logger.info(f"随机森林分析 - 类别分布: {class_counts.to_dict()}")
            
            if len(class_counts) < 2:
                logger.warning("只有一个类别，无法进行分类分析")
                return None, None
            
            if len(X) < 20:  # 增加最小样本要求
                logger.warning(f"样本量不足: {len(X)}，建议至少20个样本")
                return None, None
            
            # 分割数据
            test_size = min(config['analysis']['test_size'], 0.5)  # 确保训练集有足够样本
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=test_size, 
                random_state=config['analysis']['random_state'],
                stratify=y_encoded if len(class_counts) > 1 and min(class_counts) > 1 else None
            )
            
            logger.info(f"训练集: {len(X_train)}样本, 测试集: {len(X_test)}样本")
            
            # 训练模型
            rf = RandomForestClassifier(
                n_estimators=config['analysis']['random_forest_n_estimators'],
                random_state=config['analysis']['random_state'],
                class_weight='balanced'  # 处理类别不平衡
            )
            rf.fit(X_train, y_train)
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 生成特征重要性图
            plt.figure(figsize=(12, max(8, len(feature_importance) * 0.4)))
            top_features = feature_importance.head(min(20, len(feature_importance)))
            
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('特征重要性 (Feature Importance)')
            plt.title('随机森林特征重要性分析', fontsize=16, pad=20)
            plt.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, (idx, row) in enumerate(top_features.iterrows()):
                plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                        va='center', fontsize=9)
            
            plt.tight_layout()
            rf_plot_path = os.path.join(app.config['RESULTS_FOLDER'], 'random_forest_importance.png')
            plt.savefig(rf_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 预测和评估
            y_pred = rf.predict(X_test)
            y_pred_proba = rf.predict_proba(X_test)
            
            # 详细的分类报告
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            
            # 模型性能指标
            model_score = rf.score(X_test, y_test)
            
            # 特征选择建议
            important_features = feature_importance[feature_importance['importance'] > 0.01]
            
            # 确保所有数据都是JSON可序列化的
            def make_json_serializable(obj):
                """递归转换numpy类型为Python原生类型"""
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: make_json_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                else:
                    return obj
            
            results = {
                'feature_importance': make_json_serializable(feature_importance.to_dict('records')),
                'important_features_count': int(len(important_features)),
                'classification_report': make_json_serializable(classification_rep),
                'model_score': float(model_score),
                'feature_count': int(len(X.columns)),
                'train_samples': int(len(X_train)),
                'test_samples': int(len(X_test))
            }
            
            logger.info(f"随机森林分析完成 - 模型准确率: {model_score:.3f}")
            
            return results, rf_plot_path
            
        except Exception as e:
            logger.error(f"随机森林分析失败: {str(e)}")
            return None, None
    
    def generate_analysis_report(self, merged_data):
        """生成综合分析报告"""
        try:
            if merged_data is None or len(merged_data) == 0:
                logger.error("数据不足，无法进行分析")
                return "数据不足，无法进行分析"
            
            logger.info(f"开始生成分析报告，数据样本数: {len(merged_data)}")
            
            # 查找测试结果列
            target_col = None
            possible_target_cols = ['test_result', 'TEST_RESULT', '测试结果', 'result', 'RESULT']
            
            for col in merged_data.columns:
                if col in possible_target_cols:
                    target_col = col
                    break
            
            if target_col is None:
                # 如果没找到标准列名，查找包含关键词的列
                for col in merged_data.columns:
                    if any(keyword in col.lower() for keyword in ['test', 'result', '测试', '结果']):
                        target_col = col
                        break
            
            if target_col is None:
                logger.warning("未找到测试结果列，将使用最后一列作为目标变量")
                target_col = merged_data.columns[-1]
            
            logger.info(f"使用 {target_col} 作为目标变量")
            
            # 执行各项分析
            correlation_results, corr_plot = self.correlation_analysis(merged_data)
            t_test_results = self.t_test_analysis(merged_data, target_col)
            rf_results, rf_plot = self.random_forest_analysis(merged_data, target_col)
            
            # 数据质量评估
            data_quality = self.assess_data_quality(merged_data)
            
            # 存储分析结果，确保所有数据都是JSON可序列化的
            self.analysis_results = {
                'data_summary': {
                    'total_samples': int(len(merged_data)),
                    'total_columns': int(len(merged_data.columns)),
                    'numeric_columns': int(len(merged_data.select_dtypes(include=[np.number]).columns)),
                    'target_column': str(target_col),
                    'missing_data_percent': float((merged_data.isnull().sum().sum() / (len(merged_data) * len(merged_data.columns))) * 100)
                },
                'data_quality': data_quality,
                'correlation_results': correlation_results,
                'correlation_plot': corr_plot,
                't_test_results': t_test_results,
                'random_forest_results': rf_results,
                'rf_plot': rf_plot,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info("分析报告生成完成")
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"生成分析报告失败: {str(e)}")
            return f"生成分析报告失败: {str(e)}"
    
    def assess_data_quality(self, data):
        """评估数据质量"""
        try:
            quality_report = {
                'completeness': {},
                'consistency': {},
                'outliers': {}
            }
            
            # 完整性评估
            for col in data.columns:
                missing_rate = data[col].isnull().sum() / len(data)
                quality_report['completeness'][col] = {
                    'missing_rate': float(missing_rate),
                    'quality_level': 'good' if missing_rate < 0.05 else 'warning' if missing_rate < 0.2 else 'poor'
                }
            
            # 一致性评估（数值列）
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else float('inf')
                    quality_report['consistency'][col] = {
                        'coefficient_of_variation': float(cv),
                        'stability': 'stable' if cv < 0.3 else 'moderate' if cv < 1.0 else 'unstable'
                    }
            
            return quality_report
            
        except Exception as e:
            logger.error(f"数据质量评估失败: {str(e)}")
            return {}

analyzer = DataAnalyzer()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """文件上传处理"""
    try:
        uploaded_files = {}
        sections = ['smt', 'wire_bonding', 'lens_coupling', 'sn_mapping', 'test_results']
        
        for section in sections:
            files = request.files.getlist(f'{section}_files')
            section_files = []
            
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{section}_{filename}")
                    file.save(file_path)
                    section_files.append(file_path)
            
            if section_files:
                uploaded_files[section] = section_files
        
        if not uploaded_files:
            return jsonify({'error': '没有上传任何文件'}), 400
            
        # 加载和分析数据
        analyzer.load_excel_files(uploaded_files)
        merged_data = analyzer.merge_data_by_sn()
        analysis_results = analyzer.generate_analysis_report(merged_data)
        
        # 确保返回结果完全JSON可序列化
        def deep_json_serialize(obj):
            """深度转换所有numpy类型为Python原生类型"""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(key): deep_json_serialize(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [deep_json_serialize(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        safe_analysis_results = deep_json_serialize(analysis_results)
        
        return jsonify({
            'success': True,
            'message': '文件上传和分析完成',
            'analysis_results': safe_analysis_results
        })
        
    except Exception as e:
        return jsonify({'error': f'处理文件时出错: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_with_ai():
    """AI分析接口"""
    try:
        logger.info("=== AI分析函数开始执行 ===")
        data = request.json
        logger.info(f"接收到请求数据: {data}")
        custom_prompt = data.get('custom_prompt', config.get('system_prompt', ''))
        logger.info(f"自定义提示词: {custom_prompt[:100]}...")
        
        # 检查是否有分析结果，或者是否为测试连接
        if not analyzer.analysis_results:
            # 如果是简单的测试请求，创建一个测试响应
            if custom_prompt.startswith("你是一个测试助手"):
                logger.info("检测到AI连接测试请求")
                test_summary = {
                    'test_mode': True,
                    'message': '这是一个AI连接测试'
                }
                analysis_summary = test_summary
            else:
                return jsonify({'error': '没有可分析的数据，请先上传文件'}), 400
        else:
            # 准备发送给AI的详细数据摘要
            logger.info("开始构建分析总结...")
            logger.info(f"analyzer.analysis_results类型: {type(analyzer.analysis_results)}")
            logger.info(f"analyzer.analysis_results的键: {list(analyzer.analysis_results.keys()) if analyzer.analysis_results else '无'}")
            
            analysis_summary = {
                'data_summary': analyzer.analysis_results.get('data_summary', {}),
                'data_quality': analyzer.analysis_results.get('data_quality', {}),
                'correlation_analysis': {
                    'completed': analyzer.analysis_results.get('correlation_results') is not None,
                    'strong_correlations': analyzer.analysis_results.get('correlation_results', {}).get('strong_correlations', [])
                },
                't_test_analysis': {
                    'parameters_analyzed': len(analyzer.analysis_results.get('t_test_results', {})),
                    'significant_parameters': [
                        param for param, result in analyzer.analysis_results.get('t_test_results', {}).items()
                        if result.get('significant', False)
                    ]
                },
                'random_forest_analysis': {
                    'completed': analyzer.analysis_results.get('random_forest_results') is not None,
                    'model_accuracy': analyzer.analysis_results.get('random_forest_results', {}).get('model_score', 0),
                    'top_features': [
                        feature['feature'] for feature in 
                        analyzer.analysis_results.get('random_forest_results', {}).get('feature_importance', [])[:5]
                    ]
                }
            }
            logger.info("分析总结构建完成")

        
        # 调用OpenAI API
        openai_config = config.get('openai', {})
        api_key = openai_config.get('api_key', '')
        
        logger.info(f"AI分析开始 - API Key存在: {bool(api_key)}")
        logger.info(f"API URL: {openai_config.get('api_url', 'https://api.openai.com/v1/chat/completions')}")
        logger.info(f"模型: {openai_config.get('model', 'gpt-4')}")
        
        if not api_key:
            logger.warning("AI分析失败：API Key未配置")
            # 确保分析结果JSON可序列化
            def deep_json_serialize(obj):
                """深度转换所有numpy类型为Python原生类型"""
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(key): deep_json_serialize(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [deep_json_serialize(item) for item in obj]
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            safe_analysis_results = deep_json_serialize(analyzer.analysis_results)
            
            return jsonify({
                'success': False,
                'ai_analysis': '请先配置OpenAI API密钥以使用AI分析功能。',
                'analysis_results': safe_analysis_results
            })
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 构建不同的payload用于测试模式和正常分析
        if analysis_summary.get('test_mode'):
            payload = {
                'model': openai_config.get('model', 'gpt-4'),
                'temperature': 0.1,
                'max_tokens': 50,
                'messages': [
                    {'role': 'system', 'content': custom_prompt},
                    {'role': 'user', 'content': '请简单回复"测试成功"以确认连接正常。'}
                ]
            }
        else:
            payload = {
                'model': openai_config.get('model', 'gpt-4'),
                'temperature': openai_config.get('temperature', 0.7),
                'max_tokens': openai_config.get('max_tokens', 2000),
                'messages': [
                    {'role': 'system', 'content': custom_prompt},
                    {'role': 'user', 'content': f'请基于以下800g OSFP光模块生产数据分析结果进行专业的RCA根本原因分析：\n\n{json.dumps(analysis_summary, ensure_ascii=False, indent=2)}\n\n请提供详细的分析报告，包含具体的改进建议和预防措施。'}
                ]
            }
        
        logger.info("开始调用OpenAI API...")
        logger.info(f"Payload大小: {len(json.dumps(payload, ensure_ascii=False))}")
        
        try:
            base_url = openai_config.get('api_url', 'https://api.openai.com/v1/chat/completions')
            
            # 确保第三方API有正确的端点
            if not base_url.endswith('/chat/completions') and not base_url.endswith('/v1/chat/completions'):
                if base_url.endswith('/'):
                    api_url = base_url + 'v1/chat/completions'
                else:
                    api_url = base_url + '/v1/chat/completions'
            else:
                api_url = base_url
                
            logger.info(f"正在请求API: {api_url}")
            
            # 为第三方API添加额外的请求头
            enhanced_headers = headers.copy()
            enhanced_headers.update({
                'User-Agent': 'OSFP-Analyzer/1.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            })
            
            logger.info(f"请求头: {enhanced_headers}")
            logger.info(f"请求payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            
            response = requests.post(
                api_url,
                headers=enhanced_headers, 
                json=payload,
                timeout=60,
                verify=True  # 确保SSL验证
            )
            
            logger.info(f"API响应状态码: {response.status_code}")
            logger.info(f"响应头: {dict(response.headers)}")
            logger.info(f"响应内容类型: {response.headers.get('content-type', 'unknown')}")
            logger.info(f"响应内容前500字符: {response.text[:500]}")
            
            if response.status_code == 200:
                try:
                    # 检查响应是否为空
                    if not response.text.strip():
                        logger.error("API返回空响应")
                        ai_analysis = "API返回空响应，请检查API配置和余额"
                    else:
                        ai_response = response.json()
                        # 检查响应格式是否符合OpenAI标准
                        if 'choices' in ai_response and len(ai_response['choices']) > 0:
                            ai_analysis = ai_response['choices'][0]['message']['content']
                            logger.info(f"AI分析完成，响应长度: {len(ai_analysis)}")
                            logger.info(f"AI分析预览: {ai_analysis[:200]}...")
                        elif 'error' in ai_response:
                            logger.error(f"API返回错误: {ai_response['error']}")
                            ai_analysis = f"API错误: {ai_response['error']}"
                        else:
                            logger.error(f"未知的响应格式: {ai_response}")
                            ai_analysis = f"未知的响应格式: {str(ai_response)[:200]}"
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析失败: {str(e)}")
                    logger.error(f"响应内容: {response.text[:1000]}")
                    ai_analysis = f"第三方API返回非JSON格式内容。响应内容: {response.text[:200]}"
            else:
                logger.error(f"OpenAI API调用失败: {response.status_code}")
                logger.error(f"错误响应: {response.text[:500]}")
                ai_analysis = f"API调用失败: {response.status_code} - {response.text[:500]}"
                
        except requests.exceptions.Timeout:
            logger.error("OpenAI API调用超时")
            ai_analysis = "AI分析超时，请稍后重试。"
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求失败: {str(e)}")
            ai_analysis = f"网络请求失败: {str(e)}"
        except Exception as e:
            logger.error(f"AI分析过程中出现未知错误: {str(e)}")
            ai_analysis = f"AI分析过程中出现错误: {str(e)}"
        
        # 确保分析结果JSON可序列化
        def deep_json_serialize(obj):
            """深度转换所有numpy类型为Python原生类型"""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(key): deep_json_serialize(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [deep_json_serialize(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        # 处理分析结果，测试模式时可能为空
        safe_analysis_results = deep_json_serialize(analyzer.analysis_results) if analyzer.analysis_results else {}
        
        logger.info(f"准备返回AI分析结果，ai_analysis长度: {len(ai_analysis) if ai_analysis else 0}")
        logger.info(f"AI分析结果预览: {ai_analysis[:100] if ai_analysis else 'None'}...")
        
        # 尝试将完整的AI分析结果保存到文件以供调试
        try:
            import os
            from datetime import datetime
            debug_dir = 'debug_output'
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # 简单的文本保存
            debug_content = f"""=== AI分析调试信息 ===
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
AI分析内容长度: {len(ai_analysis) if ai_analysis else 0}
AI分析内容是否为空: {not bool(ai_analysis)}
=== AI分析内容开始 ===
{ai_analysis if ai_analysis else '无内容'}
=== AI分析内容结束 ===
"""
            
            with open(os.path.join(debug_dir, 'debug_log.txt'), 'w', encoding='utf-8') as f:
                f.write(debug_content)
            
            logger.info(f"调试信息已保存到 debug_output/debug_log.txt")
        except Exception as save_error:
            logger.error(f"保存调试信息失败: {save_error}")
        
        return jsonify({
            'success': True,
            'ai_analysis': ai_analysis,
            'analysis_results': safe_analysis_results
        })
        
    except Exception as e:
        import traceback
        logger.error(f"AI分析时出错: {str(e)}")
        logger.error(f"错误类型: {type(e).__name__}")
        logger.error(f"完整错误堆栈: {traceback.format_exc()}")
        return jsonify({'error': f'AI分析时出错: {str(e)}'}), 500

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """处理配置的读取和更新"""
    if request.method == 'GET':
        # 返回当前配置（隐藏敏感信息）
        try:
            safe_config = config.copy()
            # 隐藏API key的敏感部分
            if 'openai' in safe_config and 'api_key' in safe_config['openai'] and safe_config['openai']['api_key']:
                api_key = safe_config['openai']['api_key']
                if len(api_key) > 8:
                    safe_config['openai']['api_key'] = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
            
            return jsonify({'success': True, 'config': safe_config})
        except Exception as e:
            logger.error(f'获取配置时出错: {str(e)}')
            return jsonify({'error': f'获取配置时出错: {str(e)}'}), 500
    
    elif request.method == 'POST':
        # 更新配置
        try:
            data = request.json
            
            # 更新OpenAI配置
            if 'openai_api_key' in data:
                config['openai']['api_key'] = data['openai_api_key']
            if 'openai_api_url' in data:
                config['openai']['api_url'] = data['openai_api_url']
            if 'model' in data:
                config['openai']['model'] = data['model']
            if 'system_prompt' in data:
                config['system_prompt'] = data['system_prompt']
            
            # 保存配置到文件
            try:
                with open('config.json', 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                logger.info("配置已保存到文件")
            except Exception as e:
                logger.warning(f"保存配置文件失败: {str(e)}")
            
            return jsonify({'success': True, 'message': '配置已更新'})
            
        except Exception as e:
            logger.error(f'更新配置时出错: {str(e)}')
            return jsonify({'error': f'更新配置时出错: {str(e)}'}), 500

@app.route('/test_api_key', methods=['POST'])
def test_api_key():
    """测试OpenAI API Key的有效性"""
    try:
        data = request.json
        api_key = data.get('api_key', '')
        api_url = data.get('api_url', 'https://api.openai.com/v1/chat/completions')
        model = data.get('model', 'gpt-4')
        
        if not api_key:
            return jsonify({'success': False, 'error': 'API Key不能为空'})
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 发送一个简单的测试请求
        test_payload = {
            'model': model,
            'messages': [
                {'role': 'user', 'content': 'Hello, this is a test message.'}
            ],
            'max_tokens': 5,
            'temperature': 0.1
        }
        
        # 确保第三方API有正确的端点
        if not api_url.endswith('/chat/completions') and not api_url.endswith('/v1/chat/completions'):
            if api_url.endswith('/'):
                api_url = api_url + 'v1/chat/completions'
            else:
                api_url = api_url + '/v1/chat/completions'
        
        # 为第三方API添加额外的请求头
        enhanced_headers = headers.copy()
        enhanced_headers.update({
            'User-Agent': 'OSFP-Analyzer/1.0',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        logger.info(f"测试API端点: {api_url}")
        logger.info(f"测试请求头: {enhanced_headers}")
        logger.info(f"测试载荷: {json.dumps(test_payload, ensure_ascii=False, indent=2)}")
        
        response = requests.post(
            api_url,
            headers=enhanced_headers,
            json=test_payload,
            timeout=30,
            verify=True
        )
        
        logger.info(f"测试响应状态码: {response.status_code}")
        logger.info(f"测试响应内容: {response.text[:500]}")
        
        if response.status_code == 200:
            logger.info("API Key测试成功")
            return jsonify({'success': True, 'message': 'API Key有效'})
        elif response.status_code == 401:
            return jsonify({'success': False, 'error': 'API Key无效或已过期'})
        elif response.status_code == 429:
            return jsonify({'success': False, 'error': 'API配额已耗尽或请求过于频繁'})
        elif response.status_code == 403:
            return jsonify({'success': False, 'error': 'API Key权限不足'})
        else:
            return jsonify({'success': False, 'error': f'API调用失败: HTTP {response.status_code}'})
            
    except requests.exceptions.Timeout:
        return jsonify({'success': False, 'error': 'API请求超时'})
    except requests.exceptions.ConnectionError:
        return jsonify({'success': False, 'error': '无法连接到API服务器'})
    except requests.exceptions.RequestException as e:
        return jsonify({'success': False, 'error': f'网络请求失败: {str(e)}'})
    except Exception as e:
        logger.error(f'测试API Key时出错: {str(e)}')
        return jsonify({'success': False, 'error': f'测试API Key时出错: {str(e)}'})

@app.route('/download/<filename>')
def download_file(filename):
    """下载生成的图片文件"""
    try:
        return send_file(
            os.path.join(app.config['RESULTS_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return jsonify({'error': f'下载文件时出错: {str(e)}'}), 500

if __name__ == '__main__':
    # 获取应用配置
    app_config = config.get('app', {})
    
    # 生成示例数据（如果不存在）
    if not os.path.exists('sample_data'):
        try:
            logger.info("创建示例数据...")
            from sample_data_format import create_sample_data
            create_sample_data()
        except Exception as e:
            logger.warning(f"创建示例数据失败: {str(e)}")
    
    logger.info("启动800g OSFP光模块生产数据分析工具")
    logger.info(f"访问地址: http://{app_config.get('host', '0.0.0.0')}:{app_config.get('port', 5000)}")
    
    app.run(
        debug=app_config.get('debug', False),
        host=app_config.get('host', '0.0.0.0'),
        port=app_config.get('port', 5000)
    )