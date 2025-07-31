#!/bin/bash

echo "正在启动800g OSFP光模块生产数据分析工具..."
echo

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3 未安装"
    echo "请先安装 Python 3.8 或更高版本"
    exit 1
fi

# 检查依赖包是否安装
echo "检查依赖包..."
if ! python3 -c "import flask" &> /dev/null; then
    echo "正在安装依赖包..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误: 依赖包安装失败"
        exit 1
    fi
fi

# 创建必要的目录
mkdir -p uploads results

echo
echo "应用即将启动..."
echo "请在浏览器中访问: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo

# 启动应用
python3 app.py