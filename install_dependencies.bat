@echo off
echo ====================================================
echo 800g OSFP光模块生产数据分析工具 - 依赖包安装
echo ====================================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: Python 未安装或未添加到系统PATH中
    echo 请先从 https://python.org 下载并安装 Python 3.8 或更高版本
    echo.
    pause
    exit /b 1
)

echo 检测到Python版本:
python --version
echo.

echo 正在升级pip...
python -m pip install --upgrade pip
echo.

echo 正在安装项目依赖包...
echo 这可能需要几分钟时间，请耐心等待...
echo.

python -m pip install pandas==2.0.3
if errorlevel 1 (
    echo 安装pandas失败，尝试使用国内镜像...
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas==2.0.3
)

python -m pip install numpy==1.24.3
if errorlevel 1 (
    echo 安装numpy失败，尝试使用国内镜像...
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.24.3
)

python -m pip install openpyxl==3.1.2
if errorlevel 1 (
    echo 安装openpyxl失败，尝试使用国内镜像...
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openpyxl==3.1.2
)

echo.
echo 测试依赖包安装情况...
python test_dependencies.py

echo.
echo ====================================================
echo 安装完成！现在您可以运行示例数据生成器了。
echo ====================================================
pause