@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 开始下载数据集
echo ========================================

REM 设置环境变量
set HF_ENDPOINT=https://hf-mirror.com

REM dataset 目录
set dataset_dir=.\datasets

REM 创建目录
if not exist "%dataset_dir%" (
    echo 创建目录: %dataset_dir%
    mkdir "%dataset_dir%"
)

@REM REM 下载预训练数据集
@REM echo.
@REM echo [1/3] 正在下载预训练数据集...

@REM REM 直接调用 modelscope 命令（不是 python -m）
@REM modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir %dataset_dir%

@REM REM 如果上面的命令失败，尝试用 Python API
@REM if errorlevel 1 (
@REM     echo 尝试使用 Python API 下载...
@REM     python -c "from modelscope.hub.api import HubApi; api=HubApi(); api.download(model_id='ddzhu123/seq-monkey', file_name='mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2', local_dir='./datasets')"
@REM )

@REM if errorlevel 1 (
@REM     echo 错误：预训练数据集下载失败！
@REM     echo 请检查网络连接或手动下载
@REM     pause
@REM     exit /b 1
@REM )

@REM REM 解压预训练数据集
@REM echo.
@REM echo [2/3] 正在解压预训练数据集...
@REM set tar_file=%dataset_dir%\mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2
@REM if exist "%tar_file%" (
@REM     tar -xvf "%tar_file%" -C "%dataset_dir%"
@REM     if errorlevel 1 (
@REM         echo 警告：解压可能失败，请检查文件完整性
@REM     )
@REM ) else (
@REM     echo 错误：找不到 %tar_file%
@REM     pause
@REM     exit /b 1
@REM )

REM 下载SFT数据集
REM 下载SFT数据集
echo.
echo [3/3] 正在下载 SFT 数据集...

REM 尝试使用镜像站下载
set HF_ENDPOINT=https://hf-mirror.com

REM 使用 huggingface-cli 下载（新版命令）
huggingface-cli download --repo-type dataset BelleGroup/train_3.5M_CN --local-dir "%dataset_dir%\BelleGroup"

REM 如果上述命令失败，尝试直接下载（使用官方源）
if errorlevel 1 (
    echo 镜像站下载失败，尝试使用官方源...
    set HF_ENDPOINT=
    huggingface-cli download --repo-type dataset BelleGroup/train_3.5M_CN --local-dir "%dataset_dir%\BelleGroup"
)

if errorlevel 1 (
    echo 错误：SFT 数据集下载失败！
    echo 请尝试手动下载：
    echo   1. 访问 https://hf-mirror.com/datasets/BelleGroup/train_3.5M_CN
    echo   2. 下载所有文件到 %dataset_dir%\BelleGroup\
    pause
    exit /b 1
)

echo SFT 数据集下载完成！✅