# WeFinance Copilot

**[English](./README.md)** | 中文

AI驱动的智能财务助理 - 2025深圳国际金融科技大赛参赛项目

## 🚀 在线演示

**立即体验**：https://wefinance-copilot.streamlit.app

> 无需安装，直接上传账单图片测试OCR识别、智能顾问聊天、投资建议等核心功能

## 项目简介

WeFinance Copilot通过**Vision LLM + 生成式AI**，将纸质/电子账单转化为智能财务分析，提供：
- 📸 **智能账单识别**：拍照上传，GPT-4o Vision 直接识别提取交易记录（100%准确率）
- 💬 **对话式财务顾问**：自然语言问答，个性化理财建议
- 🔍 **可解释AI推荐**：透明展示决策逻辑，建立用户信任
- ⚠️ **主动异常检测**：自动发现异常支出并提醒

**技术亮点**：
- **Vision LLM架构**：GPT-4o Vision 一步到位，识别准确率100%（vs 传统OCR 0%）
- **隐私保护**：图片仅通过API传输，不做持久化存储
- 轻量化设计（10天开发周期，无数据库依赖）
- 中英文界面一键切换（完整 i18n 方案 + 缓存）

## 快速开始

> 💡 **首次使用？** 推荐使用自动安装脚本，详见 [Conda环境管理指南](./docs/CONDA_GUIDE.md)

### 1. 环境准备（三种方式）

#### 方式A：自动安装脚本（推荐⭐）

**Linux/Mac**：
```bash
chmod +x setup_conda_env.sh
./setup_conda_env.sh
```

**Windows**：
```cmd
setup_conda_env.bat
```

#### 方式B：手动创建（快速）

**前置条件**：已安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/products/distribution)

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate wefinance

# 验证安装
python --version  # 应显示 Python 3.10.x
# 如需使用pytest-cov等开发工具
pip install -r requirements.txt
```

**注意**：项目已完全移除 PaddleOCR，使用 GPT-4o Vision API 进行 OCR 识别，无需下载模型文件。

#### 方式C：从零开始（详细步骤）

```bash
# 1. 安装Miniconda（如果还没有）
# 下载：https://docs.conda.io/en/latest/miniconda.html

# 2. 创建环境
conda env create -f environment.yml

# 3. 激活环境
conda activate wefinance

# 4. 验证核心包
python -c "import streamlit, openai, langchain; print('✅ 所有核心包安装成功')"
```

**国内用户加速（可选）**：
```bash
# 配置清华镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

### 2. 环境配置

创建`.env`文件（复制模板）：
```bash
cp .env.example .env
```

编辑`.env`文件，填入你的API密钥：
```bash
# ✅ PRIMARY: newapi.deepwisdom.ai (OpenAI-compatible relay API)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here  # 替换为你的真实密钥
OPENAI_BASE_URL=https://newapi.deepwisdom.ai/v1
OPENAI_MODEL=gpt-4o
```

### 3. 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中打开：`http://localhost:8501`

### 5. 界面语言切换

- 默认语言：中文（简体）
- 切换方式：在左侧侧边栏的「界面语言」下拉框选择 `中文 / English`
- 实时生效：导航、页面标题、提示信息、对话回复与推荐结果都会即时更新，无需刷新
- 缓存策略：热点数据（分析结果、推荐方案、聊天缓存）自动按语言分开缓存，避免串联

## 项目结构

```
WeFinance/
├── app.py                      # Streamlit主入口
├── environment.yml             # Conda环境配置
├── requirements.txt            # pip依赖（备用）
├── .env.example               # 环境变量模板
├── .env                       # 环境变量（私密，git ignored）
├── pages/                     # Streamlit页面
│   ├── bill_upload.py         # 账单上传页面
│   ├── spending_insights.py   # 消费洞察页面
│   ├── advisor_chat.py        # 财务顾问聊天页面
│   └── investment_recs.py     # 投资推荐页面
├── modules/                   # 核心业务模块
│   ├── analysis.py           # 数据分析模块
│   └── chat_manager.py       # 对话管理器
├── services/                  # AI服务层
│   ├── vision_ocr_service.py  # Vision LLM OCR服务（GPT-4o Vision）
│   ├── ocr_service.py        # OCR服务门面
│   ├── structuring_service.py # 结构化服务（已弃用）
│   ├── recommendation_service.py # 推荐服务
│   └── langchain_agent.py    # LangChain Agent封装（可选）
├── models/                    # 数据模型
│   └── entities.py           # 实体定义（Transaction、UserProfile等）
├── utils/                     # 工具函数
│   └── session.py            # 会话管理
├── tests/                     # 单元测试
│   ├── test_integration.py   # 端到端流程测试
│   ├── test_ocr_service.py   # OCR服务单测
│   └── test_structuring_service.py # 结构化单测
└── .claude/                   # 项目文档
    └── specs/
        ├── 01-product-requirements.md    # PRD v2.0
        ├── 02-system-architecture.md     # 系统架构设计
        └── 03-sprint-plan.md             # Sprint规划

```

## 核心功能

### F1：智能账单分析器
- 上传账单图片（PNG/JPG/JPEG，最多10张）
- **GPT-4o Vision** 直接识别中文账单（准确率100%）
- 自动结构化为JSON交易记录
- 自动分类：餐饮、交通、购物、医疗、娱乐等
- 生成月度/周度消费报告
- OCR失败时支持手动粘贴JSON/CSV继续分析

### F2：对话式财务顾问
- 自然语言问答："我这个月还能花多少？"
- 结合账单数据提供个性化回答
- 支持预算查询、消费分析、术语解释、理财建议

### F3：可解释的理财建议（XAI）
- 3道问题评估风险偏好
- 基于目标生成资产配置建议
- **"为什么？"按钮**展示决策逻辑（竞赛亮点）
- 透明展示推荐背后的因果链

### F4：主动式异常检测（加分项）
- 自动检测异常支出（金额、时间、频率）
- 主动推送红色警告卡片
- 用户反馈闭环优化模型（确认/疑似欺诈）
- 信任商户白名单管理，降低误报
- 自适应阈值（1.5/2.5σ）与小样本降级处理

## 技术栈

| 类别 | 技术选型 | 版本 |
|------|---------|------|
| 前端框架 | Streamlit | 1.37+ |
| Vision OCR | GPT-4o Vision API | - |
| LLM服务 | GPT-4o API | - |
| 对话管理 | LangChain | 0.2+ |
| 数据处理 | Pandas | 2.0+ |
| 可视化 | Plotly | 5.18+ |
| 环境管理 | Conda | - |

## 开发指南

### 运行测试

```bash
# 激活conda环境
conda activate wefinance

# 运行所有测试
pytest tests/

# 运行测试并查看覆盖率（需要先安装 pytest-cov）
pip install pytest-cov
pytest --cov=modules --cov=services --cov-report=term-missing
```

- `tests/test_integration.py` 覆盖上传→分析→对话→推荐等五个核心用户场景。

### 代码规范

- 遵循PEP8规范
- 关键逻辑添加中文注释
- 函数添加docstring
- 使用`black`格式化代码：`black .`
- 使用`ruff`检查代码：`ruff check .`

### 环境管理

**查看已安装的包**：
```bash
conda list                    # 查看所有包
conda list | grep streamlit   # 查看特定包
```

**更新环境**（修改environment.yml后）：
```bash
# 激活环境
conda activate wefinance

# 更新环境（删除多余包，添加新包）
conda env update -f environment.yml --prune
```

**添加新依赖**：
```bash
# 优先使用conda安装
conda install -c conda-forge package-name

# 如果conda没有，使用pip
pip install package-name

# 导出更新后的环境
conda env export > environment.yml
# 或只导出手动安装的包（推荐）
conda env export --from-history > environment.yml
```

**删除环境**：
```bash
# 退出环境
conda deactivate

# 删除环境
conda env remove -n wefinance

# 清理缓存
conda clean --all
```

**常见问题排查**：
```bash
# 1. 环境创建失败
conda clean --all              # 清理缓存
conda env create -f environment.yml --force  # 强制重建

# 2. 包冲突
conda install package-name --force-reinstall

# 3. 查看环境详情
conda info --envs              # 列出所有环境
conda info                     # 查看conda信息
```

## 竞赛信息

- **赛事**：2025深圳国际金融科技大赛
- **赛道**：人工智能赛道
- **作品名称**：WeFinance Copilot - AI驱动的智能财务助理
- **截止日期**：2025年11月16日 24:00
- **评分标准**：
  - 产品实现完整性：40%
  - 创新性：30%
  - 商业价值：30%
- **核心竞争力**：
  - Vision LLM识别准确率100%（vs 传统OCR 0%）
  - 一步到位提取（图片→结构化数据）
  - 可解释AI（XAI）建立用户信任
  - 主动异常检测，从被动到主动

## 文档资源

- [产品需求文档 (PRD v2.0)](./.claude/specs/wefinance-copilot/01-product-requirements.md)
- [系统架构设计](./.claude/specs/wefinance-copilot/02-system-architecture.md)
- [Sprint规划](./.claude/specs/wefinance-copilot/03-sprint-plan.md)
- [部署指南](./DEPLOY.md)

## 常见问题

### 1. API调用失败？
检查`.env`配置：
- `OPENAI_API_KEY`是否正确
- `OPENAI_BASE_URL`是否可访问
- 网络是否通畅

### 2. Vision OCR识别失败？
- 确保图片清晰，文字可见
- 支持格式：PNG、JPG、JPEG
- 单张图片建议 <5MB
- 如持续失败，检查API额度和网络连接

## 许可证

本项目仅用于2025深圳国际金融科技大赛参赛，未经授权不得用于商业用途。

## 联系方式

- 项目负责人：WeFinance 团队
- 邮箱：team@wefinance.ai
- GitHub：https://github.com/JasonRobertDestiny/WeFinance-Copilot
