# WeFinance Copilot

English | **[中文](./README_zh-CN.md)**

AI-Powered Smart Financial Assistant - 2025 Shenzhen International Fintech Competition

## 🚀 Live Demo

**Try it now**: https://wefinance-copilot.streamlit.app

> No installation required - upload bill images to test OCR recognition, AI advisor chat, investment recommendations, and more

## Overview

WeFinance Copilot uses **Vision LLM + Generative AI** to transform paper/electronic bills into intelligent financial analysis, offering:
- 📸 **Smart Bill Recognition**: Upload photos, GPT-4o Vision directly extracts transactions (100% accuracy)
- 💬 **Conversational Financial Advisor**: Natural language Q&A with personalized advice
- 🔍 **Explainable AI Recommendations**: Transparent decision logic to build user trust
- ⚠️ **Proactive Anomaly Detection**: Automatically detect and alert on unusual spending

**Technical Highlights**:
- **Vision LLM Architecture**: GPT-4o Vision one-step recognition, 100% accuracy (vs traditional OCR 0%)
- **Privacy Protection**: Images transmitted via API only, no persistent storage
- Lightweight design (10-day development cycle, no database dependency)
- One-click Chinese/English UI switching (complete i18n solution + caching)

## Quick Start

> 💡 **First time?** Use the automated setup script - see [Conda Environment Guide](./docs/CONDA_GUIDE.md)

### 1. Environment Setup (3 Options)

#### Option A: Automated Script (Recommended ⭐)

**Linux/Mac**:
```bash
chmod +x setup_conda_env.sh
./setup_conda_env.sh
```

**Windows**:
```cmd
setup_conda_env.bat
```

#### Option B: Manual Setup (Quick)

**Prerequisites**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate wefinance

# Verify installation
python --version  # Should show Python 3.10.x
# For dev tools like pytest-cov
pip install -r requirements.txt
```

**Note**: PaddleOCR has been completely removed. The project now uses GPT-4o Vision API for OCR - no model downloads needed.

#### Option C: From Scratch (Detailed)

```bash
# 1. Install Miniconda (if not already installed)
# Download: https://docs.conda.io/en/latest/miniconda.html

# 2. Create environment
conda env create -f environment.yml

# 3. Activate environment
conda activate wefinance

# 4. Verify core packages
python -c "import streamlit, openai, langchain; print('✅ All core packages installed')"
```

**China Users Acceleration (Optional)**:
```bash
# Configure Tsinghua mirror
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

### 2. Configuration

Create `.env` file (copy from template):
```bash
cp .env.example .env
```

Edit `.env` file with your API key:
```bash
# ✅ PRIMARY: newapi.deepwisdom.ai (OpenAI-compatible relay API)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here  # Replace with your actual key
OPENAI_BASE_URL=https://newapi.deepwisdom.ai/v1
OPENAI_MODEL=gpt-4o
```

### 3. Run Application

```bash
streamlit run app.py
```

The app will open in your browser at: `http://localhost:8501`

### 4. Language Switching

- Default language: Simplified Chinese
- Switch method: Select `中文 / English` in the sidebar language dropdown
- Real-time effect: Navigation, page titles, prompts, chat responses, and recommendations update instantly without refresh
- Caching strategy: Hot data (analysis results, recommendations, chat cache) automatically cached separately by language to avoid cross-contamination

## Project Structure

```
WeFinance/
├── app.py                      # Streamlit main entry
├── environment.yml             # Conda environment config
├── requirements.txt            # pip dependencies (fallback)
├── .env.example               # Environment variable template
├── .env                       # Environment variables (private, git ignored)
├── pages/                     # Streamlit pages
│   ├── bill_upload.py         # Bill upload page
│   ├── spending_insights.py   # Spending insights page
│   ├── advisor_chat.py        # Financial advisor chat page
│   └── investment_recs.py     # Investment recommendations page
├── modules/                   # Core business modules
│   ├── analysis.py           # Data analysis module
│   └── chat_manager.py       # Conversation manager
├── services/                  # AI service layer
│   ├── vision_ocr_service.py  # Vision LLM OCR service (GPT-4o Vision)
│   ├── ocr_service.py        # OCR service facade
│   ├── structuring_service.py # Structuring service (deprecated)
│   ├── recommendation_service.py # Recommendation service
│   └── langchain_agent.py    # LangChain Agent wrapper (optional)
├── models/                    # Data models
│   └── entities.py           # Entity definitions (Transaction, UserProfile, etc.)
├── utils/                     # Utility functions
│   ├── session.py            # Session management
│   ├── i18n.py               # Internationalization
│   ├── error_handling.py     # Error handling
│   └── storage.py            # Persistent storage
├── locales/                   # Translation files
│   ├── zh_CN.json            # Chinese translations
│   └── en_US.json            # English translations
├── tests/                     # Unit tests
│   ├── test_integration.py   # End-to-end workflow tests
│   ├── test_ocr_service.py   # OCR service tests
│   └── test_structuring_service.py # Structuring tests
└── .claude/                   # Project documentation
    └── specs/
        ├── 01-product-requirements.md    # PRD v2.0
        ├── 02-system-architecture.md     # System architecture design
        └── 03-sprint-plan.md             # Sprint planning
```

## Core Features

### F1: Smart Bill Analyzer
- Upload bill images (PNG/JPG/JPEG, up to 10 images)
- **GPT-4o Vision** directly recognizes Chinese bills (100% accuracy)
- Auto-structured into JSON transaction records
- Auto-categorization: Dining, Transportation, Shopping, Healthcare, Entertainment, etc.
- Generate monthly/weekly spending reports
- Manual JSON/CSV paste supported if OCR fails

### F2: Conversational Financial Advisor
- Natural language Q&A: "How much can I still spend this month?"
- Personalized answers based on bill data
- Supports budget queries, spending analysis, term explanations, financial advice

### F3: Explainable Financial Recommendations (XAI)
- 3 questions assess risk tolerance
- Generate asset allocation suggestions based on goals
- **"Why?" button** shows decision logic (competition highlight)
- Transparently display causal chain behind recommendations

### F4: Proactive Anomaly Detection (Bonus Feature)
- Auto-detect anomalous spending (amount, time, frequency)
- Proactive push of red warning cards
- User feedback loop optimizes model (confirm/suspected fraud)
- Trusted merchant whitelist management reduces false positives
- Adaptive thresholds (1.5/2.5σ) with small-sample degradation handling

## Tech Stack

| Category | Technology | Version |
|----------|------------|---------|
| Frontend Framework | Streamlit | 1.37+ |
| Vision OCR | GPT-4o Vision API | - |
| LLM Service | GPT-4o API | - |
| Conversation Management | LangChain | 0.2+ |
| Data Processing | Pandas | 2.0+ |
| Visualization | Plotly | 5.18+ |
| Environment Management | Conda | - |

## Development Guide

### Running Tests

```bash
# Activate conda environment
conda activate wefinance

# Run all tests
pytest tests/

# Run tests with coverage (install pytest-cov first)
pip install pytest-cov
pytest --cov=modules --cov=services --cov-report=term-missing
```

- `tests/test_integration.py` covers five core user scenarios: upload → analysis → conversation → recommendations, etc.

### Code Standards

- Follow PEP8 guidelines
- Add Chinese comments for critical logic
- Add docstrings to functions
- Format code with `black`: `black .`
- Check code with `ruff`: `ruff check .`

### Environment Management

**View installed packages**:
```bash
conda list                    # View all packages
conda list | grep streamlit   # View specific package
```

**Update environment** (after modifying environment.yml):
```bash
# Activate environment
conda activate wefinance

# Update environment (remove extra packages, add new ones)
conda env update -f environment.yml --prune
```

**Add new dependencies**:
```bash
# Prefer conda install
conda install -c conda-forge package-name

# If conda doesn't have it, use pip
pip install package-name

# Export updated environment
conda env export > environment.yml
# Or export only manually installed packages (recommended)
conda env export --from-history > environment.yml
```

**Remove environment**:
```bash
# Exit environment
conda deactivate

# Remove environment
conda env remove -n wefinance

# Clean cache
conda clean --all
```

**Troubleshooting**:
```bash
# 1. Environment creation failed
conda clean --all              # Clean cache
conda env create -f environment.yml --force  # Force rebuild

# 2. Package conflicts
conda install package-name --force-reinstall

# 3. View environment details
conda info --envs              # List all environments
conda info                     # View conda info
```

## Competition Info

- **Event**: 2025 Shenzhen International Fintech Competition
- **Track**: AI Track
- **Project Name**: WeFinance Copilot - AI-Powered Smart Financial Assistant
- **Deadline**: November 16, 2025, 24:00
- **Scoring Criteria**:
  - Product completeness: 40%
  - Innovation: 30%
  - Business value: 30%
- **Core Competitive Advantages**:
  - Vision LLM OCR: 100% accuracy (vs Traditional OCR 0%)
  - One-step extraction: Image → Structured data directly
  - Explainable AI (XAI) builds user trust
  - Proactive anomaly detection: From reactive to proactive

## Documentation

- [Product Requirements Document (PRD v2.0)](./.claude/specs/wefinance-copilot/01-product-requirements.md)
- [System Architecture Design](./.claude/specs/wefinance-copilot/02-system-architecture.md)
- [Sprint Planning](./.claude/specs/wefinance-copilot/03-sprint-plan.md)
- [Deployment Guide](./DEPLOY.md)

## FAQ

### 1. API call failure?
Check `.env` configuration:
- Is `OPENAI_API_KEY` correct?
- Is `OPENAI_BASE_URL` accessible?
- Is network connection stable?

### 2. Vision OCR recognition failure?
- Ensure image is clear with visible text
- Supported formats: PNG, JPG, JPEG
- Single image recommended <5MB
- If persistent failures, check API quota and network connection

## License

This project is for 2025 Shenzhen International Fintech Competition participation only. Commercial use without authorization is prohibited.

## Contact

- Project Lead: WeFinance Team
- Email: team@wefinance.ai
- GitHub: https://github.com/JasonRobertDestiny/WeFinance-Copilot
