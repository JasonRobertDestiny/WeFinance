# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WeFinance Copilot is an AI-powered financial assistant for the 2025 Shenzhen Fintech Competition. It uses Vision LLM (GPT-4o) for bill OCR, conversational AI for financial advice, and explainable AI for investment recommendations.

**Key Architecture Decision**: Originally used PaddleOCR, but migrated to GPT-4o Vision OCR for 100% recognition accuracy (vs 0% with synthetic images). This is the core competitive advantage.

## Essential Commands

### Environment Setup
```bash
# Activate conda environment (REQUIRED before any command)
conda activate wefinance

# Install dependencies
conda env create -f environment.yml  # First time
pip install -r requirements.txt      # Development tools
```

### Running the Application
```bash
# Start Streamlit app
streamlit run app.py --server.port 8501

# For background/headless mode
streamlit run app.py --server.port 8501 --server.headless true
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ocr_service.py -v

# Run with coverage
pytest --cov=modules --cov=services --cov=utils --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=modules --cov=services --cov=utils --cov-report=html
# Open htmlcov/index.html to view
```

### Code Quality
```bash
# Format code (REQUIRED before commits)
black .

# Lint code
ruff check .
ruff check --fix .  # Auto-fix safe issues
```

### Vision OCR Testing
```bash
# Test Vision OCR with sample bills
python test_vision_ocr.py

# This uses assets/sample_bills/*.png and validates 100% recognition rate
```

## Architecture Overview

### Core Innovation: Vision LLM Pipeline

**CRITICAL**: The OCR system underwent a major architectural shift:

```
Old (PaddleOCR): Image → PaddleOCR → Text → GPT-4o → Structured Data
New (Vision LLM): Image → GPT-4o Vision → Structured Data (ONE STEP)
```

**Why this matters**:
- PaddleOCR: 0% recognition on synthetic images, requires heavy dependencies
- Vision LLM: 100% recognition, zero additional dependencies
- **services/vision_ocr_service.py** is the implementation (145 lines)
- **services/ocr_service.py** maintains backward compatibility

### Data Flow Architecture

```
User Upload (pages/bill_upload.py)
    ↓
OCRService.process_files()
    ↓
VisionOCRService.extract_transactions_from_image()
    ├─ Base64 encode image
    ├─ GPT-4o Vision API call with structured prompt
    ├─ JSON parsing → Transaction objects
    └─ Return List[Transaction]
    ↓
Session State (utils/session.py)
    ├─ st.session_state["transactions"]
    └─ Shared across all pages
    ↓
Multiple Consumers:
    ├─ Advisor Chat (modules/chat_manager.py)
    ├─ Investment Recs (services/recommendation_service.py)
    └─ Spending Insights (modules/analysis.py)
```

### Session State Management

**CRITICAL PATTERN**: Streamlit session_state is the single source of truth. All data flows through `utils/session.py`:

```python
# CORRECT: Centralized state management
from utils.session import set_transactions, get_transactions

set_transactions(transactions)  # In bill_upload.py
transactions = get_transactions()  # In other pages

# WRONG: Direct session_state manipulation
st.session_state["transactions"] = transactions  # Bypasses validation
```

**Key session state keys**:
- `transactions`: List[Transaction] - Core financial data
- `chat_history`: List[dict] - Conversation history
- `locale`: str - "zh_CN" or "en_US"
- `monthly_budget`: float - User's budget
- `anomaly_state`: dict - Active anomaly alerts

### LLM Integration Patterns

**Three LLM use cases**:

1. **Vision OCR** (`services/vision_ocr_service.py`):
   - Model: GPT-4o Vision
   - Input: Image bytes
   - Output: Structured JSON → Transaction objects
   - Temperature: 0.0 (deterministic)

2. **Chat** (`modules/chat_manager.py`):
   - Model: GPT-4o (text)
   - Input: User query + transaction context + budget
   - Output: Natural language financial advice
   - Caching: Query-level cache with 20-item LRU

3. **Recommendations** (`services/recommendation_service.py`):
   - Model: GPT-4o (text)
   - Input: Transactions + risk profile + investment goal
   - Output: Structured recommendation with reasoning chain
   - Caching: Streamlit @st.cache_data on deterministic inputs

### Internationalization (i18n)

**Architecture**: Lazy-loaded translation system with fallback mechanism.

```python
# CORRECT: Get translations
from utils.session import get_i18n

i18n = get_i18n()
title = i18n.t("chat.title")  # Returns Chinese or English based on locale

# Translation files
utils/i18n.py  # Core i18n engine
locales/zh_CN.json  # Chinese translations
locales/en_US.json  # English translations
```

**Fallback chain**: `i18n.t(key)` → locale file → fallback locale → key itself

**Locale switching**: `utils/session.switch_locale("en_US")` triggers full UI rerun

## Critical Implementation Details

### Vision OCR Prompt Engineering

**Location**: `services/vision_ocr_service.py:62-81`

The prompt is carefully engineered for structured output:
- Specifies exact JSON format
- Lists valid categories: 餐饮、交通、购物、娱乐、医疗、教育、其他
- Requires YYYY-MM-DD date format
- Explicitly requests no markdown code blocks
- Returns empty array [] if no transactions found

**DO NOT** modify this prompt without testing against all sample bills.

### Error Handling Philosophy

**Pattern**: Graceful degradation, not crash-and-burn.

```python
# Vision OCR failure → Return empty list, allow manual input
except Exception as e:
    logger.error(f"OCR failed: {e}")
    return []  # NOT: raise

# LLM timeout → Use fallback response
except TimeoutError:
    return cached_or_default_response()  # NOT: raise
```

**Why**: User experience > perfect execution. Always provide a fallback path.

### Test Coverage Requirements

**Current status**: 58% coverage, target 70%+

**Weak spots**:
- `modules/chat_manager.py`: 48% (needs fallback, cache, whitelist tests)
- `utils/session.py`: 36% (needs session init, locale switch, persistence tests)
- `services/vision_ocr_service.py`: NEW FILE, no tests yet

**Testing Vision OCR**:
```python
# Use mocking to avoid real API calls
from unittest.mock import patch, MagicMock

@patch('services.vision_ocr_service.OpenAI')
def test_extract_transactions(mock_openai):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '[{"date": "2025-11-01", ...}]'
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    service = VisionOCRService()
    transactions = service.extract_transactions_from_image(b'fake_image')

    assert len(transactions) == 1
    assert transactions[0].date == "2025-11-01"
```

## Project Roles (see .claude/PROJECT_RULES.md)

**Claude Code**: Architect, planner, code reviewer. Provides design decisions and generates detailed prompts for codex.

**Codex**: Implementation engineer. Executes on Claude Code's plans, writes code, tests, and documentation.

**Workflow**: User → Claude Code (analyze, design, prompt) → Codex (implement, test) → Claude Code (review) → User

## Configuration

### Required .env Variables
```bash
# OpenAI-compatible API (REQUIRED)
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://newapi.deepwisdom.ai/v1
OPENAI_MODEL=gpt-4o

# Optional: LLM provider selection
LLM_PROVIDER=openai  # Currently only supports OpenAI-compatible APIs
```

### Conda Environment
**Python Version**: 3.10.x (specified in environment.yml)

**Why Conda**: Isolates PaddlePaddle dependencies (even though we don't use it anymore, some legacy code remains).

**IMPORTANT**: Always activate environment before running any command:
```bash
conda activate wefinance
```

## Common Pitfalls

### 1. Forgetting to Activate Conda Environment
```bash
# WRONG
python test_vision_ocr.py  # Uses system Python, missing packages

# CORRECT
conda activate wefinance
python test_vision_ocr.py
```

### 2. Breaking Session State Consistency
```bash
# WRONG: Direct manipulation
st.session_state["transactions"] = new_data

# CORRECT: Use helper functions
from utils.session import set_transactions
set_transactions(new_data)
```

### 3. Ignoring Vision OCR JSON Parsing
The Vision LLM sometimes wraps JSON in markdown code blocks:
```
```json
[{"date": "2025-11-01", ...}]
```
```

The code in `vision_ocr_service.py:108-116` handles this automatically. Don't bypass it.

### 4. Testing Without Sample Data
Use `assets/sample_bills/` for testing, not random images. These bills are:
- **bill_dining.png**: 4 dining transactions
- **bill_mixed.png**: 4 mixed-category transactions
- **bill_shopping.png**: 3 shopping transactions

Generated with `generate_sample_bills.py` using Noto CJK fonts.

### 5. Modifying Transaction Model Without Migration
`models/entities.py::Transaction` is used everywhere. Adding/removing fields requires:
1. Update model with default values for backward compatibility
2. Update all consumers (chat, analysis, recommendations)
3. Update tests
4. Run full test suite

## Architecture Decisions Log

### 2025-11-06: PaddleOCR → GPT-4o Vision
**Problem**: PaddleOCR couldn't recognize synthetic bill images (0% accuracy).

**Solution**: Replaced with GPT-4o Vision, achieving 100% accuracy.

**Impact**:
- Removed PaddlePaddle dependency (still in environment.yml for compatibility)
- Simplified pipeline from 2-step to 1-step
- Increased API cost but acceptable for MVP/competition

**Files Changed**:
- NEW: `services/vision_ocr_service.py`
- MODIFIED: `services/ocr_service.py` (now delegates to VisionOCRService)
- DEPRECATED: PaddleOCR usage (kept for backward compatibility)

### 2025-11-06: Codex Collaboration Model
**Decision**: Claude Code provides architecture/design, Codex implements.

**Rationale**:
- Claude Code: Better at analysis, design, code review (Linus philosophy)
- Codex: Better at bulk implementation, repetitive tasks

**Process**: See `.claude/PROJECT_RULES.md` for detailed workflow.

## Debugging Tips

### Vision OCR Not Working
1. Check .env has valid OPENAI_API_KEY
2. Test API connectivity: `python test_vision_ocr.py`
3. Check logs: `services/vision_ocr_service.py` logs at INFO level
4. Verify image format: PNG/JPG/JPEG only

### Streamlit Session State Issues
1. Check `utils/session.py` for state initialization
2. Use `st.session_state` debugger: Add `st.write(st.session_state)` temporarily
3. Verify state keys match between pages

### LLM Timeout/Failure
1. Check API rate limits
2. Verify network connectivity to OpenAI-compatible endpoint
3. Review fallback mechanisms in `chat_manager.py` and `recommendation_service.py`

### Test Failures
1. Activate conda environment first: `conda activate wefinance`
2. Check test isolation: Each test should be independent
3. Mock external dependencies (LLM APIs, file I/O)
4. Run single test for debugging: `pytest tests/test_file.py::test_function -v`

## Performance Considerations

### LLM API Calls
- **Vision OCR**: ~2-5 seconds per image (acceptable for demo)
- **Chat**: ~1-3 seconds per message (cached queries return instantly)
- **Recommendations**: ~3-7 seconds (cached by input hash)

### Caching Strategy
1. **Query-level**: `chat_manager.py` caches exact query matches
2. **Deterministic inputs**: `@st.cache_data` on recommendation generation
3. **Session state**: All transaction data lives in session, not re-fetched

### Optimization Opportunities
- Add `timeout=30` to all LLM calls (Task 5 in backlog)
- Implement progressive loading for long transaction lists
- Pre-cache common queries on startup

## Security Notes

### API Key Management
- **NEVER** commit .env file
- Use .env.example as template
- API keys should be in .env, loaded via python-dotenv

### Input Validation
- Transaction amounts: Validated in Pydantic models
- File uploads: Limited to PNG/JPG/PDF in Streamlit uploader
- User inputs: Sanitized before LLM prompts (no injection attacks)

### Privacy
- Images processed locally (base64 encoded for API call)
- No persistent storage of images or transactions (session-only)
- LLM provider (newapi.deepwisdom.ai) may log API calls - inform users

## For More Information

- **Project Structure**: See README.md
- **Development Rules**: See .claude/PROJECT_RULES.md
- **System Architecture**: See .claude/specs/wefinance-copilot/02-system-architecture.md
- **Testing Guide**: Run `pytest --cov` and review htmlcov/index.html
