# WeFinance Copilot

AI-powered financial copilot built for the 2025 Shenzhen International Fintech Competition (AI Track).

## Highlights

- 📸 **Bill OCR + Structuring**: Hybrid PaddleOCR + GPT-4o pipeline extracts clean transaction data.
- 📝 **Manual Import Fallback**: Paste JSON/CSV transactions if OCR encounters an issue.
- 💬 **Conversational Advisor**: Budget queries, spending insights, and term explanations in natural language.
- 💡 **Explainable Recommendations**: Risk questionnaire, allocation plan, and XAI narrative packaged together.
- ⚠️ **Proactive Anomaly Detection**: Z-score engine, whitelist, and feedback loop help flag suspicious spending.
- 🌐 **Bilingual Experience**: Full Chinese/English interface with a one-click sidebar switch and cached translations.

## Quick Start

> Prefer the automated scripts? See [`docs/CONDA_GUIDE.md`](./docs/CONDA_GUIDE.md).

### 1. Environment Setup

```bash
# Create the Conda environment
conda env create -f environment.yml

# Activate
conda activate wefinance

# (Optional) install extra dev tools such as pytest-cov
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and provide an OpenAI-compatible API key:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key
OPENAI_BASE_URL=https://newapi.deepwisdom.ai/v1
OPENAI_MODEL=gpt-4o
```

### 3. Preload PaddleOCR (optional)

```bash
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='ch')"
```

### 4. Run the App

```bash
streamlit run app.py
```

Visit `http://localhost:8501`.

### 5. Switch Languages

- Default locale: Simplified Chinese (`zh_CN`)
- Sidebar → *Interface Language* → choose `中文` or `English`
- Navigation labels, page text, chat replies, and recommendations update instantly
- Cached data is isolated per-language, so switching does not leak previous content

## Project Structure

```
WeFinance/
├── app.py                    # Streamlit entry point (language switch + anomaly alerts)
├── locales/                  # zh_CN / en_US translation dictionaries
├── utils/
│   ├── i18n.py               # i18n loader with caching
│   └── session.py            # Session helpers + whitelist/anomaly state
├── pages/                    # Streamlit pages (OCR, insights, chat, recommendations)
├── modules/                  # Core logic (analysis, chat manager)
├── services/                 # OCR, structuring, recommendation, LangChain agent
├── tests/                    # Unit & integration tests (bilingual scenarios)
└── docs/                     # Supplemental documentation
```

## Testing

```bash
conda activate wefinance

# Run all tests (16 total as of Sprint 3)
pytest

# Coverage (install pytest-cov first)
pip install pytest-cov
pytest --cov=modules --cov=services --cov-report=term-missing
```

Key integration scenarios:
- `test_full_workflow_locale` – end-to-end Chinese & English flows
- `test_anomaly_detection_feedback` – confirm/fraud feedback loop
- `test_language_switching_behaviour` – ensures context survives locale flips
- `test_error_handling_messages` – friendly fallbacks on OCR/LLM failures

## How to Use the Demo

1. **Upload sample bills** (see `assets/sample_bills/` when populated) to trigger OCR → structuring.
2. **Review the dashboard** for category breakdowns, anomaly alerts, and AI insights.
3. **Chat with the advisor** for budget or spending questions (supports both languages).
4. **Complete the risk questionnaire** to generate personalised allocations with XAI explanation.
5. **Handle anomalies** via the homepage warning card or the dashboard sidebar.

## Development Notes

- Code follows PEP8/Black. Run `black .` and `ruff check .` before committing.
- Error handling uses i18n-friendly messages—no raw stack traces reach the UI.
- LangChain toolkit is optional; the app falls back gracefully if not configured.
- Tests mock external APIs, so no external calls occur during CI.

## License

The project is provided for competition/demo purposes only. Commercial use requires separate authorisation.

## Contact

- Team: WeFinance Copilot
- Email: team@wefinance.ai
- GitHub: <https://github.com/wefinance/copilot>
