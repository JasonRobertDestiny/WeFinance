# Repository Guidelines

## 项目结构与模块组织
- `app.py` 为 Streamlit 入口，所有页面位于 `pages/`（如 `bill_upload.py`、`advisor_chat.py` 等），遵循“单页单职责”便于扩展导航。
- 业务能力集中在 `modules/` 与 `services/`：前者处理分析与对话状态，后者封装 OCR、推荐与 LangChain 代理，若新增服务请同步 `services/__init__.py`。
- 数据模型在 `models/entities.py`，工具与缓存逻辑在 `utils/`，多语言资源放在 `locales/`；i18n 文案增删需保持 `zh_CN.json` 与 `en_US.json` 键一致。
- 资源文件（图片、演示）位于 `assets/` 与 `screenshots/`，测试样例和规格说明在 `tests/`、`docs/`、`.claude/`，其中 `docs/CONDA_GUIDE.md` 记录完整环境策略。

## 构建、测试与开发命令
```bash
conda env create -f environment.yml          # 首次创建环境
conda activate wefinance                     # 进入开发环境
pip install -r requirements.txt              # 安装开发/测试依赖
streamlit run app.py --server.port 8501      # 本地预览
pytest tests/ -v                             # 运行全量测试
pytest --cov=modules --cov=services --cov=utils --cov-report=term-missing
black . && ruff check .                      # 统一格式化+静态检查
```
- 任何命令前务必激活 `wefinance` 环境，避免系统 Python 缺包；部署镜像亦需 `conda env update -f environment.yml --prune` 保持一致。
- 缺省自动化场景下，修改业务逻辑前后请手工跑一次关键页面流程（账单上传→洞察→顾问聊天），必要时附运行日志或 `streamlit` 控制台输出。

## 编码风格与命名约定
- Python 统一遵循 PEP8，4 空格缩进，模块/函数使用 `snake_case`，类名 `PascalCase`，常量全大写并集中定义，必要时补充类型注解以便 IDE 推断。
- 关键业务逻辑（账单解析、推荐、OCR 适配等）必须配中文注释，注释描述意图而非语句本身。
- 禁止魔法数字，需提取为具名常量；最大嵌套不超过三层，可通过拆分辅助函数保持单一职责。
- 新文件须集成在 `__init__.py` 或相关工厂，避免破坏既有模块导出。

## 测试规范
- 测试框架为 `pytest`，测试文件放在 `tests/`，命名规则 `test_<module>.py`，函数命名 `test_<行为>`，必要时通过 `pytest tests/test_ocr_service.py -v` 精准验证。
- 重要分支需附带最小化集成测试（参考 `tests/test_integration.py`），并提供示例账单/环境变量。
- 覆盖率目标：`modules/` 与 `services/` 关键路径 ≥80%；使用 `pytest --cov` 审查，并在 PR 描述中贴出结果。
- 无 CI 情况下，提交前请截图或录屏关键页面，佐证手工验收已通过。

## 提交与 PR 指南
- Git 历史遵循 `type(scope): summary`（例：`feat(ux): improve sidebar caching`），使用祈使句，英文小写类型：`feat`、`fix`、`docs`、`chore` 等。
- PR 必需内容：变更概述、相关 Issue/需求编号、测试证明（命令输出/截图）、风险与回滚方案。
- 若涉及配置或密钥，提供 `.env.example` 变更说明，严禁提交真实密钥。
- 合并前等待至少一名维护者复核，确保未触碰受限函数（如 `modules/analyzer.py` 中 `is_quality_keyword()`）。

## 安全与配置提示
- `.env` 通过 `cp .env.example .env` 初始化，仅填写代理 API Key、Base URL、模型名等敏感字段；不要上传真实凭据。
- GPT-4o Vision OCR 为唯一识别路径，若需替换模型，务必更新 `services/vision_ocr_service.py` 并在 README/DEPLOY 记录。
- 生产部署依赖 Streamlit Cloud 自动构建，推送前请确认 `requirements.txt` 与 `environment.yml` 同步，避免线上环境失配。
