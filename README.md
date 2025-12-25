# Manuscript Assistant

AI-powered manuscript preparation tool aligned with global journal requirements.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- 📊 **Manuscript Analyzer**: Upload and analyze against journal requirements
- 🏥 **7 Global Journals**: IJQHC, JAMA, Lancet, BMJ, NEJM, Nature, Science
- 📋 **Submission Checklists**: Auto-generated for each journal
- 📄 **Template Generator**: Create journal-specific templates
- ✅ **Compliance Checking**: STROBE, CONSORT, and more
- 💡 **Smart Recommendations**: Real-time feedback

## Supported Journals

- International Journal for Quality in Health Care (IJQHC)
- JAMA
- The Lancet
- British Medical Journal (BMJ)
- New England Journal of Medicine (NEJM)
- Nature
- Science

## Usage

### 1. Analyze Manuscript
- Upload your DOCX manuscript
- Select target journal
- Get instant compliance feedback

### 2. View Requirements
- Check specific journal requirements
- Compare across journals
- Download specification sheets

### 3. Generate Templates
- Create journal-specific templates
- Auto-formatted with proper margins
- IMRAD structure included

### 4. Submission Checklist
- Complete pre-submission checklist
- Per-journal requirements
- Download as text file

### 5. Read Guidelines
- Best practices
- Common mistakes
- Typical submission timeline

## Architecture

```
ManuscriptAssistant/
├── app.py                  # Streamlit UI
├── journals_db.py          # Journal requirements database
├── manuscript_tools.py     # Analysis and template functions
└── requirements.txt        # Dependencies
```

## Requirements

- Python 3.8+
- Streamlit 1.28+
- python-docx 0.8+
- Pandas 2.0+
- PyYAML 6.0+

## License

MIT License
