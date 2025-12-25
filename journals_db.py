# Journal Configuration Database
journals = {
    "IJQHC": {
        "name": "International Journal for Quality in Health Care",
        "word_limit": 3000,
        "abstract_limit": 400,
        "references_limit": 30,
        "tables_limit": 5,
        "title_limit": 150,
        "keywords": 6,
        "format": "double-spaced, 11pt font",
        "style_guide": "IMRAD",
        "reporting_guideline": "STROBE",
        "url": "https://academic.oup.com/intqhc"
    },
    "JAMA": {
        "name": "JAMA - Journal of the American Medical Association",
        "word_limit": 3000,
        "abstract_limit": 300,
        "references_limit": 50,
        "tables_limit": 6,
        "title_limit": 120,
        "keywords": 5,
        "format": "double-spaced, 12pt font",
        "style_guide": "IMRAD",
        "reporting_guideline": "CONSORT",
        "url": "https://jamanetwork.com/journals/jama"
    },
    "Lancet": {
        "name": "The Lancet",
        "word_limit": 3500,
        "abstract_limit": 450,
        "references_limit": 40,
        "tables_limit": 6,
        "title_limit": 140,
        "keywords": 6,
        "format": "single-spaced, 12pt font",
        "style_guide": "IMRAD",
        "reporting_guideline": "STROBE",
        "url": "https://www.thelancet.com"
    },
    "BMJ": {
        "name": "British Medical Journal",
        "word_limit": 2750,
        "abstract_limit": 350,
        "references_limit": 35,
        "tables_limit": 5,
        "title_limit": 120,
        "keywords": 6,
        "format": "double-spaced, 12pt font",
        "style_guide": "IMRAD",
        "reporting_guideline": "CONSORT/STROBE",
        "url": "https://www.bmj.com"
    },
    "NEJM": {
        "name": "New England Journal of Medicine",
        "word_limit": 3500,
        "abstract_limit": 400,
        "references_limit": 50,
        "tables_limit": 6,
        "title_limit": 130,
        "keywords": 5,
        "format": "double-spaced, 12pt font",
        "style_guide": "IMRAD",
        "reporting_guideline": "CONSORT",
        "url": "https://www.nejm.org"
    },
    "Nature": {
        "name": "Nature",
        "word_limit": 4000,
        "abstract_limit": 300,
        "references_limit": 60,
        "tables_limit": 7,
        "title_limit": 150,
        "keywords": 5,
        "format": "single-spaced, 12pt font",
        "style_guide": "Varies",
        "reporting_guideline": "STROBE/CONSORT",
        "url": "https://www.nature.com"
    },
    "Science": {
        "name": "Science",
        "word_limit": 3500,
        "abstract_limit": 250,
        "references_limit": 50,
        "tables_limit": 6,
        "title_limit": 120,
        "keywords": 6,
        "format": "double-spaced, 12pt font",
        "style_guide": "IMRAD",
        "reporting_guideline": "CONSORT/STROBE",
        "url": "https://www.science.org"
    }
}

def get_journal(name):
    return journals.get(name)

def list_journals():
    return list(journals.keys())
