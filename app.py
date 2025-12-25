import streamlit as st
from pathlib import Path
import json
from journals_db import get_journal, list_journals
from manuscript_tools import analyze_manuscript, check_compliance, generate_template, export_checklist

st.set_page_config(page_title="Manuscript Assistant", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #2ca02c; }
    .success { color: #27ae60; font-weight: bold; }
    .error { color: #e74c3c; font-weight: bold; }
    .warning { color: #f39c12; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📝 Manuscript Assistant</p>', unsafe_allow_html=True)
st.write("AI-powered tool for manuscript preparation aligned with global journal requirements")

# Sidebar navigation
page = st.sidebar.radio("Navigation", 
    ["Home", "Manuscript Analyzer", "Journal Requirements", "Template Generator", "Submission Checklist", "Guidelines"])

# ============= HOME =============
if page == "Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Welcome to Manuscript Assistant</p>', unsafe_allow_html=True)
        st.write("""
        Your AI-powered companion for manuscript preparation across global journals:
        
        **Features:**
        - 📊 Analyze your manuscript against journal requirements
        - 🏥 Support for IJQHC, JAMA, Lancet, BMJ, NEJM, Nature, Science
        - 📋 Auto-generate submission checklists
        - 📄 Create journal-specific templates
        - ✅ Compliance checking (STROBE, CONSORT, etc.)
        - 💡 Smart recommendations
        """)
    
    with col2:
        st.metric("Supported Journals", len(list_journals()))
        st.metric("Available Templates", 7)

# ============= MANUSCRIPT ANALYZER =============
elif page == "Manuscript Analyzer":
    st.markdown('<p class="sub-header">Manuscript Analyzer</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your manuscript (DOCX)", type=["docx"])
        
    with col2:
        selected_journal = st.selectbox("Select Target Journal", list_journals())
    
    if uploaded_file and selected_journal:
        # Analyze manuscript
        manuscript_data = analyze_manuscript(uploaded_file)
        
        if "error" not in manuscript_data:
            st.success("✅ Manuscript analyzed successfully")
            
            # Get journal requirements
            journal = get_journal(selected_journal)
            
            # Display manuscript info
            st.subheader("📋 Manuscript Information")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Word Count", manuscript_data["word_count"])
            col2.metric("Tables", manuscript_data["tables"])
            col3.metric("Est. References", manuscript_data["estimated_references"])
            col4.metric("Paragraphs", manuscript_data["paragraphs"])
            
            # Compliance check
            st.subheader(f"🔍 Compliance Check - {selected_journal}")
            compliance = check_compliance(manuscript_data, journal)
            
            for issue in compliance["issues"]:
                if "✅" in issue:
                    st.success(issue)
                elif "❌" in issue:
                    st.error(issue)
            
            for warning in compliance["warnings"]:
                st.warning(warning)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if manuscript_data["word_count"] > journal["word_limit"]:
                reduction_needed = manuscript_data["word_count"] - journal["word_limit"]
                st.info(f"Reduce manuscript by approximately {reduction_needed} words ({int(reduction_needed/manuscript_data['word_count']*100)}%)")
        else:
            st.error(f"Error analyzing manuscript: {manuscript_data['error']}")

# ============= JOURNAL REQUIREMENTS =============
elif page == "Journal Requirements":
    st.markdown('<p class="sub-header">Global Journal Requirements</p>', unsafe_allow_html=True)
    
    selected_journal = st.selectbox("Select Journal", list_journals())
    
    if selected_journal:
        journal = get_journal(selected_journal)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"### {journal['name']}")
            
            # Requirements table
            requirements = {
                "Word Limit": f"{journal['word_limit']} words",
                "Abstract Limit": f"{journal['abstract_limit']} words",
                "References Limit": f"{journal['references_limit']}",
                "Tables Limit": f"{journal['tables_limit']}",
                "Title Limit": f"{journal['title_limit']} characters",
                "Keywords": f"{journal['keywords']} terms",
                "Format": journal['format'],
                "Style Guide": journal['style_guide'],
                "Reporting Guideline": journal['reporting_guideline']
            }
            
            for key, value in requirements.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.write("### Quick Links")
            st.markdown(f"[Journal Website]({journal['url']})")
            st.write(f"**Reporting:** {journal['reporting_guideline']}")

# ============= TEMPLATE GENERATOR =============
elif page == "Template Generator":
    st.markdown('<p class="sub-header">Template Generator</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        journal_name = st.selectbox("Select Journal for Template", list_journals())
    
    with col2:
        author_name = st.text_input("Author Name", "Dr. Your Name")
    
    with col3:
        institution = st.text_input("Institution", "Your University")
    
    if st.button("Generate Template"):
        doc = generate_template(journal_name, author_name, institution)
        
        # Save to temp file
        template_path = f"template_{journal_name}.docx"
        doc.save(template_path)
        
        with open(template_path, "rb") as f:
            st.download_button(
                label=f"Download {journal_name} Template",
                data=f.read(),
                file_name=template_path,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        
        st.success(f"✅ Template generated for {journal_name}")

# ============= SUBMISSION CHECKLIST =============
elif page == "Submission Checklist":
    st.markdown('<p class="sub-header">Submission Checklist</p>', unsafe_allow_html=True)
    
    selected_journal = st.selectbox("Select Journal", list_journals())
    
    if selected_journal:
        journal = get_journal(selected_journal)
        checklist = export_checklist(journal)
        
        st.text_area("Submission Checklist", checklist, height=400)
        
        # Download checklist
        st.download_button(
            label="Download Checklist",
            data=checklist,
            file_name=f"checklist_{selected_journal}.txt",
            mime="text/plain"
        )

# ============= GUIDELINES =============
elif page == "Guidelines":
    st.markdown('<p class="sub-header">Publication Guidelines</p>', unsafe_allow_html=True)
    
    st.subheader("Writing Best Practices")
    st.write("""
    **IMRAD Structure:**
    - **Introduction**: Set context and research question
    - **Methods**: Describe study design and procedures
    - **Results**: Present findings objectively
    - **Discussion**: Interpret findings and implications
    
    **Abstract Tips:**
    - Structured format: Background, Methods, Results, Conclusions
    - 300-400 words typical
    - No citations or abbreviations without definition
    
    **References:**
    - Use consistent style (APA, Vancouver, Chicago)
    - Include all cited sources
    - Keep current and relevant
    
    **Reporting Guidelines:**
    - STROBE for observational studies
    - CONSORT for randomized trials
    - PRISMA for systematic reviews
    - Check journal requirements
    """)
    
    st.subheader("Common Mistakes to Avoid")
    st.write("""
    ❌ Exceeding word/reference limits
    ❌ Incomplete author information
    ❌ Missing conflict of interest statement
    ❌ Inconsistent formatting
    ❌ Poor quality tables/figures
    ❌ Not following journal template
    ❌ Submitting before peer review
    ❌ Plagiarism or self-plagiarism
    """)
    
    st.subheader("Submission Timeline")
    st.write("""
    1. **Preparation** (2-4 weeks): Write and format manuscript
    2. **Internal Review** (1-2 weeks): Internal feedback
    3. **Submission** (1 day): Upload to journal portal
    4. **Editorial Review** (2-4 weeks): Initial screening
    5. **Peer Review** (8-12 weeks): Reviewer feedback
    6. **Revision** (4-6 weeks): Address comments
    7. **Publication** (2-4 weeks): Final copyediting
    
    **Total: 4-7 months typical**
    """)

# Footer
st.markdown("---")
st.write("**Manuscript Assistant v1.0** | Powered by AI | MIT License")
st.write("Based on IJQHC, JAMA, Lancet, BMJ, NEJM, Nature, Science requirements")
