import streamlit as st
import os
import json
import re
import requests
from pypdf import PdfReader

# Page Configuration
st.set_page_config(
    page_title="Neural Nexus — Resume Skill-Gap Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Secure Key Management (Hidden) ----------------- #

# Load API key automatically from Streamlit Secrets or Environment Variables
GEMINI_KEY = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not GEMINI_KEY:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")

# ----------------- Helper Functions ----------------- #

def extract_text_from_pdf(uploaded_file):
    """Extracts raw text from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_idx + 1} ---\n" + page_text
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def get_available_gemini_models(api_key):
    """Dynamically queries Google to find all supported models."""
    clean_key = api_key.strip().strip('"').strip("'").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            models = [
                m["name"].replace("models/", "")
                for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            return models
        elif resp.status_code == 400:
            return []
    except Exception:
        pass
    return []

def analyze_resume_skill_gap(resume_text, jd_text, api_key):
    """Calls Google Gemini API using dynamically discovered active models."""
    
    clean_key = api_key.strip().strip('"').strip("'").strip()
    
    prompt = f"""
You are an expert AI Career Coach & Skill Alignment Specialist.
Your mission is to compare a candidate's resume with a target Job Description (JD), identify relevant skill gaps, and provide clear, evidence-based recommendations to help the candidate improve.

CRITICAL ETHICAL GUARDRAILS:
1. DO NOT make hiring or rejection decisions (Do NOT say "You are hired", "You are rejected", or provide pass/fail decisions).
2. DO NOT make statements on behalf of employers.
3. Keep tone supportive, educational, constructive, and strictly evidence-based.
4. Base all claims directly on quotes or facts found in the provided Resume and Job Description.

--- RESUME CONTENT ---
{resume_text}

--- TARGET JOB DESCRIPTION ---
{jd_text}

--- INSTRUCTIONS ---
Respond ONLY with a valid, raw JSON object (without markdown code fences, backticks, or extra commentary).
The JSON must follow this exact schema:
{{
  "overall_summary": "Constructive 2-3 sentence overview of how candidate skills align with the target role.",
  "alignment_percentage": 75,
  "matched_skills": [
    {{
      "skill_name": "Name of matched skill",
      "category": "Technical | Soft Skill | Domain Knowledge | Tool",
      "evidence_from_resume": "Direct reference or quote from the candidate's resume showing this skill",
      "relevance_to_jd": "Brief reason why this aligns with the job requirements"
    }}
  ],
  "missing_skills": [
    {{
      "skill_name": "Name of missing or gap skill",
      "category": "Technical | Soft Skill | Domain Knowledge | Tool",
      "jd_requirement": "Direct quote or requirement mentioned in the job description",
      "gap_severity": "High | Medium | Low",
      "explanation": "Why this gap matters and what is missing based on the resume"
    }}
  ],
  "actionable_recommendations": [
    {{
      "target_skill": "Skill to learn or improve",
      "action_plan": "Specific practical step (e.g. build a project, take a course, earn a certification)",
      "suggested_topics": ["Topic 1", "Topic 2"],
      "estimated_timeline": "e.g. 1-2 weeks"
    }}
  ],
  "disclaimer": "This analysis is an informational skill-gap tool designed solely for career development and skill improvement. It does not constitute or reflect official hiring decisions on behalf of any employer."
}}
"""

    available_models = get_available_gemini_models(clean_key)
    
    candidates = []
    for m in available_models:
        if "flash" in m.lower() and m not in candidates:
            candidates.append(m)
            
    for m in available_models:
        if "gemini" in m.lower() and m not in candidates:
            candidates.append(m)
            
    for m in available_models:
        if m not in candidates:
            candidates.append(m)
            
    if not candidates:
        candidates = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash-latest"]

    last_error = None
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2
        }
    }

    for model_name in candidates:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={clean_key}"
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            
            error_data = resp.json().get("error", {})
            error_msg = error_data.get("message", resp.text)
            last_error = f"Model '{model_name}' Error ({resp.status_code}): {error_msg}"
        except Exception as e:
            last_error = f"Model '{model_name}' Exception: {e}"
            continue

    raise RuntimeError(f"Unable to generate analysis. {last_error}")

def clean_json_response(raw_text):
    """Strips markdown code fences if Gemini outputs them."""
    cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.rstrip("`").strip()
    return json.loads(cleaned)

# ----------------- Clean User Interface (No API Keys Displayed) ----------------- #

# Clean Sidebar (Zero API/Key/Settings Clutter)
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/resume.png", width=110)
    st.title("Neural Nexus")
    st.markdown("**AI Skill-Gap Assistant**")
    st.caption("GENAI-09 Solution")
    
    st.markdown("---")
    st.markdown("### 📋 Quick Guide")
    st.markdown("""
    1. **Upload Resume**: Select your candidate resume (PDF format).
    2. **Target Job**: Paste the job description or requirements.
    3. **Run AI Analysis**: Receive instant skill alignment, identified gaps, and study advice.
    """)
    
    st.markdown("---")
    st.markdown("### ⚖️ Ethical AI Notice")
    st.caption("Neural Nexus is designed solely as an educational career coach. It provides constructive recommendations and does **not** issue hiring or rejection verdicts.")

# Main Header
st.title("🧠 Neural Nexus — Resume Skill-Gap Assistant")
st.caption("AI-Powered Skill Alignment & Learning Recommendations | Problem: GENAI-09")

st.warning("⚠️ **Notice**: This tool is an educational skill-gap assistant. It provides recommendations for personal development and **does not make hiring decisions** on behalf of any employer.", icon="⚖️")

# Sample Data Presets
sample_jd = """Senior Backend Developer (Python)
Requirements:
- 3+ years of experience with Python and FastAPI or Django
- Strong experience with PostgreSQL database modeling and query optimization
- Hands-on experience with Docker, Kubernetes, and AWS cloud deployment (ECS/EKS, S3)
- Familiarity with CI/CD pipelines (GitHub Actions)
- Excellent understanding of microservices architecture and Redis caching
- Strong communication and cross-functional collaboration skills"""

# Input Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Candidate Resume")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    resume_text = ""
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text:
            st.success(f"✅ Successfully read resume ({len(resume_text.split())} words)")
            with st.expander("📄 Preview Extracted Resume Text"):
                st.text_area("Extracted Content", resume_text, height=200, disabled=True)

with col2:
    st.subheader("2. Target Job Description")
    
    load_sample = st.checkbox("Fill with sample Job Description for testing", value=False)
    default_jd_value = sample_jd if load_sample else ""
    
    jd_input = st.text_area(
        "Paste the Job Description here:",
        value=default_jd_value,
        height=260,
        placeholder="Paste requirements, qualifications, and responsibilities..."
    )

st.markdown("---")

# Analyze Button
col_btn, _ = st.columns([1, 3])
with col_btn:
    analyze_clicked = st.button("🚀 Analyze Skill Gap", type="primary", use_container_width=True)

if analyze_clicked:
    # Check if API key is present in secrets
    active_key = GEMINI_KEY.strip().strip('"').strip("'")
    
    if not active_key:
        st.error("⚠️ System Configuration Required: Please configure `GEMINI_API_KEY` in Streamlit Cloud Secrets (Settings -> Secrets).")
    elif not resume_text:
        st.error("❌ Please upload a valid PDF resume.")
    elif not jd_input.strip():
        st.error("❌ Please paste a Job Description to compare against.")
    else:
        with st.spinner("🔍 Analyzing skills, identifying gaps, and generating recommendations..."):
            try:
                raw_response = analyze_resume_skill_gap(resume_text, jd_input, active_key)
                data = clean_json_response(raw_response)
                
                # Success Display
                st.balloons()
                st.success("🎉 Analysis Complete!")
                
                # Summary Card
                st.markdown("### 📊 Skill Alignment Overview")
                metric_col1, metric_col2 = st.columns([1, 3])
                with metric_col1:
                    match_pct = data.get("alignment_percentage", 0)
                    st.metric(label="Skill Match Estimate", value=f"{match_pct}%")
                with metric_col2:
                    st.info(data.get("overall_summary", "Analysis completed."))
                
                st.markdown("---")
                
                # Skills Breakdown Columns
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.subheader("🟢 Matched Skills (In Resume)")
                    matched = data.get("matched_skills", [])
                    if matched:
                        for item in matched:
                            with st.container(border=True):
                                st.markdown(f"**✅ {item.get('skill_name')}** `[{item.get('category')}]`")
                                st.markdown(f"* **Resume Evidence**: *\"{item.get('evidence_from_resume')}\"*")
                                st.markdown(f"* **Why it fits**: {item.get('relevance_to_jd')}")
                    else:
                        st.write("No strong direct skill matches identified.")
                
                with res_col2:
                    st.subheader("🔴 Skill Gaps (To Develop)")
                    missing = data.get("missing_skills", [])
                    if missing:
                        for item in missing:
                            severity = item.get("gap_severity", "Medium")
                            badge_color = "🔴" if severity == "High" else "🟡"
                            with st.container(border=True):
                                st.markdown(f"**{badge_color} {item.get('skill_name')}** `[Priority: {severity}]`")
                                st.markdown(f"* **Required by JD**: *\"{item.get('jd_requirement')}\"*")
                                st.markdown(f"* **Explanation**: {item.get('explanation')}")
                    else:
                        st.write("No significant skill gaps detected!")
                
                st.markdown("---")
                
                # Actionable Roadmap
                st.subheader("💡 Actionable Learning Recommendations")
                recommendations = data.get("actionable_recommendations", [])
                if recommendations:
                    for idx, rec in enumerate(recommendations, 1):
                        with st.expander(f"📌 Step {idx}: Bridge Gap in **{rec.get('target_skill')}** (Est: {rec.get('estimated_timeline')})", expanded=True):
                            st.markdown(f"**Action Plan**: {rec.get('action_plan')}")
                            topics = rec.get("suggested_topics", [])
                            if topics:
                                st.markdown("**Recommended Topics to Study:**")
                                for t in topics:
                                    st.markdown(f"- {t}")
                
                # Disclaimer Footer
                st.markdown("---")
                st.caption(f"🛡️ **Disclaimer**: {data.get('disclaimer')}")
                
                # Download Report Option
                report_text = f"""# Neural Nexus — Resume Skill-Gap Report
Overall Summary: {data.get('overall_summary')}
Skill Match Estimate: {data.get('alignment_percentage')}%

## Matched Skills
{json.dumps(data.get('matched_skills', []), indent=2)}

## Missing Skills / Gaps
{json.dumps(data.get('missing_skills', []), indent=2)}

## Recommendations
{json.dumps(data.get('actionable_recommendations', []), indent=2)}

Disclaimer:
{data.get('disclaimer')}
"""
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=report_text,
                    file_name="neural_nexus_report.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"{e}")
