# ==========================================
# AI SMART RESUME ANALYZER & JOB MATCHER
# Complete Professional Version
# ==========================================



import gradio as gr
import fitz
import os
import re
import html
import tempfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER


# ==========================================
# 1. SKILLS AND JOB ROLES
# ==========================================

skills_list = [
    "Python", "Java", "SQL", "Machine Learning",
    "Deep Learning", "TensorFlow", "Scikit-learn",
    "HTML", "CSS", "JavaScript", "Power BI","R","c","c++"
]

job_roles = {
    "Data Analyst": "Python SQL Data Analysis Power BI",
    "Python Developer": "Python SQL Programming",
    "ML Engineer": "Python Machine Learning Deep Learning TensorFlow Scikit-learn",
    "Web Developer": "HTML CSS JavaScript"
}

required = {
    "Data Analyst": ["Python", "SQL", "Power BI"],
    "Python Developer": ["Python", "SQL"],
    "ML Engineer": [
        "Python", "Machine Learning", "Deep Learning",
        "TensorFlow", "Scikit-learn"
    ],
    "Web Developer": ["HTML", "CSS", "JavaScript"]
}


# ==========================================
# 2. EXTRACT RESUME TEXT
# ==========================================

def extract_resume_text(file_path):

    if not file_path:
        return ""

    try:
        document = fitz.open(file_path)

        text = ""

        for page in document:
            text += page.get_text() + "\n"

        document.close()

        return text

    except Exception as e:
        return ""


# ==========================================
# 3. EXTRACT SKILLS
# ==========================================

def extract_skills(text):

    found_skills = []

    text_lower = text.lower()

    for skill in skills_list:

        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills


# ==========================================
# 4. JOB ROLE RECOMMENDATION
# ==========================================

def recommend_job_role(found_skills):

    if not found_skills:
        return "No matching role", 0

    resume_text = " ".join(found_skills)

    roles = list(job_roles.keys())

    descriptions = [
        job_roles[role] for role in roles
    ]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume_text] + descriptions
    )

    scores = cosine_similarity(
        vectors[0:1],
        vectors[1:]
    )[0]

    best_index = scores.argmax()

    best_role = roles[best_index]

    match_percentage = round(
        scores[best_index] * 100, 2
    )

    return best_role, match_percentage


# ==========================================
# 5. ATS SCORE
# ==========================================

def calculate_ats_score(text, found_skills):

    score = 0

    if len(text.split()) >= 50:
        score += 20

    sections = [
        "education", "experience",
        "skills", "projects", "contact"
    ]

    text_lower = text.lower()

    for section in sections:

        if section in text_lower:
            score += 10

    score += min(len(found_skills) * 5, 30)

    return min(score, 100)


# ==========================================
# 6. RESUME SUGGESTIONS
# ==========================================

def get_resume_suggestions(text, found_skills):

    suggestions = []

    if len(text.split()) < 50:
        suggestions.append(
            "Add more relevant details to your resume."
        )

    if "education" not in text.lower():
        suggestions.append(
            "Add an Education section."
        )

    if "projects" not in text.lower():
        suggestions.append(
            "Include your academic or personal projects."
        )

    if "experience" not in text.lower():
        suggestions.append(
            "Add internship or practical experience if available."
        )

    if len(found_skills) < 5:
        suggestions.append(
            "Add more relevant technical skills that you genuinely know."
        )

    if not suggestions:
        suggestions.append(
            "Your resume contains several useful sections. "
            "Keep your project and skill details updated."
        )

    return suggestions


# ==========================================
# 7. PDF REPORT GENERATION
# ==========================================

def create_pdf_report(
    role, match, ats, found, missing, suggestions
):

    file_path = os.path.join(
        tempfile.gettempdir(),
        "Resume_Analysis_Report.pdf"
    )

    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        textColor=colors.HexColor("#173B70"),
        alignment=TA_CENTER,
        fontSize=20
    )

    story = []

    story.append(
        Paragraph(
            "AI SMART RESUME ANALYZER",
            title_style
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Resume Analysis Report",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 15))

    data = [
        ["Analysis", "Result"],
        ["Recommended Job Role", role],
        ["Job Match Percentage", str(match) + "%"],
        ["ATS Score", str(ats) + "%"],
        ["Total Skills Found", str(len(found))]
    ]

    table = Table(data, colWidths=[220, 220])

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#173B70")),
            ("TEXTCOLOR", (0, 0), (-1, 0),
             colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.grey),
            ("PADDING", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 1), (-1, -1),
             colors.HexColor("#EFF5FF"))
        ])
    )

    story.append(table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph("Available Skills", styles["Heading2"])
    )

    story.append(
        Paragraph(
            html.escape(", ".join(found) if found else "None"),
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph("Missing Skills", styles["Heading2"])
    )

    story.append(
        Paragraph(
            html.escape(", ".join(missing) if missing else "None"),
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph("Improvement Suggestions", styles["Heading2"])
    )

    for item in suggestions:

        story.append(
            Paragraph(
                "• " + html.escape(item),
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 7))

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Note: This is a prototype analysis. "
            "The ATS score and job match are estimates, "
            "not actual hiring predictions.",
            styles["Italic"]
        )
    )

    doc.build(story)

    return file_path


# ==========================================
# 8. PROFESSIONAL DASHBOARD
# ==========================================

def dashboard_html(
    role, match, ats, found, missing, suggestions
):

    available_badges = "".join(
        f'<span class="badge green">{html.escape(s)}</span>'
        for s in found
    )

    missing_badges = "".join(
        f'<span class="badge red">{html.escape(s)}</span>'
        for s in missing
    )

    suggestion_items = "".join(
        f"<li>{html.escape(s)}</li>"
        for s in suggestions
    )

    return f"""
    <div style="font-family:Arial; padding:18px;">

      <div style="
        background:linear-gradient(120deg,#172554,#2563EB);
        color:white; padding:25px; border-radius:16px;">

        <h1 style="margin:0;">AI Resume Dashboard</h1>
        <p style="margin-bottom:0;">
        Smart Resume Analysis & Job Recommendation
        </p>
      </div>

      <div style="
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(145px,1fr));
        gap:14px; margin-top:20px;">

        <div style="
          background:#EDE9FE;padding:20px;
          border-radius:14px;text-align:center;">
          <h2 style="color:#6D28D9;">{ats}%</h2>
          <b>ATS Score</b>
        </div>

        <div style="
          background:#DBEAFE;padding:20px;
          border-radius:14px;text-align:center;">
          <h2 style="color:#1D4ED8;">{match}%</h2>
          <b>Job Match</b>
        </div>

        <div style="
          background:#DCFCE7;padding:20px;
          border-radius:14px;text-align:center;">
          <h2 style="color:#15803D;">{len(found)}</h2>
          <b>Skills Found</b>
        </div>

        <div style="
          background:#FFEDD5;padding:20px;
          border-radius:14px;text-align:center;">
          <h2 style="color:#C2410C;">{len(missing)}</h2>
          <b>Missing Skills</b>
        </div>

      </div>

      <div style="
        background:#F1F5F9;padding:20px;
        margin-top:20px;border-radius:14px;">

        <h2 style="color:#173B70;">
        Recommended Job Role
        </h2>

        <h2 style="color:#2563EB;">{html.escape(role)}</h2>

      </div>

      <div style="margin-top:20px;">

        <h2 style="color:#15803D;">Available Skills</h2>

        {available_badges if available_badges else "No skills found"}

        <h2 style="color:#DC2626;margin-top:20px;">
        Missing Skills
        </h2>

        {missing_badges if missing_badges else "No missing skills"}

      </div>

      <div style="
        background:#FFF7ED;padding:20px;
        margin-top:20px;border-radius:14px;">

        <h2 style="color:#C2410C;">
        Resume Improvement Suggestions
        </h2>

        <ul>{suggestion_items}</ul>

      </div>

      <p style="color:#64748B;font-size:12px;margin-top:20px;">
      ATS score and job match are prototype estimates.
      </p>

    </div>

    <style>
    .badge {{
        display:inline-block;
        padding:8px 13px;
        margin:5px;
        border-radius:20px;
        font-size:13px;
        font-weight:bold;
    }}

    .green {{
        background:#DCFCE7;
        color:#166534;
    }}

    .red {{
        background:#FEE2E2;
        color:#991B1B;
    }}
    </style>
    """


# ==========================================
# 9. RESUME ANALYSIS
# ==========================================

def analyze_resume(file_path):

    if not file_path:
        return (
            "<h3>Please upload a PDF resume.</h3>",
            None,
            "<p>Upload a resume to view your dashboard.</p>"
        )

    text = extract_resume_text(file_path)

    if not text.strip():

        return (
            "<h3>Unable to extract text. Please upload a text-based PDF.</h3>",
            None,
            "<p>No readable resume text was found.</p>"
        )

    found = extract_skills(text)

    role, match = recommend_job_role(found)

    ats = calculate_ats_score(text, found)

    if role in required:
        missing = [
            skill for skill in required[role]
            if skill not in found
        ]
    else:
        missing = []

    suggestions = get_resume_suggestions(
        text, found
    )

    pdf_path = create_pdf_report(
        role, match, ats, found, missing, suggestions
    )

    dashboard = dashboard_html(
        role, match, ats, found, missing, suggestions
    )

    found_text = ", ".join(found) if found else "None"

    missing_text = ", ".join(missing) if missing else "None"

    suggestion_text = "".join(
        f"<li>{html.escape(s)}</li>"
        for s in suggestions
    )

    report = f"""
    <div style="
      font-family:Arial;
      border-radius:15px;
      overflow:hidden;
      border:1px solid #DCE5F1;">

      <div style="
        background:linear-gradient(120deg,#173B70,#2563EB);
        color:white;padding:23px;">

        <h2 style="margin:0;">Resume Analysis Report</h2>
        <p>Detailed Resume Evaluation</p>

      </div>

      <div style="padding:22px;background:#F8FAFC;">

        <h3 style="color:#173B70;">
        Recommended Job Role: {html.escape(role)}
        </h3>

        <p><b>Job Match:</b> {match}%</p>
        <p><b>ATS Score:</b> {ats}%</p>
        <p><b>Total Skills Found:</b> {len(found)}</p>

        <hr>

        <h3 style="color:#15803D;">Available Skills</h3>
        <p>{html.escape(found_text)}</p>

        <h3 style="color:#DC2626;">Missing Skills</h3>
        <p>{html.escape(missing_text)}</p>

        <hr>

        <h3 style="color:#C2410C;">
        Improvement Suggestions
        </h3>

        <ul>{suggestion_text}</ul>

        <p style="font-size:12px;color:#64748B;">
        Note: ATS score and job match are prototype estimates.
        </p>

      </div>
    </div>
    """

    return report, pdf_path, dashboard


# ==========================================
# 10. JOB DESCRIPTION MATCHING
# ==========================================

def check_job_description(file_path, job_description):

    if not file_path:
        return "<h3>Please upload your resume first.</h3>"

    if not job_description or not job_description.strip():
        return "<h3>Please enter a Job Description.</h3>"

    resume_text = extract_resume_text(file_path)

    if not resume_text.strip():
        return "<h3>Unable to read the uploaded PDF.</h3>"

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume_text, job_description]
    )

    similarity = cosine_similarity(
        vectors[0:1], vectors[1:]
    )[0][0]

    percentage = round(similarity * 100, 2)

    jd_skills = extract_skills(job_description)

    resume_skills = extract_skills(resume_text)

    matched = [
        skill for skill in jd_skills
        if skill in resume_skills
    ]

    missing = [
        skill for skill in jd_skills
        if skill not in resume_skills
    ]

    matched_badges = "".join(
        f'<span class="badge green">{html.escape(s)}</span>'
        for s in matched
    )

    missing_badges = "".join(
        f'<span class="badge red">{html.escape(s)}</span>'
        for s in missing
    )

    return f"""
    <div style="
      font-family:Arial;
      border:1px solid #DCE5F1;
      border-radius:16px;
      overflow:hidden;">

      <div style="
        background:linear-gradient(120deg,#172554,#2563EB);
        color:white;padding:24px;">

        <h2 style="margin:0;">
        Job Description Matching Report
        </h2>

        <p>Professional Resume & Job Comparison</p>

      </div>

      <div style="padding:22px;background:#F8FAFC;">

        <h2 style="color:#173B70;">
        Text Similarity: {percentage}%
        </h2>

        <div style="
          background:#DBEAFE;
          height:16px;
          border-radius:20px;
          overflow:hidden;">

          <div style="
            width:{percentage}%;
            background:#2563EB;
            height:16px;
            border-radius:20px;">
          </div>

        </div>

        <p style="color:#64748B;font-size:12px;">
        This percentage represents text similarity,
        not the probability of getting hired.
        </p>

        <hr>

        <h3 style="color:#15803D;">
        Matched Skills ({len(matched)})
        </h3>

        {matched_badges if matched_badges else "No matching skills found"}

        <h3 style="color:#DC2626;margin-top:22px;">
        Missing Skills ({len(missing)})
        </h3>

        {missing_badges if missing_badges else "No missing skills found"}

        <hr>

        <div style="
          background:#E0F2FE;
          padding:18px;
          border-radius:12px;">

          <h3 style="color:#075985;">
          Matching Summary
          </h3>

          <p>Total Required Skills: {len(jd_skills)}</p>
          <p>Matched Skills: {len(matched)}</p>
          <p>Missing Skills: {len(missing)}</p>

        </div>

      </div>
    </div>

    <style>
    .badge {{
        display:inline-block;
        padding:8px 13px;
        margin:5px;
        border-radius:20px;
        font-size:13px;
        font-weight:bold;
    }}

    .green {{
        background:#DCFCE7;
        color:#166534;
    }}

    .red {{
        background:#FEE2E2;
        color:#991B1B;
    }}
    </style>
    """


# ==========================================
# 11. GRADIO INTERFACE
# ==========================================
custom_css = """
.gradio-container {
    font-family: Arial, sans-serif;
}

#main-title {
    text-align: center;
    background: linear-gradient(120deg,#172554,#2563EB);
    color: white;
    padding: 25px;
    border-radius: 15px;
}

button.primary {
    background: #2563EB !important;
    color: white !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo"
    ),
    css=custom_css,
    title="AI Smart Resume Analyzer"
) as demo:

    gr.HTML("""
    <div id="main-title">
      <h1>AI SMART RESUME ANALYZER</h1>
      <p>Resume Analysis | ATS Score | Job Matching</p>
    </div>
    """)

    gr.Markdown(
        "### Upload your resume and analyze your career opportunities."
    )

    with gr.Tab("Resume Analysis"):

        file_input = gr.File(
            label="Upload Resume PDF",
            file_types=[".pdf"],
            type="filepath"
        )

        analyze_button = gr.Button(
            "Analyze Resume",
            variant="primary"
        )

        gr.Markdown("## Professional Dashboard")

        dashboard = gr.HTML()

        gr.Markdown("## Resume Analysis Report")

        result = gr.HTML()

        pdf_output = gr.File(
            label="Download Resume Analysis PDF"
        )

        analyze_button.click(
            fn=analyze_resume,
            inputs=file_input,
            outputs=[result, pdf_output, dashboard]
        )

    with gr.Tab("Job Description Matching"):

        gr.Markdown(
            "### Compare your resume with a job description."
        )

        jd_input = gr.Textbox(
            label="Enter Job Description",
            placeholder="Paste the job description here...",
            lines=8
        )

        jd_button = gr.Button(
            "Check Job Match",
            variant="primary"
        )

        jd_result = gr.HTML()

        jd_button.click(
            fn=check_job_description,
            inputs=[file_input, jd_input],
            outputs=jd_result
        )

    gr.Markdown("""
    ---
    **AI Smart Resume Analyzer & Job Matcher**

    Developed using Python, Machine Learning, NLP and Gradio.
    """)




# ==========================================
# 12. LAUNCH APPLICATION 

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
