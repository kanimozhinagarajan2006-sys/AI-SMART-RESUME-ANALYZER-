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
    "HTML", "CSS", "JavaScript", "Power BI",
    "R", "c", "c++"
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

    except Exception:
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
            ("TEXTCOLOR", (0, 1), (-1, -1),
             colors.black),
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
    role, match, ats, found, missing,