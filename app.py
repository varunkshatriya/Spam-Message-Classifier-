# =========================================
# 🚀 AI RESUME SCREENER USING GRADIO
# =========================================

# Install libraries in Google Colab first:
# !pip install gradio pandas scikit-learn PyPDF2

import gradio as gr
import PyPDF2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# JOB ROLE SKILLS DATABASE
# =========================================

job_roles = {

    "Data Scientist": [
        "python",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pandas",
        "numpy",
        "sql",
        "statistics"
    ],

    "AI Engineer": [
        "python",
        "tensorflow",
        "pytorch",
        "nlp",
        "deep learning",
        "machine learning"
    ],

    "Web Developer": [
        "html",
        "css",
        "javascript",
        "react",
        "mongodb",
        "frontend",
        "backend"
    ],

    "Python Developer": [
        "python",
        "flask",
        "django",
        "api",
        "sql",
        "git"
    ]
}

# =========================================
# PDF TEXT EXTRACTION
# =========================================

def extract_text(pdf_file):

    text = ""

    try:

        # Open uploaded PDF correctly
        with open(pdf_file.name, "rb") as file:

            pdf_reader = PyPDF2.PdfReader(file)

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + " "

    except Exception as e:

        text = f"Error Reading PDF: {e}"

    return text.lower()

# =========================================
# SKILL EXTRACTION
# =========================================

def extract_skills(resume_text):

    detected_skills = []

    all_skills = []

    # Collect all skills
    for skills in job_roles.values():
        all_skills.extend(skills)

    # Find skills in resume
    for skill in set(all_skills):

        if skill.lower() in resume_text:
            detected_skills.append(skill)

    return detected_skills

# =========================================
# RESUME SUMMARY
# =========================================

def generate_summary(resume_text):

    words = resume_text.split()

    summary = " ".join(words[:80])

    return summary

# =========================================
# MAIN ANALYSIS FUNCTION
# =========================================

def analyze_resume(pdf_file, selected_role):

    # Extract text from PDF
    resume_text = extract_text(pdf_file)

    # DEBUG PRINT
    print("\n========== EXTRACTED TEXT ==========\n")
    print(resume_text)

    # Extract skills
    extracted_skills = extract_skills(resume_text)

    # Required skills
    required_skills = job_roles[selected_role]

    # Missing skills
    missing_skills = []

    for skill in required_skills:

        if skill not in extracted_skills:
            missing_skills.append(skill)

    # =====================================
    # MACHINE LEARNING SIMILARITY
    # =====================================

    documents = [
        " ".join(extracted_skills),
        " ".join(required_skills)
    ]

    cv = CountVectorizer()

    matrix = cv.fit_transform(documents)

    similarity = cosine_similarity(matrix)[0][1]

    score = round(similarity * 100, 2)

    # =====================================
    # SUGGESTIONS
    # =====================================

    suggestions = []

    if score >= 80:

        suggestions.append("✅ Excellent Resume Match")
        suggestions.append("✅ Strong Technical Skills")

    elif score >= 50:

        suggestions.append("⚠️ Add More Relevant Skills")
        suggestions.append("⚠️ Improve Project Experience")

    else:

        suggestions.append("❌ Add More Skills")
        suggestions.append("❌ Add Projects & Certifications")

    # =====================================
    # FINAL OUTPUT
    # =====================================

    result = f"""

# 🚀 AI Resume Screening Result

---

# 🎯 Selected Role
### {selected_role}

---

# 📊 Resume Match Score
# ✅ {score} %

---

# 🧠 Detected Skills

{", ".join(extracted_skills)}

---

# ❌ Missing Skills

{", ".join(missing_skills)}

---

# 💡 Suggestions

{chr(10).join(suggestions)}

---

# 📝 Resume Summary

{generate_summary(resume_text)}

"""

    return result

# =========================================
# GRADIO UI
# =========================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""

# 🚀 AI Resume Screener

### 📄 Upload Resume PDF → Analyze Skills → Get Resume Score

Built with ❤️ using Python + Gradio

""")

    with gr.Row():

        pdf_input = gr.File(
            label="📎 Upload Resume PDF",
            file_types=[".pdf"]
        )

        role_input = gr.Dropdown(
            choices=list(job_roles.keys()),
            value="Data Scientist",
            label="💼 Select Job Role"
        )

    analyze_btn = gr.Button("🔍 Analyze Resume")

    output = gr.Markdown()

    analyze_btn.click(
        fn=analyze_resume,
        inputs=[pdf_input, role_input],
        outputs=output
    )

# =========================================
# RUN APP
# =========================================

demo.launch(share=True)
