import streamlit as st
import pandas as pd
from main import load_resumes, load_job_description, REQUIRED_SKILLS, skill_match_percentage, find_skill_gaps
from matcher import ResumeMatcher

st.title("AI Resume Screening & Semantic Hiring System")

st.markdown("This system performs semantic resume-job matching and skill gap analysis.")

if st.button("Run Resume Screening"):

    df, resume_col = load_resumes()
    job_text = load_job_description()

    resume_texts = df[resume_col].dropna().astype(str).tolist()

    matcher = ResumeMatcher()
    scores = matcher.compute_batch_similarity(resume_texts, job_text)

    df = df.iloc[:len(scores)].copy()
    df["match_score"] = scores

    results = []

    for index, row in df.iterrows():

        resume_text = str(row[resume_col])
        score = row["match_score"]

        skill_match = skill_match_percentage(resume_text, REQUIRED_SKILLS)
        missing_skills = find_skill_gaps(resume_text, REQUIRED_SKILLS)

        if score >= 0.85 and skill_match >= 80:
            decision = "Perfect Candidate"
        elif score >= 0.75 and skill_match >= 60:
            decision = "Strong Match"
        elif score >= 0.60 and skill_match >= 40:
            decision = "Moderate Match"
        else:
            decision = "Not Suitable"

        results.append({
            "Resume ID": row.get("ID", index),
            "Category": row.get("Category", "N/A"),
            "Match Score": round(score, 4),
            "Skill Match %": skill_match,
            "Decision": decision,
            "Missing Skills": ", ".join(missing_skills) if missing_skills else "None"
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Match Score", ascending=False)

    st.subheader("Screening Results")
    st.dataframe(results_df)

    eligible = results_df[
        results_df["Decision"].isin(["Perfect Candidate", "Strong Match"])
    ]

    st.subheader("Eligible Candidates")

    if eligible.empty:
        st.warning("No eligible candidates found.")
    else:
        st.success(f"{len(eligible)} candidates shortlisted.")
        st.dataframe(eligible)
