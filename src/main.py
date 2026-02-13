from pathlib import Path
import pandas as pd

from preprocess import clean_text
from matcher import ResumeMatcher

from typing import cast

# CONFIG

BASE_DIR = Path(__file__).resolve().parent.parent

REQUIRED_SKILLS = [
    "recruitment",
    "onboarding",
    "employee relations",
    "payroll",
    "attendance",
    "hr operations",
    "labor laws",
    "hr policies",
    "performance management",
    "training"
]

# SKILL GAP FUNCTIONS

def find_skill_gaps(resume_text, required_skills):
    resume_text = resume_text.lower()
    return [skill for skill in required_skills if skill not in resume_text]


def skill_match_percentage(resume_text, required_skills):
    resume_text = resume_text.lower()
    matched = sum(1 for skill in required_skills if skill in resume_text)
    return round((matched / len(required_skills)) * 100, 2)

# LOAD DATA

def load_resumes():
    path = BASE_DIR / "data" / "Resume.csv"

    if not path.exists():
        raise FileNotFoundError("Resume.csv not found")

    df = pd.read_csv(path)

    resume_col = None
    for col in ["Resume_str", "resume_str", "Resume_text", "resume_text", "text"]:
        if col in df.columns:
            resume_col = col
            break

    if resume_col is None:
        raise ValueError("No resume text column found")

    return df.head(50).copy(), resume_col


def load_job_description():
    path = BASE_DIR / "data" / "job_description.txt"

    if not path.exists():
        raise FileNotFoundError("job_description.txt not found")

    return clean_text(path.read_text(encoding="utf-8"))


# MAIN PIPELINE

def main():

    df, resume_col = load_resumes()
    job_text = load_job_description()

    resume_texts = df[resume_col].dropna().astype(str).tolist()

    matcher = ResumeMatcher()

    print("\nEncoding resumes and computing similarity...\n")

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
            "Missing Skills": missing_skills
        })

    results_df = pd.DataFrame(results)

    # Filter eligible candidates
    eligible = results_df[
        results_df["Decision"].isin([
            "Perfect Candidate",
            "Strong Match",
        ])
    ].copy()

    eligible["Skill Gaps"] = eligible["Missing Skills"].apply(
        lambda x: ", ".join(x) if x else "None"
    )

    eligible = cast(pd.DataFrame, eligible)
    eligible = eligible.sort_values(by=["Match Score"], ascending=False)
    eligible = eligible.reset_index(drop=True)


    print("\n===== ELIGIBLE CANDIDATES =====\n")

    if eligible.empty:
        print("No candidates meet the eligibility criteria.")
    else:
        display_table = eligible[[
            "Resume ID",
            "Category",
            "Match Score",
            "Skill Match %",
            "Decision",
            "Skill Gaps"
        ]]
        print(display_table.to_string(index=False))


# ENTRY POINT

if __name__ == "__main__":
    main()
