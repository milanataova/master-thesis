"""
05_synthetic_generation.py

Generate synthetic survey responses using LLMs.
Generates 30 responses per (country x income x health) group.
Resumes from where it stopped if interrupted.

Before running:
    1. Update API_BASE_URL and API_KEY
    2. Update MODEL_ID for your LLM provider
"""

import os
import re
import time
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============================================================================
# Configuration - UPDATE API_KEY BEFORE RUNNING
# ============================================================================

API_BASE_URL = "https://chat-ai.academiccloud.de/v1"
API_KEY = "your-api-key"  # UPDATE THIS

# Choose one model configuration (uncomment the one you want to use):

# Option 1: LLaMA 3.1 8B
MODEL_ID = "meta-llama-3.1-8b-instruct"
MODEL_NAME = "llama3.1_8b"

# Option 2: LLaMA 3.3 70B
# MODEL_ID = "llama-3.3-70b-instruct"
# MODEL_NAME = "llama3.3_70b"

# Option 3: Qwen 2.5 72B
# MODEL_ID = "qwen2.5-vl-72b-instruct"
# MODEL_NAME = "qwen2.5_72b"

# ============================================================================
# Fixed Settings
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "Datasets" / "synthetic_data" / "synthetic_raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNTRIES = ["USA", "IDN", "NLD", "MEX"]
INCOME_LEVELS = ["Low", "Medium", "High"]
HEALTH_LEVELS = ["Good", "Fair", "Poor"]
N_SAMPLES = 30

# ============================================================================
# Questionnaire Prompts
# ============================================================================

def prompt_base(country, income, health):
    """Standard WVS life satisfaction question (1-10)."""
    return (
        "You are an interviewee. Based on your previous answers, provide an "
        "answer to the last question.\n"
        f"Interviewer: Where do you live?\nInterviewee: I live in {country}.\n"
        f"Interviewer: What is your income level?\nInterviewee: My income level is {income}.\n"
        "Interviewer: All in all, how would you describe your state of health these days? "
        f"Would you say it is good, fair, or poor?\nInterviewee: My health is {health}.\n"
        "Interviewer: All things considered, how satisfied are you with your life as a whole these days? "
        "Please respond with a number from 1 to 10, where 10 = completely satisfied and 1 = completely dissatisfied. "
        "Reply with a single number from 1 to 10. Do not explain your answer.\nInterviewee:"
    )

def prompt_cantril(country, income, health):
    """Cantril Ladder - ladder metaphor (1-10)."""
    return (
        "You are an interviewee. Based on your previous answers, provide an "
        "answer to the last question.\n"
        f"Interviewer: Where do you live?\nInterviewee: I live in {country}.\n"
        f"Interviewer: What is your income level?\nInterviewee: My income level is {income}.\n"
        "Interviewer: All in all, how would you describe your state of health these days? "
        f"Would you say it is good, fair, or poor?\nInterviewee: My health is {health}.\n"
        "Interviewer: Please imagine a ladder with steps numbered from 1 at the bottom to 10 at the top. "
        "The top of the ladder represents the best possible life for you, and the "
        "bottom of the ladder represents the worst possible life for you. "
        "On which step of the ladder would you say you personally feel you stand at this time? "
        "Reply with a single number from 1 to 10. Do not explain your answer.\nInterviewee:"
    )

def prompt_reverse(country, income, health):
    """Reverse Scale - 1=satisfied, 10=dissatisfied."""
    return (
        "You are an interviewee. Based on your previous answers, provide an "
        "answer to the last question.\n"
        f"Interviewer: Where do you live?\nInterviewee: I live in {country}.\n"
        f"Interviewer: What is your income level?\nInterviewee: My income level is {income}.\n"
        "Interviewer: All in all, how would you describe your state of health these days? "
        f"Would you say it is good, fair, or poor?\nInterviewee: My health is {health}.\n"
        "Interviewer: All things considered, how satisfied are you with your life as a whole these days? "
        "Please respond with a number from 1 to 10, where 1 = completely satisfied and 10 = completely dissatisfied. "
        "Reply with a single number from 1 to 10. Do not explain your answer.\nInterviewee:"
    )

SWLS_QUESTIONS = [
    "In most ways my life is close to my ideal.",
    "The conditions of my life are excellent.",
    "I am satisfied with my life.",
    "So far I have gotten the important things I want in life.",
    "If I could live my life over, I would change almost nothing."
]

def prompt_swls(country, income, health):
    """SWLS - 5 items, 1-7 scale each."""
    questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(SWLS_QUESTIONS)])
    return (
        "You are an interviewee. Based on your previous answers, provide answers "
        "to the following questions.\n"
        f"Interviewer: Where do you live?\nInterviewee: I live in {country}.\n"
        f"Interviewer: What is your income level?\nInterviewee: My income level is {income}.\n"
        "Interviewer: All in all, how would you describe your state of health these days? "
        f"Would you say it is good, fair, or poor?\nInterviewee: My health is {health}.\n"
        "Interviewer: Below are five statements. Using the scale below, indicate your agreement.\n\n"
        "Scale: 1=strongly disagree, 2=disagree, 3=slightly disagree, 4=neutral, "
        "5=slightly agree, 6=agree, 7=strongly agree\n\n"
        f"Statements:\n{questions}\n\n"
        "Reply with exactly 5 numbers separated by commas (e.g., 5, 6, 4, 5, 3). "
        "Do not explain.\nInterviewee:"
    )

OHQ_QUESTIONS = [
    "I don't feel particularly pleased with the way I am. (R)",
    "I am intensely interested in other people.",
    "I feel that life is very rewarding.",
    "I have very warm feelings towards almost everyone.",
    "I rarely wake up feeling rested. (R)",
    "I am not particularly optimistic about the future. (R)",
    "I find most things amusing.",
    "I am always committed and involved.",
    "Life is good.",
    "I do not think that the world is a good place. (R)",
    "I laugh a lot.",
    "I am well satisfied about everything in my life.",
    "I don't think I look attractive. (R)",
    "There is a gap between what I would like to do and what I have done. (R)",
    "I am very happy.",
    "I find beauty in some things.",
    "I always have a cheerful effect on others.",
    "I can fit in (find time for) everything I want to.",
    "I feel that I am not especially in control of my life. (R)",
    "I feel able to take anything on.",
    "I feel fully mentally alert.",
    "I often experience joy and elation.",
    "I don't find it easy to make decisions. (R)",
    "I don't have a particular sense of meaning and purpose in life. (R)",
    "I feel I have a great deal of energy.",
    "I usually have a good influence on events.",
    "I don't have fun with other people. (R)",
    "I don't feel particularly healthy. (R)",
    "I don't have particularly happy memories of the past. (R)"
]
OHQ_REVERSE = [1, 5, 6, 10, 13, 14, 19, 23, 24, 27, 28, 29]

def prompt_ohq(country, income, health):
    """OHQ - 29 items, 1-6 scale each."""
    questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(OHQ_QUESTIONS)])
    return (
        "You are an interviewee. Based on your previous answers, provide answers "
        "to the following questions.\n"
        f"Interviewer: Where do you live?\nInterviewee: I live in {country}.\n"
        f"Interviewer: What is your income level?\nInterviewee: My income level is {income}.\n"
        "Interviewer: All in all, how would you describe your state of health these days? "
        f"Would you say it is good, fair, or poor?\nInterviewee: My health is {health}.\n"
        "Interviewer: Below are 29 statements. Using the scale below, indicate your agreement.\n\n"
        "Scale: 1=strongly disagree, 2=moderately disagree, 3=slightly disagree, "
        "4=slightly agree, 5=moderately agree, 6=strongly agree\n\n"
        f"Statements:\n{questions}\n\n"
        "Reply with exactly 29 numbers separated by commas. Do not explain.\nInterviewee:"
    )

# ============================================================================
# Score Extraction
# ============================================================================

def extract_score_1_10(text):
    """Extract single score 1-10."""
    matches = re.findall(r'\b(10|[1-9])\b', text)
    if matches:
        score = int(matches[0])
        if 1 <= score <= 10:
            return {"score": score}
    return None

def extract_swls(text):
    """Extract 5 SWLS scores (1-7 each) and compute derived columns."""
    numbers = re.findall(r'\b[1-7]\b', text)
    if len(numbers) >= 5:
        scores = [int(n) for n in numbers[:5]]
        total = sum(scores)

        # Map total (5-35) to 7-scale and label
        if total >= 31:
            scale_7, label = 7, "Highly Satisfied"
        elif total >= 26:
            scale_7, label = 6, "Satisfied"
        elif total >= 21:
            scale_7, label = 5, "Slightly Satisfied"
        elif total == 20:
            scale_7, label = 4, "Neutral"
        elif total >= 15:
            scale_7, label = 3, "Slightly Dissatisfied"
        elif total >= 10:
            scale_7, label = 2, "Dissatisfied"
        else:
            scale_7, label = 1, "Highly Dissatisfied"

        return {
            "q1": scores[0], "q2": scores[1], "q3": scores[2],
            "q4": scores[3], "q5": scores[4],
            "score": total,
            "life_satisfaction_7scale": scale_7,
            "satisfaction_label": label
        }
    return None

def extract_ohq(text):
    """Extract 29 OHQ scores (1-6 each), apply reverse scoring."""
    numbers = re.findall(r'\b[1-6]\b', text)
    if len(numbers) >= 29:
        scores = [int(n) for n in numbers[:29]]
        result = {f"q{i+1}": scores[i] for i in range(29)}
        # Reverse scoring
        for item in OHQ_REVERSE:
            result[f"q{item}"] = 7 - result[f"q{item}"]
        result["ohq_score"] = sum(result[f"q{i+1}"] for i in range(29)) / 29
        return result
    return None

# ============================================================================
# Questionnaire Configs
# ============================================================================

QUESTIONNAIRES = {
    "base": {"prompt": prompt_base, "extract": extract_score_1_10, "max_tokens": 10,
             "columns": ["country", "income_level", "health_level", "priming", "role_adoption", "response", "score"]},
    "cantril": {"prompt": prompt_cantril, "extract": extract_score_1_10, "max_tokens": 10,
                "columns": ["country", "income_level", "health_level", "priming", "role_adoption", "response", "score"]},
    "reverse": {"prompt": prompt_reverse, "extract": extract_score_1_10, "max_tokens": 10,
                "columns": ["country", "income_level", "health_level", "priming", "role_adoption", "response", "score"]},
    "swls": {"prompt": prompt_swls, "extract": extract_swls, "max_tokens": 50,
             "columns": ["country", "income_level", "health_level", "priming", "role_adoption", "response",
                        "q1", "q2", "q3", "q4", "q5", "score", "life_satisfaction_7scale", "satisfaction_label"]},
    "ohq": {"prompt": prompt_ohq, "extract": extract_ohq, "max_tokens": 150,
            "columns": ["country", "income_level", "health_level", "priming", "role_adoption", "response"] +
                      [f"q{i}" for i in range(1, 30)] + ["ohq_score"]},
}

# ============================================================================
# Generation Function
# ============================================================================

def generate(questionnaire_name):
    """Generate responses for one questionnaire type."""
    config = QUESTIONNAIRES[questionnaire_name]
    output_file = OUTPUT_DIR / f"synthetic_{questionnaire_name}_{MODEL_NAME}.csv"

    print(f"\n{'='*50}")
    print(f"Generating: {questionnaire_name} with {MODEL_NAME}")
    print(f"Output: {output_file}")
    print(f"{'='*50}")

    # Load existing or create new
    if output_file.exists():
        df = pd.read_csv(output_file)
        print(f"Loaded existing: {len(df)} rows")
    else:
        df = pd.DataFrame(columns=config["columns"])
        print("Starting fresh")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    new_rows = []

    for country in TARGET_COUNTRIES:
        for income in INCOME_LEVELS:
            for health in HEALTH_LEVELS:
                # Count existing for this group
                mask = (df["country"] == country) & (df["income_level"] == income) & (df["health_level"] == health)
                existing = mask.sum()

                if existing >= N_SAMPLES:
                    continue

                to_generate = N_SAMPLES - existing
                print(f"{country}/{income}/{health}: need {to_generate} more")

                for _ in tqdm(range(to_generate), desc=f"{country}-{income}-{health}", leave=False):
                    prompt = config["prompt"](country, income, health)

                    for attempt in range(5):
                        try:
                            response = client.chat.completions.create(
                                model=MODEL_ID,
                                messages=[
                                    {"role": "system", "content": "You are a synthetic survey respondent."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=1.0,
                                max_tokens=config["max_tokens"]
                            )
                            reply = response.choices[0].message.content.strip()
                            result = config["extract"](reply)

                            if result:
                                row = {
                                    "country": country,
                                    "income_level": income,
                                    "health_level": health,
                                    "priming": "explicit",
                                    "role_adoption": "interview",
                                    "response": reply,
                                    **result
                                }
                                new_rows.append(row)
                                break
                        except Exception as e:
                            if "rate limit" in str(e).lower() or "429" in str(e):
                                wait = min(60 * (2 ** attempt), 3600)
                                print(f"\nRate limit, waiting {wait}s...")
                                time.sleep(wait)
                            else:
                                print(f"\nError: {e}")
                                time.sleep(5)

                    # Save every 10 new rows
                    if len(new_rows) % 10 == 0 and new_rows:
                        df_new = pd.DataFrame(new_rows)
                        df_combined = pd.concat([df, df_new], ignore_index=True)
                        df_combined.to_csv(output_file, index=False)

    # Final save
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, df_new], ignore_index=True)
        df_combined.to_csv(output_file, index=False)
        print(f"Saved {len(new_rows)} new rows. Total: {len(df_combined)}")
    else:
        print("No new rows needed")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Synthetic Response Generation")
    print(f"Model: {MODEL_NAME}")
    print(f"Countries: {TARGET_COUNTRIES}")
    print(f"Samples per group: {N_SAMPLES}")
    print(f"Total groups per questionnaire: {len(TARGET_COUNTRIES) * len(INCOME_LEVELS) * len(HEALTH_LEVELS)}")

    # Generate for each questionnaire
    for q_name in QUESTIONNAIRES:
        generate(q_name)

    print("\nDone!")
