import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

CATEGORIES = [
    "Food & Drink", "Transport", "Entertainment", "Shopping",
    "Health & Fitness", "Utilities", "Rent", "Travel",
    "Salary", "Investment", "Other"
]

CATEGORY_MAP = {}

MODEL_CANDIDATES = [
    "cross-encoder/nli-MiniLM2-L6-H768",
    "facebook/bart-large-mnli",
    "typeform/distilbert-base-uncased-mnli"
]

SAVING_RULES = [
    ('Food & Drink', 500, "Food & Drink spending is high. Consider meal prepping at home."),
    ('Entertainment', 300, "Entertainment budget is significant. Review subscriptions."),
    ('Shopping', 400, "Shopping expenses are elevated. Try a 24-hour rule."),
    ('Travel', 800, "Travel costs are notable. Book in advance."),
    ('Utilities', 300, "Utilities are above average. Check for better plans."),
    ('Transport', 200, "Transport spending is high. Consider public transit."),
    ('Health & Fitness', 400, "Health & Fitness costs are high. Look for free workout options."),
]

classifier = None
best_model_name = None

def select_best_model(sample_texts):
    print("Selecting best model for current dataset...")

    best_model = None
    best_score = -1

    for model_name in MODEL_CANDIDATES:
        try:
            clf = pipeline("zero-shot-classification", model=model_name)

            total_score = 0
            for text in sample_texts:
                result = clf(text, CATEGORIES)
                total_score += result['scores'][0]

            avg_score = total_score / len(sample_texts)

            print(f"{model_name} score: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_model = model_name

        except Exception as e:
            print(f"Model {model_name} failed: {e}")

    print(f"Selected model: {best_model}")
    return best_model

def map_category(original: str) -> str:
    if original in CATEGORY_MAP:
        return CATEGORY_MAP[original]

    result = classifier(original, CATEGORIES)
    mapped = result['labels'][0]
    CATEGORY_MAP[original] = mapped
    return mapped

def generate_advice(exp_by_cat, savings_rate):
    tips = []

    for cat, threshold, msg in SAVING_RULES:
        if cat in exp_by_cat and exp_by_cat[cat] > threshold:
            tips.append(f"- {msg}")
        if len(tips) == 3:
            break

    if savings_rate < 20:
        tips.append(f"- Your savings rate is {savings_rate:.1f}%. Aim for at least 20%.")

    if not tips:
        tips.append("- Great job! Your spending looks well balanced.")

    return "\n".join(tips[:3])

@app.route("/", methods=["GET", "POST"])
def index():
    global classifier, best_model_name

    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded.", 400

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        df['Date'] = pd.to_datetime(df['Date'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        df['Description'] = df['Transaction Description'].fillna('').astype(str).str.strip()
        df['Type'] = df['Type'].fillna('').astype(str).str.strip()
        df['Month'] = df['Date'].dt.to_period('M')

        sample_texts = df['Transaction Description'].dropna().astype(str).head(10).tolist()

        if len(sample_texts) == 0:
            sample_texts = ["sample transaction"]

        print("Re-evaluating models for this upload...")

        best_model_name = select_best_model(sample_texts)
        classifier = pipeline("zero-shot-classification", model=best_model_name)

        print(f"Using model: {best_model_name}")

        df['FINAL_CATEGORY'] = df['Category'].fillna('Other').apply(map_category)

        expenses = df[df['Type'] == 'Expense'].copy()
        income = df[df['Type'] == 'Income'].copy()

        exp_by_cat = expenses.groupby('FINAL_CATEGORY')['Amount'].sum().sort_values(ascending=False)

        total_expense = expenses['Amount'].sum()
        total_income = income['Amount'].sum()
        net_savings = total_income - total_expense

        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0

        monthly = df.groupby(['Month', 'Type'])['Amount'].sum().unstack(fill_value=0)

        advice = generate_advice(exp_by_cat, savings_rate)

        summary = {
            "income": f"{total_income:,.2f}",
            "expense": f"{total_expense:,.2f}",
            "net": f"{net_savings:,.2f}",
            "rate": f"{savings_rate:.1f}",
            "advice": advice
        }

        os.makedirs("static", exist_ok=True)

        bar_path = "static/bar.png"
        pie_path = "static/pie.png"
        line_path = "static/line.png"

        plt.figure(figsize=(6, 4))
        exp_by_cat.plot(kind='bar', color='steelblue', edgecolor='white')
        plt.title("Expenses by Category")
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()

        plt.figure(figsize=(6, 4))
        exp_by_cat.plot(kind='pie', autopct='%1.1f%%', startangle=140)
        plt.title("Expense Distribution")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(pie_path)
        plt.close()

        plt.figure(figsize=(6, 4), dpi=300)
        monthly.plot(marker='o')
        plt.title("Monthly Income vs Expense")
        plt.tight_layout()
        plt.savefig(line_path)
        plt.close()

        return render_template(
            "result.html",
            bar=bar_path,
            pie=pie_path,
            line=line_path,
            summary=summary
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)