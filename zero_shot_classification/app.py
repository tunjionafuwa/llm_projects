from dotenv import load_dotenv

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline

load_dotenv(dotenv_path="../.env")

classifier = classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device="mps"
)
sequence = """
Over the past decade, renewable energy sources such as solar and wind have grown at an
unprecedented rate. Governments around the world are investing billions into clean energy
projects, aiming to reduce dependence on fossil fuels and combat climate change. The rise
of electric vehicles has also spurred demand for better battery technology and charging
infrastructure. However, challenges remain: supply chains for rare earth materials are
strained, grid stability is still a concern, and political disagreements slow progress
in some regions. Despite these hurdles, the global trend toward sustainability and
innovation in energy technology appears unstoppable, with new breakthroughs in hydrogen
fuel cells and large-scale energy storage systems on the horizon.
"""
candidate_labels = [
    "technology",
    "politics",
    "climate change",
    "economy",
    "sports",
    "entertainment",
]
response = classifier(sequence, candidate_labels, multi_label=True)

threshold = 0.8
scores = np.array(response["scores"])
valid_score = scores >= threshold
pred = np.array(response["labels"])[valid_score].tolist()
true_labels = [
    ['technology', 'climate change', 'politics']
]

mlb = MultiLabelBinarizer(classes=candidate_labels)
y_true = mlb.fit_transform(true_labels)
y_pred = mlb.transform([pred])

print(classification_report(y_true=y_true, y_pred=y_pred))
