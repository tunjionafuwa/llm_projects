from dotenv import load_dotenv

import numpy as np
import pandas as pd

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("./book-recommender/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "./book-recommender/cover-not-found.jpg",
    books["large_thumbnail"],
)

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
raw_documents = TextLoader("./book-recommender/tagged_description.txt").load()
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings(),
)


def retrieve_semantic_recommendations(
    query: str,
    category: str | None,
    tone: str | None,
    top_k: int = 16,
    k: int = 50,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query=query, k=k)
    ids = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(ids[:top_k])]

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:top_k]

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(
        query=query, tone=tone, category=category
    )
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc = description if len(description) < 30 else description[:30] + "..." 

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors = authors_split[0]

        caption = f"{row["title"]} by {authors}: {truncated_desc}"
        results.append((row["large_thumbnail"], caption))
    
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommendation books", columns=8, rows=2)
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )


if __name__ == "__main__":
    dashboard.launch()
