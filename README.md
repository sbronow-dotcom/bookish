# 📖 Bookish — Two-Stage Book Recommender with LLM Re-Ranking

**[▶ Live demo](https://bookish-app.streamlit.app/)** · **[🎥 Video walkthrough](https://drive.google.com/file/d/1zCsk9pohiW-Qc2TAASAwLQIo4DtRB_b0/view?usp=sharing)** · Built by Sammi Bronow

Bookish recommends books in two stages. A tuned **item-based collaborative filtering** model generates personalized candidates from a reader's rating history. Then an **LLM re-ranker** reorders those candidates to match what the reader asks for in their own words ("a twisty mystery for a rainy Sunday") and explains each pick.

I then built an **offline evaluation harness** for the LLM layer. It uses LLM-as-judge scorers in Braintrust to compare prompt variants and pick the one to ship.

---

## Architecture

```mermaid
flowchart LR
    A[Reader rating history] --> B[Item-based CF<br/>pearson_baseline, k=20]
    B -->|top 100 by predicted rating| C[Candidate pool]
    C -->|top 40, shuffled| D[Gemini re-ranker<br/>schema-enforced JSON]
