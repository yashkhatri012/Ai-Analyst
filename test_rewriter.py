import asyncio
from app.session.store import create_session, add_turn
from app.agents.rewriter_agent import rewrite

DATASET = "0b4580e7-1564-434a-91dd-8af7fd74a05d"


def session_with_result():
    s = create_session(DATASET)
    add_turn(
        s,
        question="Analyze the relationship between budget, revenue, popularity, and vote_count.",
        standalone_question="Analyze the relationship between budget, revenue, popularity, and vote_count.",
        status="completed",
        answer="Popularity and vote_count are most strongly related (Spearman 0.96). "
               "Budget and revenue correlate at 0.71 (n=3229, 1,574 rows excluded).",
        steps=[{"goal": "Pairwise Pearson and Spearman correlations",
                "columns": ["budget", "revenue", "popularity", "vote_count"]}],
        assumptions=["Rows where budget or revenue equal 0 were excluded (treated as missing)."],
    )
    return s


def session_with_pending_clarification():
    s = create_session(DATASET)
    add_turn(
        s,
        question="Which genre is best?",
        standalone_question="Which genre is best?",
        status="unanswerable",
        clarification="Best by which metric: revenue, popularity, or vote_count?",
    )
    return s


CASES = [
    ("A", "now break that down by genre",              "refine"),
    ("A", "why were so many rows excluded?",           "history"),
    ("A", "what is the average runtime?",              "independent"),
    ("A", "include the zeros this time",               "refine"),
    ("B", "revenue",                                   "clarification"),
]


async def main():
    for which, question, expected in CASES:
        s = session_with_result() if which == "A" else session_with_pending_clarification()
        r = await rewrite(question, s)
        ok = "OK " if r.kind == expected else "BAD"
        print(f"{ok} expected={expected:<13} got={r.kind:<13} | {question}")
        print(f"    standalone: {r.standalone_question}")
        print(f"    overrides: {r.user_overrides}  ask_user: {r.ask_user}\n")

asyncio.run(main())