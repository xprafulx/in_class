from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DB_PATH = Path("data/news.db")
SUMMARY_PATH = Path("outputs/run_summary.json")
DOCS_DIR = Path("docs")
OUTPUT_HTML = DOCS_DIR / "index.html"


def parse_time_published(value: str) -> datetime | None:
    """
    Parse AlphaVantage-like timestamps such as:
    - 20260312T102309
    - 20260312T1023
    """
    if not value:
        return None

    value = value.strip()

    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def choose_text_column(columns: list[str]) -> str | None:
    for candidate in ("summary", "text", "content", "body"):
        if candidate in columns:
            return candidate
    return None


def load_articles() -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    columns = get_table_columns(conn, "articles")
    text_col = choose_text_column(columns)

    select_cols = [
        "title",
        "url",
        "source",
        "time_published",
        "sentiment_score",
        "sentiment_label",
        "tickers",
    ]
    if text_col:
        select_cols.append(text_col)

    sql = f"SELECT {', '.join(select_cols)} FROM articles"
    rows = conn.execute(sql).fetchall()
    conn.close()

    articles: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["parsed_dt"] = parse_time_published(item.get("time_published", ""))
        item["text_for_summary"] = item.get(text_col, "") if text_col else ""
        articles.append(item)

    return articles


def filter_last_n_days(articles: list[dict[str, Any]], days: int = 7) -> list[dict[str, Any]]:
    dated = [a for a in articles if a.get("parsed_dt") is not None]
    if not dated:
        return []

    latest_dt = max(a["parsed_dt"] for a in dated)
    start_date = (latest_dt.date() - timedelta(days=days - 1))

    return [a for a in dated if a["parsed_dt"].date() >= start_date]


def build_daily_aggregates(articles: list[dict[str, Any]]) -> dict[str, Any]:
    if not articles:
        return {
            "labels": [],
            "avg_scores": [],
            "label_counts": {},
            "article_counts": [],
            "latest_day": None,
            "latest_day_groups": {},
        }

    # Sort by datetime
    articles = sorted(articles, key=lambda x: x["parsed_dt"])

    all_dates = sorted({a["parsed_dt"].date() for a in articles})
    labels = [d.isoformat() for d in all_dates]

    all_sentiment_labels = sorted(
        {str(a.get("sentiment_label", "Unknown")) for a in articles if a.get("sentiment_label")}
    )

    counts_by_day = defaultdict(int)
    score_sums_by_day = defaultdict(float)
    score_counts_by_day = defaultdict(int)
    label_counts_by_day = {label: defaultdict(int) for label in all_sentiment_labels}

    for article in articles:
        day = article["parsed_dt"].date().isoformat()
        counts_by_day[day] += 1

        score = safe_float(article.get("sentiment_score"), 0.0)
        score_sums_by_day[day] += score
        score_counts_by_day[day] += 1

        sentiment_label = str(article.get("sentiment_label", "Unknown"))
        if sentiment_label not in label_counts_by_day:
            label_counts_by_day[sentiment_label] = defaultdict(int)
        label_counts_by_day[sentiment_label][day] += 1

    avg_scores = []
    article_counts = []
    for day in labels:
        article_counts.append(counts_by_day[day])
        if score_counts_by_day[day] > 0:
            avg_scores.append(round(score_sums_by_day[day] / score_counts_by_day[day], 4))
        else:
            avg_scores.append(0.0)

    label_counts = {
        label: [label_counts_by_day[label][day] for day in labels]
        for label in sorted(label_counts_by_day.keys())
    }

    latest_day = labels[-1]
    latest_day_articles = [a for a in articles if a["parsed_dt"].date().isoformat() == latest_day]

    latest_day_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for article in latest_day_articles:
        latest_day_groups[str(article.get("sentiment_label", "Unknown"))].append(
            {
                "title": str(article.get("title", "")),
                "source": str(article.get("source", "")),
                "url": str(article.get("url", "")),
                "text": str(article.get("text_for_summary", ""))[:300],
            }
        )

    return {
        "labels": labels,
        "avg_scores": avg_scores,
        "label_counts": label_counts,
        "article_counts": article_counts,
        "latest_day": latest_day,
        "latest_day_groups": dict(latest_day_groups),
    }


def load_run_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        return {}
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_latest_day_html(latest_day: str | None, latest_day_groups: dict[str, list[dict[str, str]]]) -> str:
    if not latest_day:
        return "<p>No recent articles found.</p>"

    parts = [f"<h2>Latest day details ({latest_day})</h2>"]

    for sentiment_label, items in sorted(latest_day_groups.items()):
        parts.append(f"<div class='sentiment-group'>")
        parts.append(f"<h3>{sentiment_label} ({len(items)})</h3>")
        parts.append("<ul>")
        for item in items[:8]:
            title = escape_html(item["title"])
            source = escape_html(item["source"])
            url = escape_html(item["url"])
            parts.append(
                f"<li><a href='{url}' target='_blank' rel='noopener noreferrer'>{title}</a>"
                f" <span class='meta'>— {source}</span></li>"
            )
        parts.append("</ul>")
        parts.append("</div>")

    return "\n".join(parts)


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def generate_html(daily: dict[str, Any], run_summary: dict[str, Any]) -> str:
    labels_json = json.dumps(daily["labels"])
    avg_scores_json = json.dumps(daily["avg_scores"])
    article_counts_json = json.dumps(daily["article_counts"])
    label_counts_json = json.dumps(daily["label_counts"])

    latest_update = escape_html(str(run_summary.get("timestamp", "N/A")))
    fetched = escape_html(str(run_summary.get("articles_fetched", "N/A")))
    sql_inserted = escape_html(str(run_summary.get("sql_inserted", "N/A")))
    vector_inserted = escape_html(str(run_summary.get("vector_inserted", "N/A")))

    latest_day_html = build_latest_day_html(daily["latest_day"], daily["latest_day_groups"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Financial News Pipeline Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 0;
      padding: 0;
      background: #f7f7fb;
      color: #2f2f2f;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }}

    h1 {{
      margin-bottom: 8px;
      color: #4e4471;
    }}

    .subtitle {{
      color: #666;
      margin-bottom: 28px;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}

    .card {{
      background: white;
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}

    .card h2 {{
      font-size: 1rem;
      margin-top: 0;
      color: #4e4471;
    }}

    .metric {{
      font-size: 1.8rem;
      font-weight: bold;
      margin: 8px 0 0;
    }}

    .panel {{
      background: white;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      margin-bottom: 20px;
    }}

    .chart-wrap {{
      position: relative;
      width: 100%;
      min-height: 320px;
    }}

    .latest-section h3 {{
      margin-bottom: 8px;
      color: #4e4471;
    }}

    .latest-section ul {{
      margin-top: 0;
      padding-left: 20px;
    }}

    .latest-section li {{
      margin-bottom: 8px;
    }}

    .meta {{
      color: #666;
      font-size: 0.95rem;
    }}

    a {{
      color: #426eb4;
      text-decoration: none;
    }}

    a:hover {{
      text-decoration: underline;
    }}

    .footer-note {{
      margin-top: 20px;
      color: #666;
      font-size: 0.95rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Financial News Pipeline Dashboard</h1>
    <p class="subtitle">GitHub Pages report generated automatically from the pipeline artifacts.</p>

    <div class="cards">
      <div class="card">
        <h2>Last pipeline update</h2>
        <div class="metric" style="font-size:1.05rem;">{latest_update}</div>
      </div>
      <div class="card">
        <h2>Articles fetched</h2>
        <div class="metric">{fetched}</div>
      </div>
      <div class="card">
        <h2>New SQL rows</h2>
        <div class="metric">{sql_inserted}</div>
      </div>
      <div class="card">
        <h2>New vectors</h2>
        <div class="metric">{vector_inserted}</div>
      </div>
    </div>

    <div class="panel">
      <h2>Articles collected per day (last 7 days)</h2>
      <div class="chart-wrap">
        <canvas id="articlesChart"></canvas>
      </div>
    </div>

    <div class="panel">
      <h2>Average sentiment score per day</h2>
      <div class="chart-wrap">
        <canvas id="avgScoreChart"></canvas>
      </div>
    </div>

    <div class="panel">
      <h2>Sentiment label distribution per day</h2>
      <div class="chart-wrap">
        <canvas id="distributionChart"></canvas>
      </div>
    </div>

    <div class="panel latest-section">
      {latest_day_html}
    </div>

    <p class="footer-note">
      This dashboard is static HTML generated from SQLite and pipeline outputs during GitHub Actions.
    </p>
  </div>

  <script>
    const labels = {labels_json};
    const articleCounts = {article_counts_json};
    const avgScores = {avg_scores_json};
    const labelCounts = {label_counts_json};

    new Chart(document.getElementById('articlesChart'), {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: [{{
          label: 'Articles',
          data: articleCounts,
          borderWidth: 1
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          y: {{
            beginAtZero: true
          }}
        }}
      }}
    }});

    new Chart(document.getElementById('avgScoreChart'), {{
      type: 'line',
      data: {{
        labels: labels,
        datasets: [{{
          label: 'Average sentiment score',
          data: avgScores,
          tension: 0.25,
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          y: {{
            suggestedMin: -1,
            suggestedMax: 1
          }}
        }}
      }}
    }});

    const distributionDatasets = Object.keys(labelCounts).map((label) => {{
      return {{
        label: label,
        data: labelCounts[label],
        borderWidth: 1
      }};
    }});

    new Chart(document.getElementById('distributionChart'), {{
      type: 'bar',
      data: {{
        labels: labels,
        datasets: distributionDatasets
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{
            stacked: true
          }},
          y: {{
            stacked: true,
            beginAtZero: true
          }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    articles = load_articles()
    recent_articles = filter_last_n_days(articles, days=7)
    daily = build_daily_aggregates(recent_articles)
    run_summary = load_run_summary()

    html = generate_html(daily, run_summary)
    OUTPUT_HTML.write_text(html, encoding="utf-8")

    print(f"Dashboard written to: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()