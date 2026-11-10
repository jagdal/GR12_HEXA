"""Génère benchmark_llm.pdf (racine du projet) à partir de benchmark/results/*.json
et du texte d'analyse de benchmark/analysis.json.

Usage : python benchmark/make_pdf.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmark" / "results"
FIG = ROOT / "benchmark" / "figures"
FIG.mkdir(exist_ok=True)
OUT = ROOT / "benchmark_llm.pdf"

# Police Unicode (accents, ✓) fournie avec matplotlib
_dejavu = Path(font_manager.findfont("DejaVu Sans"))
pdfmetrics.registerFont(TTFont("DejaVu", str(_dejavu)))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_dejavu.with_name("DejaVuSans-Bold.ttf"))))
pdfmetrics.registerFont(TTFont("DejaVu-Italic", str(_dejavu.with_name("DejaVuSans-Oblique.ttf"))))
pdfmetrics.registerFont(TTFont("DejaVu-BoldItalic", str(_dejavu.with_name("DejaVuSans-BoldOblique.ttf"))))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Italic",
                              boldItalic="DejaVu-BoldItalic")

ACCENT = colors.HexColor("#C8102E")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#E5E7EB")
SOFT = colors.HexColor("#FDECEE")
BEST = colors.HexColor("#E8F5E9")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="DejaVu-Bold", fontSize=22, leading=27, textColor=INK, alignment=0, spaceAfter=4),
    "kicker": ParagraphStyle("k", fontName="DejaVu-Bold", fontSize=9, textColor=ACCENT, spaceAfter=6),
    "sub": ParagraphStyle("s", fontName="DejaVu", fontSize=10.5, leading=15, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=14, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15, textColor=INK, spaceBefore=10, spaceAfter=5),
    "p": ParagraphStyle("p", fontName="DejaVu", fontSize=9.5, leading=14, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("sm", fontName="DejaVu", fontSize=8, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("c", fontName="DejaVu", fontSize=8, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="DejaVu-Bold", fontSize=8, leading=10.5, textColor=INK),
    "box": ParagraphStyle("b", fontName="DejaVu", fontSize=10, leading=15, textColor=INK),
    "center": ParagraphStyle("ce", fontName="DejaVu", fontSize=8, leading=10, alignment=TA_CENTER),
}


def load(prefix: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RES.glob(f"{prefix}_*.json"))]


def pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f} %"


def fmt(x, unit="") -> str:
    return "—" if x is None else f"{x:g}{unit}".replace(".", ",")


def table(data, widths, best_row=None, header_bg=SOFT):
    rows = [[Paragraph(str(c), S["cellb"] if r == 0 else S["cell"]) for c in row] for r, row in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if best_row is not None:
        style.append(("BACKGROUND", (0, best_row), (-1, best_row), BEST))
    t.setStyle(TableStyle(style))
    return t


def box(html: str):
    t = Table([[Paragraph(html, S["box"])]], colWidths=[17 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT), ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


# ----------------------------------------------------------------------------- graphiques

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
RED, GREY, DARK = "#C8102E", "#9CA3AF", "#374151"


def chart_llm(llms):
    llms = sorted(llms, key=lambda r: r["overall"])
    names = [r["model"] for r in llms]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(llms) + 0.9))
    y = range(len(llms))
    ax.barh([i + 0.2 for i in y], [r["accuracy"] * 100 for r in llms], height=0.38, color=RED, label="Justesse (32 questions)")
    ax.barh([i - 0.2 for i in y], [r["refusal_rate"] * 100 for r in llms], height=0.38, color=GREY, label="Refus d'inventer (4 pièges)")
    ax.set_yticks(list(y), names)
    ax.set_xlim(0, 105)
    ax.set_xlabel("%")
    for i, r in enumerate(llms):
        ax.text(r["accuracy"] * 100 + 1, i + 0.2, f"{r['accuracy'] * 100:.0f}", va="center", fontsize=8, color=DARK)
        ax.text(r["refusal_rate"] * 100 + 1, i - 0.2, f"{r['refusal_rate'] * 100:.0f}", va="center", fontsize=8, color=DARK)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG / "llm_quality.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def chart_tradeoff(llms):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for r in llms:
        ax.scatter(r["median_seconds"], r["overall"] * 100, s=40 + 60 * (r["ram_gb"] or r["size_gb"]),
                   color=RED, alpha=0.75, edgecolor="white")
        ax.annotate(r["model"], (r["median_seconds"], r["overall"] * 100), textcoords="offset points",
                    xytext=(7, 4), fontsize=8, color=DARK)
    ax.set_xlabel("Temps de réponse médian (secondes, CPU)")
    ax.set_ylabel("Score global (%)")
    ax.grid(alpha=0.25)
    ax.set_title("Qualité vs vitesse (taille du point = RAM utilisée)", fontsize=9, color=DARK, loc="left")
    fig.tight_layout()
    p = FIG / "llm_tradeoff.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def chart_embed(embs):
    embs = sorted(embs, key=lambda r: r["mrr"])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(embs) + 0.9))
    y = range(len(embs))
    ax.barh([i + 0.2 for i in y], [r["hit_at_1"] * 100 for r in embs], height=0.38, color=GREY, label="Bon chapitre en 1re position")
    ax.barh([i - 0.2 for i in y], [r["hit_at_k"] * 100 for r in embs], height=0.38, color=RED, label="Bon chapitre dans les 4 premiers")
    ax.set_yticks(list(y), [r["model"] for r in embs])
    ax.set_xlim(0, 105)
    ax.set_xlabel("%")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    p = FIG / "embed_quality.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------- document

def build():
    llms = load("llm")
    embs = load("embed")
    A = json.loads((ROOT / "benchmark" / "analysis.json").read_text(encoding="utf-8"))
    llms_sorted = sorted(llms, key=lambda r: (-r["overall"], r["median_seconds"]))
    embs_sorted = sorted(embs, key=lambda r: (-r["mrr"], r["index_seconds"]))

    story = []
    story += [
        Paragraph("HETIC · GIVAUDAN AI CHALLENGE · GR12_HEXA", S["kicker"]),
        Paragraph("Benchmark des modèles Ollama", S["title"]),
        Paragraph(f"Assistant RAG sur les données publiques de Givaudan — tests réalisés le "
                  f"{date.today().strftime('%d/%m/%Y')} sur le poste de développement de l'équipe.", S["sub"]),
        box(A["recommendation"]),
        Spacer(1, 10),
        Paragraph("1. Objectif et mission", S["h1"]),
    ]
    for para in A["context"]:
        story.append(Paragraph(para, S["p"]))

    story.append(Paragraph("2. Protocole de test", S["h1"]))
    for para in A["method"]:
        story.append(Paragraph(para, S["p"]))

    # Embeddings
    story.append(Paragraph("3. Modèles d'embeddings (recherche des passages)", S["h1"]))
    story.append(Paragraph(A["embed_intro"], S["p"]))
    data = [["Modèle", "Taille", "Top-1", "Top-4", "MRR", "Indexation (35 morceaux)", "Requête"]]
    for r in embs_sorted:
        data.append([r["model"], f"{fmt(r['size_gb'])} Go", pct(r["hit_at_1"]), pct(r["hit_at_k"]),
                     fmt(r["mrr"]), f"{fmt(r['index_seconds'])} s", f"{fmt(r['query_ms'])} ms"])
    story.append(table(data, [4.2 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm, 3.6 * cm, 2.2 * cm], best_row=1))
    story.append(Spacer(1, 6))
    story.append(Image(str(chart_embed(embs)), width=16 * cm, height=16 * cm * (0.42 * len(embs) + 0.9) / 7.2))
    story.append(Paragraph(A["embed_analysis"], S["p"]))

    # LLM
    story.append(PageBreak())
    story.append(Paragraph("4. Modèles de génération (LLM)", S["h1"]))
    story.append(Paragraph(A["llm_intro"], S["p"]))
    data = [["Modèle", "Disque", "RAM", "Justesse (/32)", "Refus pièges (/4)", "Score global", "Temps médian", "Vitesse"]]
    for r in llms_sorted:
        data.append([r["model"], f"{fmt(r['size_gb'])} Go", f"{fmt(r['ram_gb'])} Go",
                     f"{pct(r['accuracy'])} ({round(r['accuracy'] * r['n_answerable'])})",
                     f"{pct(r['refusal_rate'])} ({round(r['refusal_rate'] * r['n_unanswerable'])})",
                     pct(r["overall"]), f"{fmt(r['median_seconds'])} s", f"{fmt(r['gen_tps'])} tok/s"])
    story.append(table(data, [3.0 * cm, 1.6 * cm, 1.6 * cm, 2.4 * cm, 2.4 * cm, 1.9 * cm, 1.9 * cm, 2.0 * cm], best_row=1))
    story.append(Paragraph("Ligne verte : meilleur score global (à égalité, le plus rapide). Vitesse = tokens générés par seconde "
                           "(médiane). Temps médian = question posée → réponse complète, modèle déjà chargé.", S["small"]))
    story.append(Spacer(1, 8))
    story.append(Image(str(chart_llm(llms)), width=16 * cm, height=16 * cm * (0.42 * len(llms) + 0.9) / 7.2))
    story.append(Image(str(chart_tradeoff(llms)), width=16 * cm, height=8 * cm))

    for i, (title, text) in enumerate(A["llm_analysis"]):
        head = [Paragraph("5. Analyse", S["h1"])] if i == 0 else []
        story.append(KeepTogether(head + [Paragraph(title, S["h2"]), Paragraph(text, S["p"])]))

    story.append(Paragraph("6. Recommandation pour la mission", S["h1"]))
    for para in A["final"]:
        story.append(Paragraph(para, S["p"]))

    story.append(Paragraph("7. Limites du benchmark", S["h1"]))
    for para in A["limits"]:
        story.append(Paragraph("• " + para, S["p"]))

    # Annexe : matrice question × modèle
    story.append(PageBreak())
    story.append(Paragraph("Annexe A — Résultat par question", S["h1"]))
    story.append(Paragraph("✓ = réponse correcte (chiffre ou nom attendu présent ; pour les 4 pièges, refus explicite). "
                           "✗ = réponse fausse, incomplète ou inventée.", S["small"]))
    story.append(Spacer(1, 4))
    names = [r["model"] for r in llms_sorted]
    head = ["#", "Question"] + [n.replace(":", ":<br/>") for n in names]
    data = [[Paragraph(h, S["cellb"]) for h in head]]
    for qi, row in enumerate(llms_sorted[0]["rows"]):
        line = [Paragraph(str(qi + 1), S["cell"]), Paragraph(escape(row["q"]) + ("" if row["answerable"] else " <i>(piège)</i>"), S["cell"])]
        for r in llms_sorted:
            ok = r["rows"][qi]["ok"]
            line.append(Paragraph(f'<font color="{"#15803D" if ok else "#C8102E"}">{"✓" if ok else "✗"}</font>', S["center"]))
        data.append(line)
    qw = 17.4 * cm - 0.9 * cm - len(names) * 1.8 * cm
    t = Table(data, colWidths=[0.9 * cm, qw] + [1.8 * cm] * len(names), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SOFT), ("GRID", (0, 0), (-1, -1), 0.3, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)

    story.append(PageBreak())
    story.append(Paragraph("Annexe B — Exemples de réponses", S["h1"]))
    for title, text in A["examples"]:
        story.append(KeepTogether([Paragraph(title, S["h2"]), Paragraph(text, S["p"])]))
    story.append(Paragraph("Annexe C — Reproduire le benchmark", S["h1"]))
    for para in A["reproduce"]:
        story.append(Paragraph(para, S["p"]))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(1.8 * cm, 1.1 * cm, "GR12_HEXA · Givaudan AI Challenge · Benchmark des modèles Ollama")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                            title="Benchmark des modèles Ollama — GR12_HEXA", author="GR12_HEXA")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"PDF écrit : {OUT}")


if __name__ == "__main__":
    build()
