from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


def _money(v: float) -> str:
    return f"${v:,.0f}"


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def build_dossier_pdf(
    *,
    city_name: str,
    summary: dict,
    selected: pd.DataFrame,
    provenance: dict,
    evidence_ledger: pd.DataFrame | None = None,
    robustness_summary: dict | None = None,
    verification_projects: pd.DataFrame | None = None,
) -> bytes:
    """Build a compact city-ready ThermalOS decision dossier as PDF bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title=f"ThermalOS Decision Dossier - {city_name}",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TOS_Title", parent=styles["Title"], fontSize=22, leading=25, alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="TOS_Sub", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=14))
    styles.add(ParagraphStyle(name="TOS_H2", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#0f172a"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="TOS_Small", parent=styles["BodyText"], fontSize=7.5, leading=10, textColor=colors.HexColor("#475569")))

    decision_state = str(summary.get("decision_state", "action_plan"))
    no_action_triggered = decision_state == "no_action_triggered"
    no_action_reason = str(summary.get("no_action_reason", "")).strip()

    story = [
        Paragraph("ThermalOS Decision Dossier", styles["TOS_Title"]),
        Paragraph(f"{city_name} - urban heat resilience planning scenario", styles["TOS_Sub"]),
    ]
    kpis = [
        ["Capital budget", _money(float(summary.get("budget_usd", 0)))],
        ["Budget deployed", _money(float(summary.get("spent_usd", 0)))],
        ["Funded projects", f"{int(summary.get('projects', len(selected)))}"],
        ["Direct modeled relief", f"{float(summary.get('planning_person_hours_avoided_first_order', 0)):,.0f} person-hours"],
        ["System-level modeled relief", f"{float(summary.get('modeled_person_hours_avoided_with_spillover', 0)):,.0f} person-hours"],
        ["Modeled reduction", _pct(float(summary.get("modeled_reduction_fraction", 0)))],
        [
            "Equity-aligned spend",
            "N/A - no action trigger"
            if no_action_triggered
            else _pct(float(summary.get("vulnerable_spend_fraction", 0))),
        ],
    ]
    t = Table(kpis, colWidths=[2.2 * inch, 4.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [t, Spacer(1, 8)]

    story.append(Paragraph("Recommended capital portfolio", styles["TOS_H2"]))
    if selected.empty and no_action_triggered:
        reason = no_action_reason or "The configured event threshold did not produce positive heat burden."
        story.append(
            Paragraph(
                f"No capital intervention was triggered for this event. {reason} "
                "The configured capital budget is therefore not deployed.",
                styles["BodyText"],
            )
        )
    elif selected.empty:
        story.append(Paragraph("No feasible projects under the supplied planning constraints.", styles["BodyText"]))
    else:
        top = selected.copy().sort_values("benefit_expected_person_hours", ascending=False).head(15)
        data = [["Project", "Area", "Cost", "Expected relief", "Vulnerability"]]
        for _, r in top.iterrows():
            data.append([
                str(r.get("label", ""))[:32],
                str(r.get("area", "")).replace("_", " ")[:22],
                _money(float(r.get("cost_usd", 0))),
                f"{float(r.get('benefit_expected_person_hours', 0)):,.0f}",
                f"{float(r.get('vulnerability', 0)):.2f}",
            ])
        pt = Table(data, repeatRows=1, colWidths=[2.2 * inch, 1.35 * inch, 0.9 * inch, 1.05 * inch, 0.8 * inch])
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("FONTSIZE", (0, 0), (-1, -1), 7.2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(pt)
        if len(selected) > 15:
            story.append(Paragraph(f"Top 15 of {len(selected)} funded projects shown.", styles["TOS_Small"]))

    story.append(Paragraph("Decision robustness", styles["TOS_H2"]))
    if no_action_triggered:
        text = (
            "Not applicable for this event: no capital portfolio was generated because "
            "the configured heat-action trigger was not met."
        )
    elif robustness_summary:
        text = (
            f"Portfolio stability: {100 * float(robustness_summary.get('portfolio_stability', 0)):.0f}%. "
            f"Median selection-set overlap: {100 * float(robustness_summary.get('median_jaccard', 0)):.0f}%. "
            f"Direct benefit central stress-test range: {float(robustness_summary.get('direct_benefit_p10', 0)):,.0f}-"
            f"{float(robustness_summary.get('direct_benefit_p90', 0)):,.0f} person-hours."
        )
    else:
        text = "Robustness analysis was not included in this export. ThermalOS can re-optimize plausible effect/cost worlds to quantify project selection stability."
    story.append(Paragraph(text, styles["BodyText"]))

    story.append(Paragraph("ThermalVerify measurement plan", styles["TOS_H2"]))
    if no_action_triggered:
        story.append(
            Paragraph(
                "Not applicable for this event because no capital intervention was triggered. "
                "No post-deployment intervention-effect claim is created.",
                styles["BodyText"],
            )
        )
    elif verification_projects is not None and len(verification_projects):
        story.append(Paragraph(
            f"Baseline records are prepared for {len(verification_projects)} funded projects. Recommended verification windows are 30, 90, and 365 days after deployment using matched control cells and weather-normalized FortyGuard observations.",
            styles["BodyText"],
        ))
    else:
        story.append(Paragraph("No verification registry was attached to this dossier.", styles["BodyText"]))

    story.append(PageBreak())
    story.append(Paragraph("Evidence and provenance", styles["TOS_H2"]))
    if evidence_ledger is not None and len(evidence_ledger):
        cell_style = ParagraphStyle(
            "TOS_TableCell",
            parent=styles["BodyText"],
            fontSize=6.1,
            leading=7.4,
            spaceAfter=0,
            spaceBefore=0,
        )
        head_style = ParagraphStyle(
            "TOS_TableHead",
            parent=cell_style,
            textColor=colors.white,
            fontName="Helvetica-Bold",
        )

        def cell(value: object, limit: int | None = None) -> Paragraph:
            text = str(value if value is not None else "")
            if limit is not None and len(text) > limit:
                text = text[: max(0, limit - 3)] + "..."
            # ReportLab Paragraph interprets a small HTML subset; escape the
            # characters most likely to occur in source/claim text.
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return Paragraph(text, cell_style)

        data = [[
            Paragraph("Layer", head_style),
            Paragraph("Class", head_style),
            Paragraph("Source", head_style),
            Paragraph("Coverage", head_style),
            Paragraph("Boundary", head_style),
        ]]
        for _, r in evidence_ledger.iterrows():
            data.append([
                cell(r.get("layer", ""), 34),
                cell(r.get("evidence_class", ""), 32),
                cell(r.get("source", ""), 70),
                cell(f"{float(r.get('coverage_pct', 0)):.0f}%"),
                cell(r.get("uncertainty_or_limit", ""), 180),
            ])
        et = Table(
            data,
            repeatRows=1,
            colWidths=[1.05 * inch, 1.0 * inch, 1.65 * inch, 0.55 * inch, 2.55 * inch],
        )
        et.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(et)

    story += [
        Spacer(1, 10),
        Paragraph("Scientific claim boundary", styles["TOS_H2"]),
        Paragraph(
            "ThermalOS is a decision-support prototype. Intervention effects are modeled planning scenarios bounded by configurable evidence priors. Population and access variables are planning proxies. The portfolio is not a causal impact evaluation, medical recommendation, engineering design, or procurement recommendation. ThermalVerify is designed to collect post-deployment evidence before local intervention priors are updated.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph(f"Data mode: {'synthetic demo' if provenance.get('synthetic_demo', False) else 'real planning data'}; source: {provenance.get('source_file', 'not recorded')}", styles["TOS_Small"]),
    ]

    doc.build(story)
    return buffer.getvalue()
