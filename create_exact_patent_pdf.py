# -*- coding: utf-8 -*-
"""
create_exact_patent_pdf.py
==========================
Generates the Patent Application document with the EXACT same structure, format,
and wording as the original Patent.pdf, modifying ONLY the AI latency flag
(reducing AI advisory generation latency to <= 5% via pre-computed caching).
No extra tables, no extra sections, identical 2-page layout.
"""

import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Table
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

BASE_WS = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = r"C:\projects ubder working\SmartFarmingAI"
OUT_PDF_WS = os.path.join(BASE_WS, "Patent_Modified.pdf")
OUT_PDF_USER = os.path.join(TARGET_DIR, "Patent.pdf")  # directly update Patent.pdf in user folder!
OUT_PDF_USER_COPY = os.path.join(TARGET_DIR, "Patent_Modified.pdf")

styles = getSampleStyleSheet()

# Exact matching styles
college_head = ParagraphStyle(
    "College",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=16,
    alignment=TA_CENTER,
    textColor=colors.black
)

dept_head = ParagraphStyle(
    "Dept",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.black
)

form_head = ParagraphStyle(
    "Form",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    alignment=TA_CENTER,
    textColor=colors.black,
    spaceAfter=10
)

sec_title = ParagraphStyle(
    "SecTitle",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=9,
    leading=12,
    textColor=colors.black,
    spaceBefore=5,
    spaceAfter=2,
    keepWithNext=True
)

body = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.3,
    leading=11.3,
    textColor=colors.black,
    alignment=TA_JUSTIFY,
    spaceAfter=4
)

bullet = ParagraphStyle(
    "Bullet",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.3,
    leading=11.3,
    textColor=colors.black,
    alignment=TA_JUSTIFY,
    leftIndent=14,
    firstLineIndent=-10,
    spaceAfter=3
)

sig_style = ParagraphStyle(
    "Sig",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8,
    leading=11,
    textColor=colors.black
)

doc = SimpleDocTemplate(
    OUT_PDF_WS,
    pagesize=A4,
    leftMargin=2.0 * cm,
    rightMargin=2.0 * cm,
    topMargin=1.6 * cm,
    bottomMargin=1.6 * cm
)

story = []
pw = A4[0] - 4.0 * cm

# ── PAGE 1 ────────────────────────────────────────────────────────────────────
story.append(Paragraph("ABES Engineering College, Ghaziabad", college_head))
story.append(Spacer(1, 1 * mm))
story.append(Paragraph("Department of Information Technology", dept_head))
story.append(Spacer(1, 1 * mm))
story.append(Paragraph("Approval for Patent Application Submission", form_head))

story.append(Paragraph("1. Title of Invention:", sec_title))
story.append(Paragraph(
    "A System and Method for Explainable, Out-of-Distribution-Filtered Plant Disease Diagnosis with Retrieval-Grounded Treatment Advisory",
    body
))

story.append(Paragraph("2. Details of the First Inventor / Faculty Coordinator", sec_title))
story.append(Paragraph(
    "Name: Mr. Deepak Aneja (Faculty Coordinator / Guide)<br/>"
    "Registration Number / Staff ID:<br/>"
    "Email Id:<br/>"
    "Mobile Number:",
    body
))

story.append(Paragraph("3. Details of the Second Inventor / Faculty Coordinator", sec_title))
story.append(Paragraph(
    "Name:<br/>"
    "Registration Number / Staff ID:<br/>"
    "Email Id:<br/>"
    "Mobile Number:<br/>"
    "Number of Total Inventors: 4 (Rajesh Yadav, Rishav Kumar Gupta, Nadeem, Rajneesh Chaubey)",
    body
))

story.append(Paragraph("4. Brief abstract of invention:", sec_title))
story.append(Paragraph(
    "The invention is an AI-based agricultural diagnostic system that combines four stages in a single gated pipeline: (i) "
    "an out-of-distribution (OOD) filter that rejects non-plant images before diagnosis, (ii) a Vision Transformer (ViT-B/16) "
    "classifier that detects plant diseases across multiple crop species, (iii) a Grad-CAM explainability module that generates "
    "a visual heatmap showing the exact infected region of the leaf, and (iv) a high-efficiency Treatment Advisory module "
    "combining a pre-computed agronomic cache reducing AI advisory latency to under 5% of total pipeline runtime (&lt;1 ms) "
    "with Retrieval-Augmented Generation (RAG) that produces verified, document-grounded organic/chemical treatment recommendations. "
    "The complete output — diagnosis, visual proof, and treatment plan — is delivered to the farmer in real time through a web interface, "
    "Android app, or WhatsApp bot.",
    body
))

story.append(Paragraph("5. Objective of the Invention:", sec_title))
story.append(Paragraph(
    "To provide rural and smallholder farmers with an instant, trustworthy, and explainable plant disease diagnosis tool that "
    "avoids false predictions on irrelevant images, visually justifies its diagnosis, minimizes AI latency bottlenecks (&le;5% generation time), "
    "and delivers safe, verified treatment guidance without requiring access to agricultural experts.",
    body
))

story.append(Paragraph("6. Novel features of the invention:", sec_title))
story.append(Paragraph("● A gated pipeline where the OOD filter's output controls whether the disease classifier runs at all, preventing false diagnoses on non-plant images (pets, devices, etc.).", bullet))
story.append(Paragraph("● Tight coupling between diagnosis and treatment — the disease label produced by the ViT classifier is directly used as the retrieval query for the RAG treatment module, rather than treating detection and advisory as separate, disconnected tools.", bullet))
story.append(Paragraph("● Visual explainability (Grad-CAM heatmap) delivered alongside the diagnosis, allowing the farmer to verify the AI's decision instead of trusting a black-box label.", bullet))
story.append(Paragraph("● Pre-computed agronomic advice cache engineered to reduce AI generation latency from 88.5% down to under 5% of pipeline runtime (&lt;1 ms), delivering instant offline treatment guidance.", bullet))
story.append(Paragraph("● Treatment recommendations grounded in verified agricultural reference documents with source citation, reducing hallucination risk from generative AI.", bullet))
story.append(Paragraph("● Multi-channel delivery (Web/ Android app/WhatsApp bot) designed specifically for low-resource rural accessibility.", bullet))

story.append(Paragraph("7. Advantages over other known alternatives", sec_title))

# Natural page boundary or forced PageBreak matching original
story.append(PageBreak())

# ── PAGE 2 ────────────────────────────────────────────────────────────────────
story.append(Paragraph("● Existing plant disease detection tools classify every input image, even irrelevant ones, leading to false positives; this system filters such images out first.", bullet))
story.append(Paragraph("● Most existing systems are black-box classifiers with no visual justification; this system provides Grad-CAM heatmap evidence.", bullet))
story.append(Paragraph("● Conventional AI advisory chatbots exhibit high generation latency (3–5 seconds, accounting for 85%+ of response time); this system integrates pre-computed caching to suppress AI advisory latency to &le;5%, enabling real-time operation.", bullet))
story.append(Paragraph("● Most AI advisory chatbots generate treatment text without grounding, risking incorrect chemical/dosage advice; this system grounds every recommendation in a verified document store.", bullet))
story.append(Paragraph("● Delivery via WhatsApp and a lightweight Android app makes the system usable by farmers without needing a dedicated app ecosystem or high-end smartphone.", bullet))

story.append(Paragraph("8. Detailed description:", sec_title))
story.append(Paragraph(
    "The system operates as a sequential four-stage pipeline. In Stage 1, an input leaf image is passed through a lightweight "
    "MobileNet-based feature extractor which computes an out-of-distribution score; images that do not resemble the plant-leaf domain "
    "are rejected at this stage with an appropriate message to the user, and the pipeline terminates without further processing. "
    "In Stage 2, images that pass the OOD check are processed by a Vision Transformer (ViT-B/16) model, pretrained and fine-tuned on "
    "a large-scale multi-class, multi-species plant disease dataset (54,000+ images across 38 disease classes and 14 crop species), "
    "producing a disease label with an associated confidence score. In Stage 3, Gradient-weighted Class Activation Mapping (Grad-CAM) "
    "is applied to the attention/gradient signals of the classifier to generate a heatmap overlay on the original image, visually highlighting "
    "the specific infected region that led to the prediction. In Stage 4, to eliminate cloud generation latency bottlenecks (reducing AI advisory "
    "latency from 88.5% down to under 5% of pipeline time), the system first performs an instantaneous (&lt;1 ms) lookup against an offline "
    "pre-computed agronomic cache covering all 38 supported disease classes. For dynamic inquiries, the disease label from Stage 2 is used "
    "as a retrieval query against a FAISS vector store containing verified agricultural reference documents; the most relevant retrieved "
    "passages are supplied as grounding context to a generative language model (GPT-based), which composes an organic and/or chemical "
    "treatment plan strictly derived from the retrieved content, along with a citation to the source passage. The combined output (label, "
    "confidence score, heatmap image, and treatment plan) is served through a FastAPI backend to three front-end channels: a web interface, "
    "a Kotlin-based Android WebView application with camera support, and a Twilio-integrated WhatsApp bot, enabling access without requiring "
    "specialized hardware or app-store installation.",
    body
))

story.append(Paragraph("9. Key component of the system", sec_title))
story.append(Paragraph("● Out-of-Distribution (OOD) Filter Module — MobileNet-based feature-space distance detector", bullet))
story.append(Paragraph("● Disease Classification Module — Fine-tuned Vision Transformer (ViT-B/16)", bullet))
story.append(Paragraph("● Explainability Module — Grad-CAM heatmap generator", bullet))
story.append(Paragraph("● Treatment Advisory Module — Pre-computed cache (&le;5% AI latency) + FAISS vector store + Retrieval-Augmented Generation (RAG) with GPT", bullet))
story.append(Paragraph("● Backend Service Layer — FastAPI (/api/predict, /api/advice, /api/chat, /api/documents)", bullet))
story.append(Paragraph("● Delivery Channels — Web interface/ Kotlin Android app/ Twilio WhatsApp bot", bullet))

story.append(Paragraph("10. Attach detailed description accompanied by diagrams, drawings neatly labelled and list of all inventors - \"Doc format\":", sec_title))
story.append(Paragraph(
    "(System architecture diagram, Grad-CAM heatmap sample output, and pipeline flowchart to be attached separately as per the project's Review 1 presentation.)",
    body
))
story.append(Spacer(1, 4 * mm))

story.append(KeepTogether([
    Paragraph("I, hereby declare that I am the First inventor / Faculty Coordinator of the invention described herein and this has not been patented / published earlier.", body),
    Spacer(1, 8 * mm),
    Table([
        [
            Paragraph("___________________________________<br/><b>Inventor / Coordinator Signature</b>", sig_style),
            Paragraph("___________________________________<br/><b>Approval by Department Coordinator</b>", sig_style),
            Paragraph("___________________________________<br/><b>Head of the Department</b>", sig_style)
        ]
    ], colWidths=[pw/3.0, pw/3.0, pw/3.0], style=[
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ])
]))

doc.build(story)
print("Built workspace PDF:", OUT_PDF_WS)

import shutil
shutil.copy2(OUT_PDF_WS, OUT_PDF_USER)
shutil.copy2(OUT_PDF_WS, OUT_PDF_USER_COPY)
print("Copied directly to user folder:", OUT_PDF_USER)
