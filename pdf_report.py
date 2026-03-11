# pdf_report.py
# Generates a professional forensic PDF report using ReportLab.

import datetime
from dataclasses import asdict
from typing import List, Optional
from collections import defaultdict

from models import ImageItem, VideoFrameResult, SingleVideoFrame


def generate_pdf_report(out_path: str,
                        mode: str,
                        image_items: Optional[List[ImageItem]] = None,
                        video_results: Optional[List[VideoFrameResult]] = None,
                        single_video=None,
                        ai_summary: str = "",
                        groups_only: bool = False) -> str:
    """
    Build and save a forensic PDF report.
    Returns out_path on success.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)

    # -- Colour palette -------------------------------------------------------
    NAVY   = colors.HexColor("#0D2B4E")
    BLUE   = colors.HexColor("#1A4F8A")
    LTBLUE = colors.HexColor("#EEF3FA")
    TEAL   = colors.HexColor("#117A8B")
    GREEN  = colors.HexColor("#1D6A3A")
    LGRAY  = colors.HexColor("#F4F4F4")
    MGRAY  = colors.HexColor("#CCCCCC")
    WHITE  = colors.white
    BLACK  = colors.black

    # Paragraph styles
    def s(sz, **kw):
        return ParagraphStyle("_", fontSize=sz, **kw)

    title_s = s(26, textColor=WHITE, alignment=1, fontName="Helvetica-Bold", spaceAfter=6)
    sub_s   = s(13, textColor=colors.HexColor("#AACCEE"), alignment=1,
                fontName="Helvetica", spaceAfter=4)
    meta_s  = s(9,  textColor=colors.HexColor("#AACCEE"), alignment=1,
                fontName="Helvetica", spaceAfter=0)
    h2_s    = s(13, textColor=NAVY, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=4)
    body_s  = s(9,  textColor=BLACK, fontName="Helvetica",
                spaceAfter=6, leading=14, alignment=4)
    label_s = s(9,  textColor=NAVY,  fontName="Helvetica-Bold")
    value_s = s(9,  textColor=BLACK, fontName="Helvetica")
    small_s = s(7,  textColor=colors.HexColor("#666666"),
                fontName="Helvetica", alignment=1)

    # Document setup
    doc   = SimpleDocTemplate(out_path, pagesize=letter,
                               leftMargin=0.65*inch, rightMargin=0.65*inch,
                               topMargin=0.65*inch,  bottomMargin=0.65*inch)
    W     = letter[0] - 1.3*inch
    story = []
    ts    = datetime.datetime.now().strftime("%B %d, %Y  %H:%M:%S")

    # Cover banner 
    scope_note = "Grouped Images Only" if groups_only else "All Images"
    cover = Table(
        [[Paragraph("MEDIA ANALYZER", title_s)],
         [Paragraph("Forensic Analysis Report", sub_s)],
         [Paragraph(f"Generated: {ts}  |  Mode: {mode}  |  Scope: {scope_note}  |  CONFIDENTIAL", meta_s)]],
        colWidths=[W]
    )
    cover.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
    ]))
    story.append(cover)
    story.append(Spacer(1, 0.2*inch))

    # Summary statistics
    story.append(HRFlowable(width=W, thickness=2, color=BLUE, spaceAfter=6))
    story.append(Paragraph("Summary Statistics", h2_s))
    stat_rows = []

    if mode in ("Image", "Single Image") and image_items:
        rows  = [asdict(it) for it in image_items]
        total = len(rows)
        bc: dict = defaultdict(int)
        for r in rows:
            bid = r.get("bucket_phash", -1)
            if bid != -1:
                bc[bid] += 1
        groups    = len(bc)
        in_groups = sum(bc.values())
        unique    = total - in_groups
        largest   = max(bc.values(), default=0)
        bvals     = [r["avg_brightness"] for r in rows if "avg_brightness" in r]
        avg_b     = round(sum(bvals) / len(bvals), 1) if bvals else 0
        stat_rows = [
            ["Total images analyzed",   str(total)],
            ["Unique (no match found)", str(unique)],
            ["pHash similarity groups", f"{groups} group(s), {in_groups} image(s) grouped"],
            ["Largest group",           str(largest) if largest > 0 else "N/A"],
            ["Average brightness",      str(avg_b)],
        ]

    elif mode in ("Video", "Single Video"):
        if mode == "Single Video" and single_video:
            story.append(Paragraph(
                f"<b>File:</b> {single_video.filename}  |  {single_video.duration}s  |  "
                f"{single_video.fps:.1f} FPS  |  "
                f"{single_video.width}x{single_video.height}  |  "
                f"Codec: {single_video.codec}", value_s))
            story.append(Spacer(1, 0.08*inch))
            rows     = [asdict(f) for f in single_video.frames]
            motions  = [r["motion_score"]   for r in rows]
            bvals    = [r["avg_brightness"] for r in rows]
            stat_rows = [
                ["Frames sampled",     str(len(rows))],
                ["Average motion score",
                 str(round(sum(motions) / len(motions), 2)) if motions else "N/A"],
                ["Max motion score",
                 str(round(max(motions), 2)) if motions else "N/A"],
                ["Average brightness",
                 str(round(sum(bvals) / len(bvals), 1)) if bvals else "N/A"],
            ]
        elif video_results:
            rows  = [asdict(r) for r in video_results]
            pairs = list({(r["video_pair_1"], r["video_pair_2"]) for r in rows})
            sims  = [r["similarity_ssim"] for r in rows]
            mses  = [r["similarity_mse"]  for r in rows]
            psnrs = [r["similarity_psnr"] for r in rows]
            mots  = [r["motion_score"]    for r in rows]
            stat_rows = [
                ["Video pairs compared",    str(len(pairs))],
                ["Total frame comparisons", str(len(rows))],
                ["Avg / Max / Min SSIM",
                 f"{round(sum(sims)/len(sims),1)}% / "
                 f"{round(max(sims),1)}% / {round(min(sims),1)}%"],
                ["Avg MSE",          str(round(sum(mses)  / len(mses),   1))],
                ["Avg PSNR",         f"{round(sum(psnrs) / len(psnrs), 1)} dB"],
                ["Avg motion score", str(round(sum(mots) / len(mots), 2))],
            ]

    if stat_rows:
        tbl = Table(
            [[Paragraph(r[0], label_s), Paragraph(r[1], value_s)] for r in stat_rows],
            colWidths=[2.0*inch, W - 2.0*inch]
        )
        tbl.setStyle(TableStyle([
            ("GRID",          (0, 0), (-1, -1), 0.5, MGRAY),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [LTBLUE, WHITE]),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.12*inch))

    # AI Narrative 
    if ai_summary and ai_summary.strip():
        story.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))
        story.append(Paragraph("AI Forensic Narrative (Llama 3 / Local)", h2_s))

        narrative_s = ParagraphStyle(
            "_narrative",
            fontSize=9,
            textColor=BLACK,
            fontName="Helvetica",
            spaceAfter=6,
            spaceBefore=2,
            leading=14,
            alignment=4,
            leftIndent=12,
            rightIndent=12,
            backColor=colors.HexColor("#F0F4FF"),
        )

        story.append(HRFlowable(width=W, thickness=1, color=BLUE, spaceAfter=4))

        for block in ai_summary.strip().split("\n\n"):
            block = block.strip()
            if not block:
                continue
            story.append(Paragraph(block.replace("\n", "<br/>"), narrative_s))

        story.append(HRFlowable(width=W, thickness=1, color=BLUE, spaceBefore=4, spaceAfter=0))
        story.append(Spacer(1, 0.12*inch))

    # Detailed findings table 
    story.append(HRFlowable(width=W, thickness=2, color=GREEN, spaceAfter=6))
    story.append(Paragraph("Detailed Findings", h2_s))

    if mode in ("Image", "Single Image") and image_items:
        hdrs  = ["ID", "Filename", "W x H", "Brightness", "Contrast",
                 "pHash Group", "Color Dist.", "Date Taken", "Camera"]
        col_w = [0.28*inch, 1.5*inch, 0.65*inch, 0.65*inch, 0.6*inch,
                 0.7*inch, 0.65*inch, 1.0*inch, W - 6.03*inch]
        rows_data = [hdrs] + [
            [str(it.id),
             it.filename[:24],
             f"{it.width}x{it.height}",
             str(it.avg_brightness),
             str(it.avg_contrast),
             str(it.bucket_phash) if it.bucket_phash != -1 else "unique",
             str(it.hash_dist) if it.hash_dist > 0 else "N/A",
             it.exif_datetime[:10] if it.exif_datetime else "",
             (it.exif_camera_model[:16] if it.exif_camera_model
              else it.exif_camera_make[:16] if it.exif_camera_make else "")]
            for it in image_items
        ]
    elif mode == "Single Video" and single_video:
        hdrs  = ["Frame", "Time (s)", "Motion", "Brightness", "Contrast"]
        col_w = [0.55*inch, 0.7*inch, 0.7*inch, 0.85*inch, 0.85*inch]
        rows_data = [hdrs] + [
            [str(f.frame_id), str(f.timestamp), str(f.motion_score),
             str(f.avg_brightness), str(f.avg_contrast)]
            for f in single_video.frames
        ]
    elif mode == "Video" and video_results:
        hdrs  = ["Video 1", "Video 2", "Frame", "Time",
                 "SSIM%", "MSE", "PSNR", "Motion"]
        col_w = [1.1*inch, 1.1*inch, 0.45*inch, 0.5*inch,
                 0.55*inch, 0.55*inch, 0.6*inch, 0.55*inch]
        rows_data = [hdrs] + [
            [r.video_pair_1[:15], r.video_pair_2[:15],
             str(r.frame_id), str(r.timestamp),
             str(r.similarity_ssim), str(r.similarity_mse),
             str(r.similarity_psnr), str(r.motion_score)]
            for r in video_results
        ]
    else:
        rows_data = [["No data available"]]
        col_w     = [W]

    # Cap at 200 data rows to keep PDF size manageable
    if len(rows_data) > 202:
        rows_data = rows_data[:202]

    tbl2 = Table(rows_data, colWidths=col_w, repeatRows=1)
    tbl2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1,  0), NAVY),
        ("TEXTCOLOR",     (0, 0), (-1,  0), WHITE),
        ("FONTNAME",      (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("GRID",          (0, 0), (-1, -1), 0.3, MGRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, LGRAY]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl2)
    story.append(Spacer(1, 0.15*inch))

    # Footer
    story.append(HRFlowable(width=W, thickness=1, color=MGRAY, spaceAfter=4))
    story.append(Paragraph(
        "All analysis performed locally. No data transmitted externally. "
        "Original files were not modified. For authorized investigative use only.",
        small_s
    ))

    doc.build(story)
    return out_path
