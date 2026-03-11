# gui.py
# ResultsWindow: results table, blur/auth gate, thumbnails, CSV/PDF export,
#                AI summary, LLaVA visual description and threat flagging.
# MediaAnalyzerGUI: main window, analysis dispatch, logging.

import os
import threading
import queue
from typing import List, Optional
from dataclasses import asdict
from collections import defaultdict
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog
import numpy as np
from PIL import Image, ImageTk, ImageFilter, ImageDraw

from constants import (DEMO_PASSWORD, VID_EXTS,
                       IMAGE_PREVIEW_COLS, VIDEO_PREVIEW_COLS, SINGLE_VIDEO_COLS,
                       COLUMN_HELP)
from models import ImageItem, VideoFrameResult, SingleVideoFrame
from ai import build_ai_summary, build_llava_narrative, analyze_image_llava
from analysis import (build_image_items, assign_image_buckets, write_image_csv,
                      analyze_single_image, VideoAnalyzer, SingleVideoAnalyzer,
                      HASH_METHODS)
from pdf_report import generate_pdf_report


# RESULTS WINDOW

class ResultsWindow:
    """Sortable, filterable results table with blur/auth, CSV/PDF export, and AI summary."""

    def __init__(self, parent: tk.Tk, mode: str,
                 image_items: Optional[List[ImageItem]] = None,
                 video_results: Optional[List[VideoFrameResult]] = None,
                 single_video: Optional[SingleVideoAnalyzer] = None,
                 video_paths: Optional[list] = None,
                 initial_threshold: int = 6,
                 hash_type: str = "phash"):
        self.parent        = parent
        self.mode          = mode
        self.image_items   = image_items   or []
        self.video_results = video_results or []
        self.single_video  = single_video
        self._video_paths  = video_paths   or []
        self._initial_threshold = initial_threshold
        self._hash_type         = hash_type
        self._sort_col: Optional[str] = None
        self._sort_reverse = False
        self._authorized   = False
        self._ai_summary_text_cache = ""
        self._thumb_cache: dict = {}

        self.win = tk.Toplevel(parent)
        self.win.title("Results Preview")
        self.win.geometry("1350x700")
        self.win.minsize(900, 450)

        if self.mode in ("Image", "Single Image") and self.image_items:
            self._preload_thumbnails()

        self._build_ui()
        self._populate()

    # -- Thumbnail preloading -------------------------------------------------

    def _preload_thumbnails(self):
        """Build blurred and clear 64x64 thumbnails for each image."""
        THUMB_W    = 52
        THUMB_H    = 52
        THUMB_SIZE = (THUMB_W, THUMB_H)

        for item in self.image_items:
            try:
                with Image.open(item.path) as im:
                    im = im.convert("RGB")
                    im.thumbnail(THUMB_SIZE, Image.LANCZOS)

                    # Center on a dark background
                    padded = Image.new("RGB", (THUMB_W, THUMB_H), (20, 20, 20))
                    offset = ((THUMB_W - im.width) // 2,
                              (THUMB_H - im.height) // 2)
                    padded.paste(im, offset)

                    # Blurred version
                    blurred = padded.copy()
                    for _ in range(8):
                        blurred = blurred.filter(
                            ImageFilter.GaussianBlur(radius=6))
                    blurred_arr = np.array(blurred, dtype=np.float32) * 0.6
                    blurred = Image.fromarray(blurred_arr.astype(np.uint8))

                    self._thumb_cache[item.filename] = (
                        ImageTk.PhotoImage(blurred),
                        ImageTk.PhotoImage(padded),
                    )
            except Exception:
                self._thumb_cache[item.filename] = (None, None)

    # -- UI construction ------------------------------------------------------

    def _build_ui(self):
        # Summary bar
        self.summary_var = tk.StringVar()
        tk.Label(self.win, textvariable=self.summary_var, anchor="w",
                 font=("", 9, "italic"), fg="#444").pack(fill="x", padx=10, pady=(6, 2))

        # Auth banner (image modes only)
        if self.mode in ("Image", "Single Image"):
            self._auth_frame = tk.Frame(self.win, bg="#8B1A1A", pady=4)
            self._auth_frame.pack(fill="x", padx=10, pady=(0, 4))
            self._lock_icon_lbl = tk.Label(
                self._auth_frame,
                text="RESTRICTED -- Image content is blurred. Authorized personnel only.",
                bg="#8B1A1A", fg="white", font=("", 9, "bold"), anchor="w"
            )
            self._lock_icon_lbl.pack(side="left", padx=8)
            self._auth_btn = tk.Button(
                self._auth_frame, text="Authorize Access",
                command=self._authorize,
                bg="#5a0000", fg="white", font=("", 9, "bold"),
                relief="flat", padx=10, cursor="hand2"
            )
            self._auth_btn.pack(side="right", padx=8)

        # Single-video metadata bar
        if self.mode == "Single Video" and self.single_video:
            sv   = self.single_video
            info = (f"File: {sv.filename}   |   Duration: {sv.duration}s   |   "
                    f"FPS: {sv.fps:.1f}   |   Resolution: {sv.width}x{sv.height}   |   "
                    f"Codec: {sv.codec}   |   Frames sampled: {len(sv.frames)}")
            tk.Label(self.win, text=info, anchor="w", font=("", 9),
                     fg="#1a4f8a", bg="#eef3fa").pack(fill="x", padx=10, pady=(0, 4))

        # Filter row
        filter_frame = tk.Frame(self.win)
        filter_frame.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(filter_frame, text="Filter:").pack(side="left")
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._apply_filter())
        tk.Entry(filter_frame, textvariable=self.filter_var, width=26).pack(side="left", padx=6)
        if self.mode in ("Image", "Single Image"):
            self.similar_only_var = tk.BooleanVar(value=False)
            tk.Checkbutton(filter_frame, text="Similar groups only",
                           variable=self.similar_only_var,
                           command=self._apply_filter).pack(side="left", padx=8)

            # Live hash threshold slider
            method_label = next(k for k, v in HASH_METHODS.items()
                                if v == self._hash_type)
            tk.Label(filter_frame,
                     text=f"  {method_label} threshold:",
                     font=("Arial", 9)).pack(side="left")
            self.threshold_live_var = tk.IntVar(value=self._initial_threshold)
            self.threshold_live_lbl = tk.Label(filter_frame, text=str(self._initial_threshold),
                                                font=("Arial", 9, "bold"),
                                                fg="#1A4F8A", width=2)
            self.threshold_live_lbl.pack(side="left")
            tk.Scale(filter_frame, from_=0, to=20,
                     variable=self.threshold_live_var,
                     orient="horizontal", length=120,
                     showvalue=False, width=10,
                     command=self._on_threshold_change).pack(side="left", padx=(2, 8))

            # View Groups button
            tk.Button(filter_frame, text="View Groups",
                      command=self._open_group_viewer,
                      bg="#0D2B4E", fg="white",
                      font=("Arial", 9, "bold"),
                      relief="flat", padx=10, cursor="hand2").pack(side="left", padx=(4, 0))

        # Button bar (packed before treeview so it anchors to the bottom)
        btn_frame = tk.Frame(self.win)
        btn_frame.pack(fill="x", padx=10, pady=(4, 2), side="bottom")
        tk.Button(btn_frame, text="Export CSV", command=self._export_csv,
                  bg="#2d6a2d", fg="white", font=("Arial", 10, "bold"),
                  relief="flat", padx=8, cursor="hand2").pack(side="left")
        tk.Button(btn_frame, text="Export PDF Report", command=self._export_pdf,
                  bg="#1a4f8a", fg="white", font=("Arial", 10, "bold"),
                  relief="flat", padx=8, cursor="hand2").pack(side="left", padx=6)
        tk.Button(btn_frame, text="Close", command=self.win.destroy,
                  font=("Arial", 9), relief="flat", padx=8).pack(side="left", padx=4)
        self.export_status = tk.Label(btn_frame, text="", fg="green")
        self.export_status.pack(side="left")
        if self.mode in ("Image", "Single Image"):
            self.groups_only_var = tk.BooleanVar(value=False)
            tk.Checkbutton(btn_frame, text="Export grouped images only",
                           variable=self.groups_only_var,
                           font=("Arial", 9)).pack(side="left", padx=(12, 0))
        else:
            self.groups_only_var = tk.BooleanVar(value=False)
        tk.Label(btn_frame,
                 text="  Yellow = pHash similarity group   Red = LLaVA flag   Double-click image to preview",
                 font=("Arial", 8), fg="#666").pack(side="right", padx=10)

        # AI Analysis panel
        ai_frame = tk.LabelFrame(self.win,
                                 text="AI Analysis (requires Ollama running locally)",
                                 padx=8, pady=8,
                                 font=("Arial", 9, "bold"), fg="#0D2B4E")
        ai_frame.pack(fill="x", padx=10, pady=(0, 4), side="bottom")

        # Button rows -- image and video modes show different tools
        if self.mode in ("Image", "Single Image"):
            # Row 1: description of what each button does
            desc_row = tk.Frame(ai_frame, bg="#EEF3FA")
            desc_row.pack(fill="x", pady=(0, 6))
            tk.Label(desc_row,
                     text="Forensic Summary interprets the analysis metrics in plain English using Llama 3, while "
                          "Visual Analysis describes what is seen in the image and flags weapons, drugs, or contraband using LLaVA.",
                     font=("Arial", 8), fg="#444", bg="#EEF3FA",
                     anchor="w", wraplength=900, justify="left",
                     pady=4, padx=6).pack(fill="x")

            # Row 2: buttons + status
            btn_row = tk.Frame(ai_frame)
            btn_row.pack(fill="x")
            self.ai_summary_btn = tk.Button(btn_row,
                                            text="Forensic Summary (Llama 3)",
                                            command=self._generate_summary,
                                            bg="#1A4F8A", fg="white",
                                            font=("Arial", 9, "bold"),
                                            relief="flat", padx=10, pady=4,
                                            cursor="hand2",
                                            activebackground="#0D2B4E",
                                            activeforeground="white")
            self.ai_summary_btn.pack(side="left", padx=(0, 8))
            self.llava_img_btn = tk.Button(btn_row,
                                           text="Visual Analysis + Threat Flag (LLaVA)",
                                           command=self._analyze_image_with_llava,
                                           bg="#117A8B", fg="white",
                                           font=("Arial", 9, "bold"),
                                           relief="flat", padx=10, pady=4,
                                           cursor="hand2",
                                           activebackground="#0D5F6E",
                                           activeforeground="white")
            self.llava_img_btn.pack(side="left", padx=(0, 8))
            self.ai_status_label = tk.Label(btn_row, text="",
                                            font=("Arial", 8), fg="#666")
            self.ai_status_label.pack(side="left", padx=4)

        elif self.mode in ("Video", "Single Video"):
            # Row 1: description
            desc_row = tk.Frame(ai_frame, bg="#EEF3FA")
            desc_row.pack(fill="x", pady=(0, 6))
            tk.Label(desc_row,
                     text="Forensic Summary interprets the analysis metrics in plain English using Llama 3, while "
                          "Visual Description sends sampled frames to LLaVA to describe what is seen, "
                          "then Llama 3 synthesizes the descriptions into a chronological narrative.",
                     font=("Arial", 8), fg="#444", bg="#EEF3FA",
                     anchor="w", wraplength=900, justify="left",
                     pady=4, padx=6).pack(fill="x")

            # Row 2: buttons + status
            btn_row = tk.Frame(ai_frame)
            btn_row.pack(fill="x")
            self.ai_summary_btn = tk.Button(btn_row,
                                            text="Forensic Summary (Llama 3)",
                                            command=self._generate_summary,
                                            bg="#1A4F8A", fg="white",
                                            font=("Arial", 9, "bold"),
                                            relief="flat", padx=10, pady=4,
                                            cursor="hand2",
                                            activebackground="#0D2B4E",
                                            activeforeground="white")
            self.ai_summary_btn.pack(side="left", padx=(0, 8))
            self.llava_btn = tk.Button(btn_row,
                                       text="Visual Description (LLaVA + Llama 3)",
                                       command=self._generate_visual_description,
                                       bg="#117A8B", fg="white",
                                       font=("Arial", 9, "bold"),
                                       relief="flat", padx=10, pady=4,
                                       cursor="hand2",
                                       activebackground="#0D5F6E",
                                       activeforeground="white")
            self.llava_btn.pack(side="left", padx=(0, 8))
            self.ai_status_label = tk.Label(btn_row, text="",
                                            font=("Arial", 8), fg="#666")
            self.ai_status_label.pack(side="left", padx=4)

        self.ai_summary_text = tk.Text(ai_frame, height=6, wrap="word",
                                       font=("Arial", 10), fg="#1a1a1a", bg="#f0f4ff",
                                       relief="flat", padx=6, pady=4,
                                       state="disabled")
        self.ai_summary_text.pack(fill="x", pady=(8, 0))

        # Packed last so expand fills the remaining space
        tree_frame = tk.Frame(self.win)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=4)

        cols    = (IMAGE_PREVIEW_COLS if self.mode in ("Image", "Single Image")
                   else SINGLE_VIDEO_COLS if self.mode == "Single Video"
                   else VIDEO_PREVIEW_COLS)
        col_ids = [c[0] for c in cols]
        self.tree = ttk.Treeview(tree_frame, columns=col_ids,
                                 show="tree headings", selectmode="browse")
        # Column #0 is the tree icon column...use it for thumbnails in image modes
        if self.mode in ("Image", "Single Image"):
            self.tree.heading("#0", text="Preview")
            self.tree.column("#0", width=58, anchor="center", stretch=False)
        else:
            self.tree.column("#0", width=0, minwidth=0, stretch=False)

        method_display = next(
            (k for k, v in HASH_METHODS.items() if v == self._hash_type),
            "pHash"
        )
        for col_id, col_label, col_width in cols:
            label = f"{method_display} Value" if col_id == "hash_value" else col_label
            self.tree.heading(col_id, text=label,
                              command=lambda c=col_id: self._sort_by(c))
            self.tree.column(col_id, width=col_width, anchor="w", stretch=False)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right",  fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)

        if self.mode in ("Image", "Single Image"):
            self.tree.bind("<Double-1>", self._on_image_double_click)

        # Right-click on column headers to show help
        self.tree.bind("<Button-3>", self._on_header_right_click)

        # Single-click on Maps Link cell opens it in the browser
        self.tree.bind("<ButtonRelease-1>", self._on_cell_click)

        self.tree.tag_configure("odd",       background="#f5f5f5")
        self.tree.tag_configure("even",      background="#ffffff")
        self.tree.tag_configure("highlight", background="#fff3cd")
        self.tree.tag_configure("flagged",   background="#FFCCCC")

    # Live threshold re-bucketing

    def _on_threshold_change(self, val):
        """Re-run pHash bucketing live when the threshold slider moves."""
        threshold = int(float(val))
        self.threshold_live_lbl.config(text=str(threshold))
        if not self.image_items:
            return
        # Re-run Union-Find bucketing with the new threshold and hash type
        assign_image_buckets(self.image_items, threshold=threshold,
                             hash_type=self._hash_type)
        self._apply_filter()

    # Group viewer

    def _open_group_viewer(self):
        """Open a popup showing all pHash similarity groups side by side."""
        if not self.image_items:
            return

        # Build groups dict: bucket_id -> [ImageItem, ...]
        groups = defaultdict(list)
        for item in self.image_items:
            if item.bucket_phash != -1:
                groups[item.bucket_phash].append(item)
        # Only keep groups with 2+ members
        groups = {k: v for k, v in groups.items() if len(v) > 1}

        if not groups:
            tk.messagebox.showinfo(
                "No Groups",
                "No similarity groups found at the current threshold. "
                "Try increasing the pHash threshold slider.",
                parent=self.win
            )
            return

        popup = tk.Toplevel(self.win)
        popup.title(f"Similarity Groups  ({len(groups)} group(s))")
        popup.geometry("1100x680")
        popup.grab_set()

        # Header
        header = tk.Frame(popup, bg="#0D2B4E")
        header.pack(fill="x")
        tk.Label(header, text="Similarity Groups",
                 bg="#0D2B4E", fg="white",
                 font=("Arial", 13, "bold"),
                 pady=10, padx=16, anchor="w").pack(side="left")
        tk.Label(header,
                 text=f"{len(groups)} group(s)  |  "
                      f"{sum(len(v) for v in groups.values())} image(s) total  |  "
                      f"Largest group: {max(len(v) for v in groups.values())} images",
                 bg="#0D2B4E", fg="#AACCEE",
                 font=("Arial", 9), padx=16).pack(side="right")
        tk.Frame(popup, bg="#1A4F8A", height=2).pack(fill="x")

        # Scrollable canvas
        canvas_frame = tk.Frame(popup)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=8)
        canvas = tk.Canvas(canvas_frame, bg="#F5F7FA")
        vsb = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=vsb.set)

        inner = tk.Frame(canvas, bg="#F5F7FA")
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_resize(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", _on_resize)
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))

        THUMB = 110
        photo_refs = []  # prevent GC

        for group_idx, (bucket_id, items) in enumerate(
                sorted(groups.items(), key=lambda x: -len(x[1]))):
            # Group header row
            grp_header = tk.Frame(inner, bg="#DDE6F5")
            grp_header.pack(fill="x", pady=(10 if group_idx > 0 else 2, 4), padx=4)
            tk.Label(grp_header,
                     text=f"  Group {bucket_id}  —  {len(items)} images",
                     bg="#DDE6F5", fg="#0D2B4E",
                     font=("Arial", 10, "bold"),
                     pady=4).pack(side="left")

            # Mean hash distance if available
            dvals = [i.hash_dist for i in items if i.hash_dist > 0]
            if dvals:
                avg_d = round(sum(dvals) / len(dvals), 2)
                tk.Label(grp_header,
                         text=f"  Avg hash distance: {avg_d}",
                         bg="#DDE6F5", fg="#555",
                         font=("Arial", 9)).pack(side="left", padx=8)

            # Thumbnails row
            thumb_row = tk.Frame(inner, bg="#F5F7FA")
            thumb_row.pack(fill="x", padx=8, pady=(0, 4))

            for item in items:
                cell = tk.Frame(thumb_row, bg="#F5F7FA", padx=4, pady=4)
                cell.pack(side="left")

                # Get or generate thumbnail
                thumb_img = None
                if item.filename in self._thumb_cache:
                    blurred, real = self._thumb_cache[item.filename]
                    thumb_img = real if self._authorized else blurred
                else:
                    try:
                        with Image.open(item.path) as im:
                            im = im.convert("RGB")
                            im.thumbnail((THUMB, THUMB), Image.LANCZOS)
                            padded = Image.new("RGB", (THUMB, THUMB), (240, 243, 248))
                            padded.paste(im, ((THUMB - im.width) // 2,
                                             (THUMB - im.height) // 2))
                            if not self._authorized:
                                for _ in range(8):
                                    padded = padded.filter(
                                        ImageFilter.GaussianBlur(radius=6))
                            thumb_img = ImageTk.PhotoImage(padded)
                            photo_refs.append(thumb_img)
                    except Exception:
                        thumb_img = None

                if thumb_img:
                    img_lbl = tk.Label(cell, image=thumb_img, bg="#F5F7FA",
                                       cursor="hand2")
                    img_lbl.image = thumb_img
                    img_lbl.pack()
                    # Double-click to preview full image
                    img_lbl.bind("<Double-1>",
                        lambda e, fn=item.filename: self._preview_image_by_name(fn))

                # Filename label under thumbnail
                short = item.filename if len(item.filename) <= 14 else item.filename[:12] + "…"
                tk.Label(cell, text=short, bg="#F5F7FA",
                         font=("Arial", 7), fg="#444",
                         wraplength=THUMB).pack()

        tk.Button(popup, text="Close", command=popup.destroy,
                  bg="#1A4F8A", fg="white",
                  font=("Arial", 9, "bold"),
                  relief="flat", padx=14, pady=4,
                  cursor="hand2").pack(pady=(4, 10))

        # Store refs to prevent GC
        popup._photo_refs = photo_refs

    def _preview_image_by_name(self, filename: str):
        """Find an image item by filename and open its preview popup."""
        for item in self.image_items:
            if item.filename == filename:
                self._show_image_preview(item.filename)
                return

    # Cell click handler (Maps Link)

    def _on_cell_click(self, event):
        """If the user clicks on a Maps Link cell, open it in the browser."""
        region = self.tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        col_id = self.tree.identify_column(event.x)
        try:
            col_index = int(col_id.lstrip("#")) - 1
        except ValueError:
            return
        cols = (IMAGE_PREVIEW_COLS if self.mode in ("Image", "Single Image")
                else SINGLE_VIDEO_COLS if self.mode == "Single Video"
                else VIDEO_PREVIEW_COLS)
        if col_index < 0 or col_index >= len(cols):
            return
        if cols[col_index][0] != "exif_gps_link":
            return
        iid = self.tree.identify_row(event.y)
        if not iid:
            return
        values = self.tree.item(iid, "values")
        if col_index >= len(values):
            return
        url = values[col_index]
        if url and url.startswith("http"):
            import webbrowser
            webbrowser.open(url)

    # Column help (right-click header)

    def _on_header_right_click(self, event):
        """Detect right-click on a column header and show a help context menu."""
        region = self.tree.identify_region(event.x, event.y)
        if region != "heading":
            return
        col_id = self.tree.identify_column(event.x)
        if not col_id:
            return
        # col_id is like "#1", "#2" -- convert to field name
        try:
            col_index = int(col_id.lstrip("#")) - 1
        except ValueError:
            return
        cols = (IMAGE_PREVIEW_COLS if self.mode in ("Image", "Single Image")
                else SINGLE_VIDEO_COLS if self.mode == "Single Video"
                else VIDEO_PREVIEW_COLS)
        if col_index < 0 or col_index >= len(cols):
            return
        field_name = cols[col_index][0]
        if field_name not in COLUMN_HELP:
            return
        menu = tk.Menu(self.win, tearoff=0)
        menu.add_command(
            label="What is this?",
            command=lambda: self._show_column_help(field_name)
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _show_column_help(self, field_name: str):
        """Show a popup explaining what a column measures."""
        if field_name not in COLUMN_HELP:
            return
        title, body = COLUMN_HELP[field_name]

        popup = tk.Toplevel(self.win)
        popup.title(f"Column Help: {title}")
        popup.resizable(False, False)
        popup.grab_set()

        # Header bar
        header = tk.Frame(popup, bg="#0D2B4E")
        header.pack(fill="x")
        tk.Label(header, text=title, bg="#0D2B4E", fg="white",
                 font=("Arial", 13, "bold"), pady=10, padx=16,
                 anchor="w").pack(fill="x")

        # Body text
        body_frame = tk.Frame(popup, padx=20, pady=14)
        body_frame.pack(fill="both")
        tk.Message(body_frame, text=body, width=380,
                   font=("Arial", 10), fg="#222",
                   justify="left").pack(anchor="w")

        tk.Button(popup, text="Close", command=popup.destroy,
                  bg="#1A4F8A", fg="white", font=("Arial", 9, "bold"),
                  padx=14, pady=4).pack(pady=(0, 14))

        popup.update_idletasks()
        x = self.win.winfo_x() + (self.win.winfo_width()  - popup.winfo_width())  // 2
        y = self.win.winfo_y() + (self.win.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{x}+{y}")

    # Data helpers 

    def _all_rows(self) -> List[dict]:
        if self.mode in ("Image", "Single Image"):
            return [asdict(it) for it in self.image_items]
        elif self.mode == "Single Video" and self.single_video:
            return [asdict(fr) for fr in self.single_video.frames]
        return [asdict(r) for r in self.video_results]

    def _bucket_counts(self) -> dict:
        counts: dict = defaultdict(int)
        for r in self._all_rows():
            counts[r.get("bucket_phash", -1)] += 1
        return counts

    # Table population

    def _populate(self, rows: Optional[List[dict]] = None):
        self.tree.delete(*self.tree.get_children())
        if rows is None:
            rows = self._all_rows()

        bucket_counts = self._bucket_counts() if self.mode in ("Image", "Single Image") else {}
        cols    = (IMAGE_PREVIEW_COLS if self.mode in ("Image", "Single Image")
                   else SINGLE_VIDEO_COLS if self.mode == "Single Video"
                   else VIDEO_PREVIEW_COLS)
        col_ids = [c[0] for c in cols]

        for i, row in enumerate(rows):
            values = []
            for c in col_ids:
                if c == "hash_value" and self.mode in ("Image", "Single Image"):
                    values.append(row.get(self._hash_type, ""))
                else:
                    values.append(row.get(c, ""))
            tag = "odd" if i % 2 else "even"
            if self.mode in ("Image", "Single Image"):
                bid = row.get("bucket_phash", -1)
                if bid != -1 and bucket_counts.get(bid, 0) > 1:
                    tag = "highlight"

            iid = self.tree.insert("", "end", values=values, tags=(tag,))

            if self.mode in ("Image", "Single Image"):
                pair = self._thumb_cache.get(row.get("filename", ""))
                if pair:
                    photo = pair[1] if self._authorized else pair[0]
                    if photo:
                        self.tree.item(iid, image=photo)

        style = ttk.Style()
        style.configure("Treeview",
                        rowheight=56 if self.mode in ("Image", "Single Image") else 22)
        self._update_summary(rows)

    def _update_summary(self, rows: List[dict]):
        if self.mode in ("Image", "Single Image"):
            total     = len(rows)
            # -1 means unique (no match); any other value is a group bucket
            unique    = sum(1 for r in rows if r.get("bucket_phash", -1) == -1)
            bc        = defaultdict(int)
            for r in rows:
                bid = r.get("bucket_phash", -1)
                if bid != -1:
                    bc[bid] += 1
            groups    = len(bc)
            in_groups = sum(bc.values())
            largest   = max(bc.values(), default=0)
            largest_str = f"   |   Largest group: {largest}" if largest > 0 else ""
            self.summary_var.set(
                f"{total} image(s)   |   {unique} unique   |   "
                f"{groups} similarity group(s)   |   {in_groups} in groups"
                f"{largest_str}"
            )
        elif self.mode == "Single Video":
            total   = len(rows)
            motions = [r["motion_score"] for r in rows if "motion_score" in r]
            avg_m   = round(sum(motions) / len(motions), 2) if motions else 0
            self.summary_var.set(
                f"{total} frame(s) sampled   |   Avg motion score: {avg_m}"
            )
        else:
            total = len(rows)
            if rows:
                sims     = [r.get("similarity_ssim", 0) for r in rows]
                msev     = [r.get("similarity_mse",  0) for r in rows]
                psnv     = [r.get("similarity_psnr", 0) for r in rows]
                self.summary_var.set(
                    f"{total} frame comparison(s)   |   "
                    f"Avg SSIM: {round(sum(sims)/len(sims),1)}%   |   "
                    f"Avg MSE: {round(sum(msev)/len(msev),1)}   |   "
                    f"Avg PSNR: {round(sum(psnv)/len(psnv),1)} dB"
                )
            else:
                self.summary_var.set("No results.")

    # Filtering and sorting

    def _apply_filter(self):
        query     = self.filter_var.get().strip().lower()
        rows      = self._all_rows()
        if query:
            rows = [r for r in rows
                    if query in str(r.get("filename",     "")).lower()
                    or query in str(r.get("video_pair_1", "")).lower()
                    or query in str(r.get("video_pair_2", "")).lower()]
        if (self.mode in ("Image", "Single Image")
                and hasattr(self, "similar_only_var")
                and self.similar_only_var.get()):
            bc   = self._bucket_counts()
            rows = [r for r in rows if bc.get(r.get("bucket_phash", -1), 0) > 1]
        self._populate(rows)

    def _sort_by(self, col: str):
        self._sort_reverse = (self._sort_col == col) and not self._sort_reverse
        self._sort_col     = col
        rows               = self._all_rows()
        sort_field = self._hash_type if col == "hash_value" else col
        try:
            rows.sort(key=lambda r: (r.get(sort_field) is None, r.get(sort_field, "")),
                      reverse=self._sort_reverse)
        except TypeError:
            rows.sort(key=lambda r: str(r.get(sort_field, "")), reverse=self._sort_reverse)
        self._populate(rows)
        cols = (IMAGE_PREVIEW_COLS if self.mode in ("Image", "Single Image")
                else SINGLE_VIDEO_COLS if self.mode == "Single Video"
                else VIDEO_PREVIEW_COLS)
        method_display = next(
            (k for k, v in HASH_METHODS.items() if v == self._hash_type), "pHash"
        )
        for col_id, col_label, _ in cols:
            label = f"{method_display} Value" if col_id == "hash_value" else col_label
            arrow = (" ^" if not self._sort_reverse else " v") if col_id == col else ""
            self.tree.heading(col_id, text=label + arrow)

    # CSV export 

    def _export_csv(self):
        path = filedialog.asksaveasfilename(
            parent=self.win, defaultextension=".csv",
            filetypes=[("CSV", "*.csv")], title="Export CSV"
        )
        if not path:
            return
        try:
            if self.mode in ("Image", "Single Image"):
                items = self.image_items
                if self.groups_only_var.get():
                    items = [it for it in items if it.bucket_phash != -1]
                    if not items:
                        messagebox.showinfo("No Groups",
                            "No similarity groups found at the current threshold. "
                            "Export cancelled.", parent=self.win)
                        return
                write_image_csv(items, path)
            elif self.mode == "Single Video" and self.single_video:
                self.single_video.write_csv(path)
            else:
                va = VideoAnalyzer()
                va.results = self.video_results
                va.write_csv(path)
            self.export_status.config(text=f"Saved: {os.path.basename(path)}", fg="green")
        except Exception as e:
            messagebox.showerror("Export Error", str(e), parent=self.win)

    # PDF export 

    def _export_pdf(self):
        path = filedialog.asksaveasfilename(
            parent=self.win, defaultextension=".pdf",
            filetypes=[("PDF Report", "*.pdf")], title="Export PDF Report"
        )
        if not path:
            return
        items = self.image_items or None
        groups_only = self.groups_only_var.get()
        if groups_only and self.mode in ("Image", "Single Image") and items:
            items = [it for it in items if it.bucket_phash != -1]
            if not items:
                messagebox.showinfo("No Groups",
                    "No similarity groups found at the current threshold. "
                    "Export cancelled.", parent=self.win)
                return
        self.export_status.config(text="Generating PDF...", fg="#555")
        self.win.update_idletasks()
        try:
            generate_pdf_report(
                out_path=path,
                mode=self.mode,
                image_items=items,
                video_results=self.video_results or None,
                single_video=self.single_video,
                ai_summary=self._ai_summary_text_cache,
                groups_only=groups_only,
            )
            self.export_status.config(
                text=f"PDF saved: {os.path.basename(path)}", fg="green"
            )
        except Exception as e:
            messagebox.showerror("PDF Export Error", str(e), parent=self.win)
            self.export_status.config(text="PDF export failed.", fg="red")

    # AI summary 

    def _generate_summary(self):
        self.ai_summary_btn.config(state="disabled")
        self.ai_status_label.config(text="  Generating summary...")
        self.ai_summary_text.configure(state="normal")
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.configure(state="disabled")

        vr = self.video_results
        if self.mode == "Single Video" and self.single_video:
            vr = [VideoFrameResult(
                video_pair_1=self.single_video.filename,
                video_pair_2="",
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                similarity_ssim=0,
                similarity_mse=0,
                similarity_psnr=0,
                motion_score=f.motion_score,
            ) for f in self.single_video.frames]

        def _run():
            try:
                summary = build_ai_summary(
                    self.mode,
                    image_items=self.image_items or None,
                    video_results=vr or None,
                )
                self.win.after(0, lambda s=summary: self._show_summary(s))
            except RuntimeError as e:
                self.win.after(0, lambda err=str(e): self._show_summary_error(err))

        threading.Thread(target=_run, daemon=True).start()

    def _show_summary(self, text: str):
        self.ai_summary_text.configure(state="normal")
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.insert(tk.END, text)
        self.ai_summary_text.configure(state="disabled")
        self.ai_status_label.config(text="  Summary generated.", fg="green")
        self.ai_summary_btn.config(state="normal")
        self._ai_summary_text_cache = text

    def _show_summary_error(self, error: str):
        self.ai_summary_text.configure(state="normal")
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.insert(tk.END, f"Error: {error}")
        self.ai_summary_text.configure(state="disabled")
        self.ai_status_label.config(text="  Failed -- see message above.", fg="red")
        self.ai_summary_btn.config(state="normal")

    # LLaVA visual description (video modes)

    def _generate_visual_description(self):
        """
        Samples frames from the video, sends each to LLaVA for a visual
        description, then synthesizes all descriptions with Llama 3.
        """
        self.llava_btn.config(state="disabled")
        self.ai_status_label.config(
            text="  LLaVA is describing frames... this may take a few minutes.",
            fg="#555"
        )
        self.ai_summary_text.configure(state="normal")
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.configure(state="disabled")

        MAX_FRAMES = 10

        def _collect_frames():
            frames_data = []
            try:
                if self.mode == "Single Video" and self.single_video:
                    import cv2
                    path = getattr(self.single_video, "_source_path", None)
                    if not path:
                        path = filedialog.askopenfilename(
                            title="Locate the video file for visual description",
                            parent=self.win,
                            filetypes=[("Video files",
                                        "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v")]
                        )
                        if not path:
                            raise RuntimeError("No video file selected.")

                    sampled = self.single_video.frames
                    step    = max(1, len(sampled) // MAX_FRAMES)
                    subset  = sampled[::step][:MAX_FRAMES]

                    cap = cv2.VideoCapture(path)
                    for svf in subset:
                        cap.set(cv2.CAP_PROP_POS_MSEC, svf.timestamp * 1000)
                        ret, frame = cap.read()
                        if ret:
                            pil_img = Image.fromarray(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            )
                            frames_data.append((svf.timestamp, svf.frame_id, pil_img))
                    cap.release()
                    filename = self.single_video.filename

                elif self.mode == "Video" and self.video_results:
                    import cv2
                    path = (self._video_paths[0] if self._video_paths else None)
                    if not path:
                        path = filedialog.askopenfilename(
                            title=f"Locate '{self.video_results[0].video_pair_1}' for visual description",
                            parent=self.win,
                            filetypes=[("Video files",
                                        "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v")]
                        )
                        if not path:
                            raise RuntimeError("No video file selected.")

                    results = self.video_results
                    step    = max(1, len(results) // MAX_FRAMES)
                    subset  = results[::step][:MAX_FRAMES]

                    cap = cv2.VideoCapture(path)
                    for vfr in subset:
                        cap.set(cv2.CAP_PROP_POS_MSEC, vfr.timestamp * 1000)
                        ret, frame = cap.read()
                        if ret:
                            pil_img = Image.fromarray(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            )
                            frames_data.append((vfr.timestamp, vfr.frame_id, pil_img))
                    cap.release()
                    filename = self.video_results[0].video_pair_1
                else:
                    raise RuntimeError("Visual description is only available for video modes.")

                if not frames_data:
                    raise RuntimeError("Could not extract any frames from the video file.")

                return frames_data, filename

            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"Frame extraction failed: {e}")

        def _run():
            try:
                frames_data, filename = _collect_frames()
                total = len(frames_data)
                self.win.after(0, lambda: self.ai_status_label.config(
                    text=f"  Describing {total} frames with LLaVA...", fg="#555"
                ))

                def _progress(cur, tot, label):
                    self.win.after(0, lambda c=cur, t=tot, l=label:
                        self.ai_status_label.config(
                            text=f"  {l} ({c}/{t})", fg="#555"
                        )
                    )

                narrative = build_llava_narrative(
                    frames_data, filename, progress_cb=_progress
                )
                self.win.after(0, lambda n=narrative: self._show_visual_description(n))

            except RuntimeError as e:
                self.win.after(0, lambda err=str(e): self._show_visual_desc_error(err))

        threading.Thread(target=_run, daemon=True).start()

    def _show_visual_description(self, text: str):
        self.ai_summary_text.configure(state="normal")
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.insert(tk.END, text)
        self.ai_summary_text.configure(state="disabled")
        self.ai_status_label.config(
            text="  Visual description generated (LLaVA + Llama 3).", fg="green"
        )
        self.llava_btn.config(state="normal")
        self._ai_summary_text_cache = text

    def _show_visual_desc_error(self, error: str):
        self.ai_summary_text.configure(state="normal")
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.insert(tk.END, f"Error: {error}")
        self.ai_summary_text.configure(state="disabled")
        self.ai_status_label.config(
            text="  Visual description failed -- see message above.", fg="red"
        )
        self.llava_btn.config(state="normal")

    # LLaVA image analysis (image modes)

    def _analyze_image_with_llava(self):
        """
        Run LLaVA visual description and threat flagging on a still image.
        In Single Image mode, analyzes the one image automatically.
        In Image folder mode, analyzes the selected row or the first image.
        """
        pil_img  = None
        filename = ""

        if self.mode == "Single Image" and self.image_items:
            item = self.image_items[0]
            filename = item.filename
            try:
                pil_img = Image.open(item.path)
            except Exception as e:
                self.ai_status_label.config(
                    text=f"  Could not open image: {e}", fg="red"
                )
                return

        elif self.mode == "Image" and self.image_items:
            selected = self.tree.selection()
            match = None
            if selected:
                iid     = selected[0]
                vals    = self.tree.item(iid, "values")
                col_ids = [c[0] for c in IMAGE_PREVIEW_COLS]
                try:
                    fname_idx = col_ids.index("filename")
                    fname     = vals[fname_idx] if fname_idx < len(vals) else ""
                    match     = next((it for it in self.image_items
                                      if it.filename == fname), None)
                except (ValueError, IndexError):
                    pass
            if match is None:
                match = self.image_items[0]
            try:
                pil_img  = Image.open(match.path)
                filename = match.filename
            except Exception as e:
                self.ai_status_label.config(
                    text=f"  Could not open image: {e}", fg="red"
                )
                return

        if pil_img is None:
            self.ai_status_label.config(
                text="  No image available to analyze.", fg="red"
            )
            return

        self.llava_img_btn.config(state="disabled")
        self.ai_status_label.config(
            text=f"  LLaVA is analyzing '{filename}'...", fg="#555"
        )
        self.ai_summary_text.configure(state="normal", bg="white")
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.configure(state="disabled")

        img_copy = pil_img.copy()
        fname    = filename

        def _run():
            try:
                result = analyze_image_llava(img_copy, fname)
                self.win.after(0, lambda r=result, f=fname:
                    self._show_llava_image_result(r, f))
            except RuntimeError as e:
                self.win.after(0, lambda err=str(e):
                    self._show_llava_image_error(err))

        threading.Thread(target=_run, daemon=True).start()

    def _show_llava_image_result(self, result: dict, filename: str):
        description  = result.get("description", "(no description)")
        flagged      = result.get("flagged", False)
        flag_details = result.get("flag_details", "")

        text = f"FILE: {filename}\n\n{description}"
        if flagged:
            text += f"\n\nFLAGGED: {flag_details}" if flag_details else "\n\nFLAGGED"

        bg_color = "#FFE4E4" if flagged else "white"
        self.ai_summary_text.configure(state="normal", bg=bg_color)
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.insert(tk.END, text)
        self.ai_summary_text.configure(state="disabled")

        if flagged:
            self.ai_status_label.config(
                text="  FLAGGED: Potential item of investigative concern detected.",
                fg="red"
            )
            # Highlight the flagged row in the treeview
            for iid in self.tree.get_children():
                vals    = self.tree.item(iid, "values")
                col_ids = [c[0] for c in IMAGE_PREVIEW_COLS]
                try:
                    fname_idx = col_ids.index("filename")
                    if vals[fname_idx] == filename:
                        self.tree.item(iid, tags=("flagged",))
                        break
                except (ValueError, IndexError):
                    pass
            detail_msg = f"\n\nDetail: {flag_details}" if flag_details else ""
            messagebox.showwarning(
                title="LLaVA Flag Alert",
                message=(
                    f"LLaVA has flagged '{filename}' as containing potential "
                    f"items of investigative concern.{detail_msg}\n\n"
                    f"Review the AI summary panel for full details."
                ),
                parent=self.win
            )
        else:
            self.ai_status_label.config(
                text="  LLaVA analysis complete. No flags raised.", fg="green"
            )

        self._ai_summary_text_cache = text
        self.llava_img_btn.config(state="normal")

    def _show_llava_image_error(self, error: str):
        self.ai_summary_text.configure(state="normal", bg="white")
        self.ai_summary_text.delete("1.0", tk.END)
        self.ai_summary_text.insert(tk.END, f"Error: {error}")
        self.ai_summary_text.configure(state="disabled")
        self.ai_status_label.config(
            text="  LLaVA analysis failed -- see message above.", fg="red"
        )
        self.llava_img_btn.config(state="normal")

    # Auth and image preview 

    def _authorize(self):
        pwd = simpledialog.askstring(
            "Authorization Required",
            "Enter investigator password to unlock image viewing:",
            show="*", parent=self.win
        )
        if pwd is None:
            return
        if pwd == DEMO_PASSWORD:
            self._authorized = True
            self._auth_frame.config(bg="#1D6A3A")
            self._lock_icon_lbl.config(
                text="AUTHORIZED -- Image content unlocked. Viewing full resolution previews.",
                bg="#1D6A3A"
            )
            self._auth_btn.config(text="Authorized", state="disabled", bg="#145228")
            self._populate()
        else:
            messagebox.showerror("Access Denied",
                                 "Incorrect password. Access denied.", parent=self.win)

    def _on_image_double_click(self, event):
        """Full-size preview popup on double-click. Blurred if not authorized."""
        item = self.tree.focus()
        if not item:
            return
        col_ids = [c[0] for c in IMAGE_PREVIEW_COLS]
        values  = self.tree.item(item, "values")
        if not values:
            return
        row      = dict(zip(col_ids, values))
        filename = row.get("filename", "")
        img_item = next((it for it in self.image_items if it.filename == filename), None)
        if img_item is None:
            return

        popup = tk.Toplevel(self.win)
        popup.title(f"Preview -- {filename}")
        popup.geometry("520x560")
        popup.resizable(False, False)

        try:
            pil_img = Image.open(img_item.path).convert("RGB")
            pil_img.thumbnail((480, 400))
            if not self._authorized:
                display_img = pil_img
                for _ in range(12):
                    display_img = display_img.filter(ImageFilter.GaussianBlur(radius=8))
            else:
                display_img = pil_img

            photo = ImageTk.PhotoImage(display_img)
            lbl   = tk.Label(popup, image=photo, bg="#1a1a1a")
            lbl.image = photo
            lbl.pack(fill="both", expand=True, padx=10, pady=10)

            if not self._authorized:
                tk.Label(popup,
                         text="RESTRICTED -- Authorize access to view this image",
                         bg="#8B1A1A", fg="white", font=("", 10, "bold")
                         ).pack(fill="x", padx=10, pady=(0, 4))
            else:
                tk.Label(popup,
                         text=(f"{img_item.filename}  |  {img_item.width}x{img_item.height}  |  "
                               f"{img_item.bytes // 1024} KB  |  "
                               f"Brightness: {img_item.avg_brightness}"),
                         fg="#444", font=("", 8)).pack(pady=(0, 6))
        except Exception as e:
            tk.Label(popup, text=f"Could not load image:\n{e}", fg="red").pack(pady=20)


# MAIN GUI

class MediaAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Media Analyzer")
        self.root.resizable(True, True)
        self.root.minsize(780, 540)
        self._log_queue: queue.Queue = queue.Queue()
        self._set_icon()
        self._build_ui()
        self._poll_log_queue()

    def _set_icon(self):
        """Generate and set a magnifying glass window icon programmatically."""
        try:
            size = 32
            img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse([2, 2, 22, 22], outline=(26, 79, 138), width=3)
            draw.ellipse([4, 4, 20, 20], fill=(200, 220, 255, 180))
            draw.line([18, 18, 29, 29], fill=(26, 79, 138), width=4)
            icon = ImageTk.PhotoImage(img)
            self.root.iconphoto(True, icon)
            self._icon_ref = icon   # prevent garbage collection
        except Exception:
            pass

    def _build_ui(self):
        root = self.root

        # Apply clam theme for a cleaner modern look
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor="#dde3ec", background="#1A4F8A",
                        thickness=8)
        style.configure("TCombobox", fieldbackground="white")

        # Menu bar
        menubar   = tk.Menu(root)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About Media Analyzer",
                              command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        root.config(menu=menubar)

        # Header banner
        banner = tk.Frame(root, bg="#0D2B4E")
        banner.pack(fill="x")
        tk.Label(banner, text="Media Analyzer",
                 bg="#0D2B4E", fg="white",
                 font=("Arial", 16, "bold"),
                 pady=10, padx=16, anchor="w").pack(side="left")
        tk.Label(banner,
                 text="Forensic Image and Video Analysis",
                 bg="#0D2B4E", fg="#AACCEE",
                 font=("Arial", 9),
                 padx=16, anchor="e").pack(side="right")

        # Divider
        tk.Frame(root, bg="#1A4F8A", height=2).pack(fill="x")

        # Content area with consistent padding
        content = tk.Frame(root, padx=14, pady=8)
        content.pack(fill="x")

        # Row 1: Mode + Input
        row1 = tk.Frame(content)
        row1.pack(fill="x", pady=(0, 6))

        mode_frame = tk.LabelFrame(row1, text="Mode", padx=8, pady=6,
                                   font=("Arial", 9, "bold"), fg="#0D2B4E")
        mode_frame.pack(side="left", padx=(0, 10))
        self.mode_var = tk.StringVar(value="Image")
        for val, lbl in [("Image",        "Images (folder)"),
                         ("Single Image", "Single Image"),
                         ("Video",        "Videos (folder)"),
                         ("Single Video", "Single Video")]:
            tk.Radiobutton(mode_frame, text=lbl, variable=self.mode_var,
                           value=val, command=self._on_mode_change,
                           font=("Arial", 9)).pack(side="left", padx=5)

        input_frame = tk.LabelFrame(row1, text="Input", padx=8, pady=6,
                                    font=("Arial", 9, "bold"), fg="#0D2B4E")
        input_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_var = tk.StringVar()
        tk.Entry(input_frame, textvariable=self.input_var,
                 width=52, font=("Arial", 9)).pack(side="left", padx=(0, 6))
        tk.Button(input_frame, text="Browse...",
                  command=self._browse_input,
                  bg="#1A4F8A", fg="white", font=("Arial", 9, "bold"),
                  relief="flat", padx=10, cursor="hand2").pack(side="left")
        self.input_hint = tk.Label(input_frame, text="folder",
                                   font=("Arial", 8), fg="#888")
        self.input_hint.pack(side="left", padx=6)

        # Row 2: Analysis options
        row2 = tk.Frame(content)
        row2.pack(fill="x", pady=(0, 8))

        opts_frame = tk.LabelFrame(row2, text="Analysis Options", padx=8, pady=6,
                                   font=("Arial", 9, "bold"), fg="#0D2B4E")
        opts_frame.pack(side="left", padx=(0, 10))
        tk.Label(opts_frame, text="Interval (s):", font=("Arial", 9)).pack(side="left")
        self.interval_var = tk.StringVar(value="0.5")
        tk.Entry(opts_frame, textvariable=self.interval_var,
                 width=5, font=("Arial", 9)).pack(side="left", padx=(4, 10))
        tk.Label(opts_frame, text="Hash method:", font=("Arial", 9)).pack(side="left", padx=(10, 0))
        self.hash_method_var = tk.StringVar(value="pHash")
        ttk.Combobox(opts_frame, textvariable=self.hash_method_var,
                     values=list(HASH_METHODS.keys()),
                     state="readonly", width=16,
                     font=("Arial", 9)).pack(side="left", padx=(4, 10))
        tk.Label(opts_frame, text="Threshold:", font=("Arial", 9)).pack(side="left")
        self.threshold_var = tk.StringVar(value="6")
        tk.Entry(opts_frame, textvariable=self.threshold_var,
                 width=4, font=("Arial", 9)).pack(side="left", padx=4)

        # Run button
        self.run_btn = tk.Button(root, text="Run Analysis",
                                 command=self._start_analysis,
                                 bg="#1D6A3A", fg="white",
                                 font=("Arial", 12, "bold"),
                                 padx=24, pady=8,
                                 relief="flat", cursor="hand2",
                                 activebackground="#155228",
                                 activeforeground="white")
        self.run_btn.pack(pady=(2, 6))

        # Progress bar
        prog_frame = tk.Frame(root, padx=14)
        prog_frame.pack(fill="x")
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(prog_frame, variable=self.progress_var,
                        maximum=100, style="TProgressbar").pack(fill="x")
        self.progress_label = tk.Label(prog_frame, text="", anchor="w",
                                       font=("Arial", 8), fg="#555")
        self.progress_label.pack(fill="x", pady=(2, 0))

        # Log
        log_frame = tk.LabelFrame(root, text="Log", padx=6, pady=6,
                                  font=("Arial", 9, "bold"), fg="#0D2B4E")
        log_frame.pack(fill="both", expand=True, padx=14, pady=(6, 12))
        vsb = tk.Scrollbar(log_frame)
        vsb.pack(side="right", fill="y")
        self.log_box = tk.Text(log_frame, height=14, width=100,
                               yscrollcommand=vsb.set, state="disabled",
                               font=("Courier", 9), bg="#f8f9fb",
                               relief="flat", padx=6, pady=4)
        self.log_box.pack(fill="both", expand=True)
        vsb.config(command=self.log_box.yview)

    # -- Event handlers -------------------------------------------------------

    def _on_mode_change(self):
        mode = self.mode_var.get()
        self.input_var.set("")
        self.input_hint.config(
            text="file" if mode in ("Single Image", "Single Video") else "folder"
        )

    def _browse_input(self):
        mode = self.mode_var.get()
        if mode == "Single Image":
            path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image files",
                            "*.jpg *.jpeg *.png *.bmp *.gif *.webp *.tiff *.tif *.heic *.heif")]
            )
        elif mode == "Single Video":
            path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files",
                            "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v")]
            )
        else:
            path = filedialog.askdirectory(title="Select Folder")
        if path:
            self.input_var.set(path)

    # Logging

    def _poll_log_queue(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert(tk.END, msg + "\n")
                self.log_box.see(tk.END)
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def log(self, msg: str):
        self._log_queue.put(msg)

    def _set_progress(self, current: int, total: int, label: str = ""):
        pct = (current / total * 100) if total > 0 else 0
        self.root.after(0, lambda: (
            self.progress_var.set(pct),
            self.progress_label.config(text=f"{label}  ({current}/{total})")
        ))

    # Analysis dispatch

    def _start_analysis(self):
        path = self.input_var.get().strip()
        if not path:
            self.log("Please select a file or folder first.")
            return
        self.run_btn.config(state="disabled")
        self.progress_var.set(0)
        self.progress_label.config(text="")
        threading.Thread(target=self._run_analysis, args=(path,), daemon=True).start()

    def _run_analysis(self, path: str):
        mode = self.mode_var.get()
        try:
            if mode == "Image":
                self._run_image_analysis(path)
            elif mode == "Single Image":
                self._run_single_image_analysis(path)
            elif mode == "Video":
                self._run_video_analysis(path)
            elif mode == "Single Video":
                self._run_single_video_analysis(path)
        except RuntimeError as e:
            self.log(f"Error: {e}")
        except Exception as e:
            self.log(f"Unexpected error: {e}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal"))
            self.root.after(0, lambda: self.progress_label.config(text="Done."))

    # Image folder

    def _run_image_analysis(self, folder: str):
        threshold   = int(self.threshold_var.get())
        hash_key    = self.hash_method_var.get()
        hash_type   = HASH_METHODS.get(hash_key, "phash")
        self.log("Scanning images...")
        items = build_image_items(
            folder,
            progress_cb=lambda cur, tot, name:
                self._set_progress(cur, tot, f"Analyzing: {name}"),
        )
        if not items:
            self.log("No images found.")
            return
        self.log(f"Found {len(items)} image(s). Running similarity analysis ({hash_key})...")
        assign_image_buckets(items, threshold=threshold, hash_type=hash_type)

        buckets: dict = defaultdict(list)
        for it in items:
            if it.bucket_phash != -1:
                buckets[it.bucket_phash].append(it.filename)
        self.log(f"   -> {len(buckets)} {hash_key} group(s) found.")
        for bid, names in buckets.items():
            self.log(f"     Group {bid}: {', '.join(names)}")
        self.log("Opening results preview...")
        self.root.after(0, lambda: ResultsWindow(self.root, "Image", image_items=items,
                                                  initial_threshold=threshold,
                                                  hash_type=hash_type))

    # Single image

    def _run_single_image_analysis(self, path: str):
        self.log(f"Analyzing single image: {os.path.basename(path)}")
        item = analyze_single_image(Path(path))
        if item is None:
            self.log("Could not open or process the image.")
            return
        self.log(f"   Size: {item.width}x{item.height}  |  {item.ext}")
        self.log(f"   pHash: {item.phash}")
        self.log(f"   Brightness: {item.avg_brightness}  |  Contrast: {item.avg_contrast}")
        self.log("Opening results preview...")
        self.root.after(0, lambda: ResultsWindow(self.root, "Single Image",
                                                  image_items=[item]))

    # Video folder

    def _run_video_analysis(self, folder: str):
        try:
            interval = float(self.interval_var.get())
        except ValueError:
            self.log("Invalid interval. Using 0.5s.")
            interval = 0.5

        files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                        if Path(f).suffix.lower() in VID_EXTS])
        if len(files) < 2:
            self.log("Need at least 2 video files for comparison. "
                     "Use Single Video mode for one file.")
            return

        self.log(f"Found {len(files)} video(s). "
                 f"Running pairwise comparisons (SSIM + MSE + PSNR)...")
        analyzer    = VideoAnalyzer()
        pairs       = [(files[i], files[j])
                       for i in range(len(files))
                       for j in range(i + 1, len(files))]
        total_pairs = len(pairs)

        for pair_idx, (v1, v2) in enumerate(pairs):
            self.log(f"  [{pair_idx+1}/{total_pairs}] "
                     f"{os.path.basename(v1)} vs {os.path.basename(v2)}")

            def on_progress(cur, tot, lbl, _p=pair_idx):
                overall = (_p / total_pairs + (cur / max(tot, 1)) / total_pairs) * 100
                self.root.after(0, lambda pct=overall, l=lbl: (
                    self.progress_var.set(pct),
                    self.progress_label.config(text=l)
                ))

            analyzer.compare_videos(v1, v2, interval=interval,
                                    progress_cb=on_progress)

        self.log(f"Done. {len(analyzer.results)} frame comparison(s) recorded.")
        self.log("Opening results preview...")
        results_copy = list(analyzer.results)
        self.root.after(0, lambda: ResultsWindow(self.root, "Video",
                                                  video_results=results_copy,
                                                  video_paths=files))

    # Single video

    def _run_single_video_analysis(self, path: str):
        try:
            interval = float(self.interval_var.get())
        except ValueError:
            self.log("Invalid interval. Using 0.5s.")
            interval = 0.5

        self.log(f"Analyzing single video: {os.path.basename(path)}")
        sv = SingleVideoAnalyzer()
        sv.analyze(
            path, interval=interval,
            progress_cb=lambda cur, tot, lbl: self._set_progress(cur, tot, lbl),
        )
        self.log(f"   Duration: {sv.duration}s  |  FPS: {sv.fps:.1f}  |  "
                 f"Resolution: {sv.width}x{sv.height}  |  Codec: {sv.codec}")
        self.log(f"   Frames sampled: {len(sv.frames)}")
        self.log("Opening results preview...")
        self.root.after(0, lambda: ResultsWindow(self.root, "Single Video",
                                                  single_video=sv))

    # About dialog 

    def _show_about(self):
        popup = tk.Toplevel(self.root)
        popup.title("About Media Analyzer")
        popup.resizable(False, False)
        popup.grab_set()

        banner = tk.Frame(popup, bg="#0D2B4E")
        banner.pack(fill="x")
        tk.Label(banner, text="Media Analyzer", bg="#0D2B4E", fg="white",
                 font=("Arial", 18, "bold"), pady=12).pack()
        tk.Label(banner, text="Forensic Image and Video Analysis Tool",
                 bg="#0D2B4E", fg="#AACCEE", font=("Arial", 10)).pack(pady=(0, 10))

        info = tk.Frame(popup, padx=24, pady=16)
        info.pack()
        lines = [
            ("Developed by:",    "Jackson Allen"),
            ("AI runtime:",      "Ollama (Llama 3 + LLaVA)"),
            ("Hashing:",          "imagehash (pHash, dHash, aHash, bHash)"),
        ]
        for label, value in lines:
            row = tk.Frame(info)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=("Arial", 9, "bold"),
                     width=18, anchor="e").pack(side="left")
            tk.Label(row, text=value, font=("Arial", 9),
                     fg="#333", anchor="w").pack(side="left", padx=8)

        tk.Button(popup, text="Close", command=popup.destroy,
                  bg="#1A4F8A", fg="white", font=("Arial", 9, "bold"),
                  padx=16, pady=4).pack(pady=(0, 16))

        popup.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width()  - popup.winfo_width())  // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{x}+{y}")

    def run(self):
        self.root.mainloop()