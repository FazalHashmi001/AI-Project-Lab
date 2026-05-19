"""
gui.py
Resume Parser AI - Enhanced GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import ResumeModel
from visualizer import plot_score_gauge, plot_metrics_bar, plot_category_distribution

# ── Colour palette ───────────────────────────────────────────────────────────
BG        = '#0f0f1a'
PANEL     = '#1a1a2e'
CARD      = '#16213e'
ACCENT    = '#6c63ff'
ACCENT2   = '#a78bfa'
TEXT      = '#e2e8f0'
SUBTEXT   = '#94a3b8'
SUCCESS   = '#10b981'
WARN      = '#f59e0b'
DANGER    = '#ef4444'
BTN_FG    = '#ffffff'
ENTRY_BG  = '#1e293b'
BORDER    = '#2d2d4e'
HIGHLIGHT = '#7c3aed'


def _hoverable(btn, normal_bg, hover_bg):
    btn.bind('<Enter>', lambda e: btn.config(bg=hover_bg))
    btn.bind('<Leave>', lambda e: btn.config(bg=normal_bg))


class ResumeParserGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Resume Parser AI  •  KNN + TF-IDF")
        self.root.geometry("1280x800")
        self.root.minsize(1000, 680)
        self.root.configure(bg=BG)

        self.model = ResumeModel(n_neighbors=5)
        self.dataset_path = None
        self.metrics = {}
        self.category_counts = {}
        self._anim_after = None

        self._apply_styles()
        self._build_ui()

    # ── Styles ───────────────────────────────────────────────────────────────

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=SUBTEXT,
                        font=('Segoe UI', 10, 'bold'), padding=[18, 8])
        style.map('TNotebook.Tab',
                  background=[('selected', ACCENT)],
                  foreground=[('selected', BTN_FG)])
        style.configure('Vertical.TScrollbar', background=PANEL,
                        troughcolor=BG, bordercolor=BG, arrowcolor=SUBTEXT)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_titlebar()
        self._build_body()
        self._build_statusbar()

    def _build_titlebar(self):
        bar = tk.Frame(self.root, bg=ACCENT, height=56)
        bar.pack(fill='x')
        bar.pack_propagate(False)

        # Left: icon + title
        left = tk.Frame(bar, bg=ACCENT)
        left.pack(side='left', padx=18, pady=8)

        tk.Label(left, text="⬡", font=('Segoe UI', 22, 'bold'),
                 bg=ACCENT, fg='#c4b5fd').pack(side='left', padx=(0, 10))

        title_col = tk.Frame(left, bg=ACCENT)
        title_col.pack(side='left')
        tk.Label(title_col, text="Resume Parser AI",
                 font=('Segoe UI', 15, 'bold'),
                 bg=ACCENT, fg=BTN_FG).pack(anchor='w')
        tk.Label(title_col, text="Intelligent Resume Analysis & Category Detection",
                 font=('Segoe UI', 8),
                 bg=ACCENT, fg='#c4b5fd').pack(anchor='w')

        # Right: badges
        right = tk.Frame(bar, bg=ACCENT)
        right.pack(side='right', padx=18)
        for badge_text, badge_bg in [("KNN", '#4f46e5'), ("TF-IDF", '#7c3aed'), ("NLTK", '#6d28d9')]:
            tk.Label(right, text=badge_text,
                     font=('Segoe UI', 8, 'bold'),
                     bg=badge_bg, fg=BTN_FG,
                     padx=8, pady=3).pack(side='left', padx=3)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=0, pady=0)

        # Sidebar
        sidebar = tk.Frame(body, bg=PANEL, width=320)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Divider
        tk.Frame(body, bg=BORDER, width=1).pack(side='left', fill='y')

        # Main content
        main = tk.Frame(body, bg=BG)
        main.pack(side='left', fill='both', expand=True)
        self._build_main(main)

    def _build_sidebar(self, parent):
        # Scrollable sidebar
        canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        inner = tk.Frame(canvas, bg=PANEL)
        win = canvas.create_window((0, 0), window=inner, anchor='nw')

        def _resize(e):
            canvas.configure(scrollregion=canvas.bbox('all'))
            canvas.itemconfig(win, width=e.width)
        inner.bind('<Configure>', _resize)
        canvas.bind('<Configure>', lambda e: canvas.itemconfig(win, width=e.width))

        self._build_sidebar_content(inner)

    def _build_sidebar_content(self, parent):
        pad = {'padx': 18}

        # ── Dataset Section ──────────────────────────────────────────────────
        self._section_header(parent, "01", "Load Dataset")

        tk.Label(parent, text="Select your resume dataset CSV file\nto train the model.",
                 font=('Segoe UI', 9), bg=PANEL, fg=SUBTEXT, justify='left'
                 ).pack(anchor='w', **pad, pady=(0, 6))

        self.dataset_label = tk.Label(parent, text="⚠  No dataset loaded",
                                      font=('Segoe UI', 9),
                                      bg=PANEL, fg=WARN, wraplength=270, justify='left')
        self.dataset_label.pack(anchor='w', **pad, pady=(0, 8))

        # Progress bar
        self.train_progress = ttk.Progressbar(parent, mode='indeterminate', length=270)
        self.train_progress.pack(**pad, pady=(0, 6))

        self._pill_btn(parent, "📂  Browse & Load Dataset",
                       self._load_dataset, ACCENT).pack(fill='x', **pad, pady=(0, 4))

        # Stats row
        self.stats_frame = tk.Frame(parent, bg=PANEL)
        self.stats_frame.pack(fill='x', **pad, pady=(4, 0))
        self._stat_box(self.stats_frame, "Categories", "—", 'cat_val')
        self._stat_box(self.stats_frame, "Resumes", "—", 'res_val')
        self.stats_frame.pack_forget()  # hidden until trained

        self._divider(parent)

        # ── Resume Section ───────────────────────────────────────────────────
        self._section_header(parent, "02", "Input Resume")

        tk.Label(parent, text="Paste candidate resume text:",
                 font=('Segoe UI', 9), bg=PANEL, fg=SUBTEXT
                 ).pack(anchor='w', **pad, pady=(0, 4))

        input_frame = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        input_frame.pack(fill='x', **pad, pady=(0, 6))

        self.resume_input = scrolledtext.ScrolledText(
            input_frame, height=12, font=('Consolas', 9),
            bg=ENTRY_BG, fg=TEXT, insertbackground=ACCENT2,
            relief='flat', wrap='word', padx=8, pady=8)
        self.resume_input.pack(fill='x')

        # Word count live
        self.wc_var = tk.StringVar(value="0 words")
        tk.Label(parent, textvariable=self.wc_var,
                 font=('Segoe UI', 8), bg=PANEL, fg=SUBTEXT
                 ).pack(anchor='e', padx=18, pady=(0, 4))
        self.resume_input.bind('<KeyRelease>', self._update_wordcount)

        self._pill_btn(parent, "🔍  Analyse Resume",
                       self._analyse_resume, SUCCESS).pack(fill='x', **pad, pady=(0, 4))
        self._pill_btn(parent, "✕  Clear",
                       self._clear, '#334155').pack(fill='x', **pad, pady=(0, 4))

        self._divider(parent)

        # ── Quick Tips ───────────────────────────────────────────────────────
        self._section_header(parent, "03", "Tips")
        tips = [
            "✦  More keywords = higher score",
            "✦  Include skills section clearly",
            "✦  Mention domain tools & tech",
            "✦  500+ word resumes score better",
        ]
        for t in tips:
            tk.Label(parent, text=t, font=('Segoe UI', 8),
                     bg=PANEL, fg=SUBTEXT, justify='left'
                     ).pack(anchor='w', padx=18, pady=1)

        tk.Frame(parent, bg=PANEL, height=20).pack()

    def _build_main(self, parent):
        self.notebook = ttk.Notebook(parent, style='TNotebook')
        self.notebook.pack(fill='both', expand=True)

        self.tab_result  = tk.Frame(self.notebook, bg=BG)
        self.tab_score   = tk.Frame(self.notebook, bg=BG)
        self.tab_metrics = tk.Frame(self.notebook, bg=BG)
        self.tab_dataset = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.tab_result,  text='  🎯  Result  ')
        self.notebook.add(self.tab_score,   text='  📊  Score Chart  ')
        self.notebook.add(self.tab_metrics, text='  📈  Model Metrics  ')
        self.notebook.add(self.tab_dataset, text='  🗂  Dataset Info  ')

        self._build_result_placeholder()

    def _build_result_placeholder(self):
        for w in self.tab_result.winfo_children():
            w.destroy()

        outer = tk.Frame(self.tab_result, bg=BG)
        outer.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(outer, text="⬡", font=('Segoe UI', 48),
                 bg=BG, fg=BORDER).pack()
        tk.Label(outer, text="No Analysis Yet",
                 font=('Segoe UI', 16, 'bold'),
                 bg=BG, fg=SUBTEXT).pack(pady=(8, 4))
        tk.Label(outer, text="Load a dataset and paste a resume\nto see the analysis result here.",
                 font=('Segoe UI', 10),
                 bg=BG, fg=BORDER, justify='center').pack()

    def _build_statusbar(self):
        bar = tk.Frame(self.root, bg='#0a0a14', height=28)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)

        self.status_var = tk.StringVar(value="●  Ready — load a dataset to begin")
        tk.Label(bar, textvariable=self.status_var,
                 font=('Segoe UI', 8), bg='#0a0a14', fg=SUBTEXT
                 ).pack(side='left', padx=14, pady=6)

        tk.Label(bar, text="Resume Parser AI  v2.0",
                 font=('Segoe UI', 8), bg='#0a0a14', fg=BORDER
                 ).pack(side='right', padx=14)

    # ── Widget helpers ────────────────────────────────────────────────────────

    def _section_header(self, parent, num, title):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill='x', padx=18, pady=(16, 6))
        tk.Label(row, text=num, font=('Segoe UI', 7, 'bold'),
                 bg=ACCENT, fg=BTN_FG, padx=5, pady=1).pack(side='left')
        tk.Label(row, text=f"  {title}", font=('Segoe UI', 10, 'bold'),
                 bg=PANEL, fg=TEXT).pack(side='left')

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=18, pady=10)

    def _pill_btn(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, command=command,
                        bg=color, fg=BTN_FG,
                        font=('Segoe UI', 9, 'bold'),
                        relief='flat', cursor='hand2',
                        padx=12, pady=8,
                        activebackground=HIGHLIGHT,
                        activeforeground=BTN_FG)
        _hoverable(btn, color, HIGHLIGHT)
        return btn

    def _stat_box(self, parent, label, value, attr):
        box = tk.Frame(parent, bg=CARD, padx=10, pady=8)
        box.pack(side='left', expand=True, fill='x', padx=(0, 6))
        lbl = tk.Label(box, text=value, font=('Segoe UI', 16, 'bold'),
                       bg=CARD, fg=ACCENT2)
        lbl.pack()
        tk.Label(box, text=label, font=('Segoe UI', 8),
                 bg=CARD, fg=SUBTEXT).pack()
        setattr(self, attr, lbl)

    def _card(self, parent, **kwargs):
        return tk.Frame(parent, bg=CARD, **kwargs)

    # ── Live word count ───────────────────────────────────────────────────────

    def _update_wordcount(self, event=None):
        text = self.resume_input.get("1.0", "end").strip()
        count = len(text.split()) if text else 0
        self.wc_var.set(f"{count} words")

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg):
        self.status_var.set(f"●  {msg}")
        self.root.update_idletasks()

    # ── Load & Train ──────────────────────────────────────────────────────────

    def _load_dataset(self):
        path = filedialog.askopenfilename(
            title="Select Resume Dataset CSV",
            filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        self.dataset_path = path
        self._set_status("Training model — please wait…")
        self.dataset_label.config(text="⏳  Training in progress…", fg=WARN)
        self.train_progress.pack(padx=18, pady=(0, 6))
        self.train_progress.start(12)

        def train():
            try:
                X_test, y_test = self.model.load_and_train(path)
                self.metrics = self.model.evaluate(X_test, y_test)

                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                cat_col = next(
                    (c for c in df.columns
                     if 'job_position' in c.lower() or c == 'Category'),
                    df.columns[0])
                self.category_counts = df[cat_col].value_counts().to_dict()
                self.root.after(0, self._on_training_done, path)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
                self.root.after(0, lambda: self.dataset_label.config(
                    text="✗  Failed to load dataset.", fg=DANGER))
                self.root.after(0, self.train_progress.stop)

        threading.Thread(target=train, daemon=True).start()

    def _on_training_done(self, path):
        self.train_progress.stop()
        self.train_progress.pack_forget()

        fname = path.split('/')[-1].split('\\')[-1]
        self.dataset_label.config(text=f"✓  {fname}", fg=SUCCESS)

        # Update stat boxes
        self.stats_frame.pack(fill='x', padx=18, pady=(4, 0))
        self.cat_val.config(text=str(len(self.category_counts)))
        self.res_val.config(text=str(sum(self.category_counts.values())))

        acc = self.metrics['accuracy'] * 100
        f1  = self.metrics['f1'] * 100
        self._set_status(f"Model ready  ·  Accuracy: {acc:.1f}%  ·  F1: {f1:.1f}%")

        self._render_metrics_tab()
        self._render_dataset_tab()

    # ── Analyse ───────────────────────────────────────────────────────────────

    def _analyse_resume(self):
        if not self.model.is_trained:
            messagebox.showwarning("Model Not Ready", "Please load a dataset first.")
            return
        resume_text = self.resume_input.get("1.0", "end").strip()
        if len(resume_text) < 50:
            messagebox.showwarning("Too Short", "Paste a resume with at least 50 characters.")
            return
        self._set_status("Analysing resume…")

        def run():
            category = self.model.predict_category(resume_text)
            score    = self.model.score_resume(resume_text)
            self.root.after(0, self._display_result, category, score, resume_text)

        threading.Thread(target=run, daemon=True).start()

    def _display_result(self, category, score, resume_text):
        # ── Colour & verdict ──
        if score >= 7:
            score_color, verdict, icon = SUCCESS, "Strong Candidate", "✦"
        elif score >= 4:
            score_color, verdict, icon = WARN,    "Moderate Candidate", "◈"
        else:
            score_color, verdict, icon = DANGER,  "Weak Candidate", "✗"

        # ── Result Tab ───────────────────────────────────────────────────────
        for w in self.tab_result.winfo_children():
            w.destroy()

        scroll_canvas = tk.Canvas(self.tab_result, bg=BG, highlightthickness=0)
        vbar = ttk.Scrollbar(self.tab_result, orient='vertical',
                             command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side='right', fill='y')
        scroll_canvas.pack(side='left', fill='both', expand=True)

        content = tk.Frame(scroll_canvas, bg=BG)
        cwin = scroll_canvas.create_window((0, 0), window=content, anchor='nw')

        def _on_resize(e):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all'))
            scroll_canvas.itemconfig(cwin, width=e.width)
        content.bind('<Configure>', _on_resize)
        scroll_canvas.bind('<Configure>',
                           lambda e: scroll_canvas.itemconfig(cwin, width=e.width))

        pad = {'padx': 30}

        # Hero score banner
        banner = tk.Frame(content, bg=score_color, pady=22)
        banner.pack(fill='x', **pad, pady=(24, 0))
        tk.Label(banner, text=f"{icon}  {score} / 10",
                 font=('Segoe UI', 36, 'bold'),
                 bg=score_color, fg=BTN_FG).pack()
        tk.Label(banner, text=verdict,
                 font=('Segoe UI', 12),
                 bg=score_color, fg=BTN_FG).pack()

        # Details card
        details = self._card(content, padx=24, pady=16)
        details.pack(fill='x', **pad, pady=(12, 0))

        rows = [
            ("🏷  Predicted Category", category, ACCENT2),
            ("📊  Confidence Score",   f"{score}/10", score_color),
            ("✦   Verdict",            verdict, score_color),
            ("📝  Word Count",         f"{len(resume_text.split())} words", TEXT),
        ]
        for i, (lbl, val, col) in enumerate(rows):
            row = tk.Frame(details, bg=CARD)
            row.pack(fill='x', pady=4)
            if i < len(rows) - 1:
                tk.Frame(details, bg=BORDER, height=1).pack(fill='x')
            tk.Label(row, text=lbl, font=('Segoe UI', 10),
                     bg=CARD, fg=SUBTEXT, width=24, anchor='w').pack(side='left')
            tk.Label(row, text=val, font=('Segoe UI', 10, 'bold'),
                     bg=CARD, fg=col).pack(side='left')

        # Score bar visual
        bar_card = self._card(content, padx=24, pady=16)
        bar_card.pack(fill='x', **pad, pady=(12, 0))
        tk.Label(bar_card, text="Score Breakdown",
                 font=('Segoe UI', 10, 'bold'),
                 bg=CARD, fg=TEXT).pack(anchor='w', pady=(0, 10))

        bar_bg = tk.Frame(bar_card, bg=BORDER, height=18)
        bar_bg.pack(fill='x')
        bar_fill = tk.Frame(bar_bg, bg=score_color, height=18)
        bar_fill.place(relwidth=score / 10, relheight=1)

        marks = tk.Frame(bar_card, bg=CARD)
        marks.pack(fill='x')
        for val, label in [(0, "0"), (4, "4"), (7, "7"), (10, "10")]:
            tk.Label(marks, text=label, font=('Segoe UI', 7),
                     bg=CARD, fg=SUBTEXT).place(
                relx=val / 10, rely=0, anchor='n')
        tk.Frame(bar_card, bg=CARD, height=14).pack()

        # Tip card
        if score < 4:
            tip = "Resume appears sparse. Add more relevant technical skills, project details, and domain keywords."
        elif score < 7:
            tip = "Moderate match detected. Strengthen your skills section with domain-specific tools and technologies."
        else:
            tip = "Excellent match! Your resume is well-aligned with the predicted category. Keep it focused."

        tip_card = self._card(content, padx=24, pady=14)
        tip_card.pack(fill='x', **pad, pady=(12, 24))
        tk.Label(tip_card, text="💡  Suggestion",
                 font=('Segoe UI', 9, 'bold'),
                 bg=CARD, fg=ACCENT2).pack(anchor='w')
        tk.Label(tip_card, text=tip, wraplength=650,
                 font=('Segoe UI', 9),
                 bg=CARD, fg=TEXT, justify='left').pack(anchor='w', pady=(4, 0))

        # ── Score Chart Tab ──────────────────────────────────────────────────
        for w in self.tab_score.winfo_children():
            w.destroy()

        fig = plot_score_gauge(score, category)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_score)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=30, pady=40)

        self.notebook.select(self.tab_result)
        self._set_status(
            f"Analysis complete  ·  {category}  ·  Score: {score}/10  ·  {verdict}")

    # ── Metrics Tab ───────────────────────────────────────────────────────────

    def _render_metrics_tab(self):
        for w in self.tab_metrics.winfo_children():
            w.destroy()

        # Metric stat cards at the top
        cards_row = tk.Frame(self.tab_metrics, bg=BG)
        cards_row.pack(fill='x', padx=30, pady=(24, 12))

        icons = {'accuracy': '🎯', 'precision': '📌', 'recall': '🔁', 'f1': '⚡'}
        for name, val in self.metrics.items():
            card = tk.Frame(cards_row, bg=CARD, padx=16, pady=14)
            card.pack(side='left', expand=True, fill='x', padx=6)
            tk.Label(card, text=icons.get(name, ''), font=('Segoe UI', 18),
                     bg=CARD, fg=ACCENT2).pack()
            tk.Label(card, text=f"{val*100:.1f}%",
                     font=('Segoe UI', 20, 'bold'),
                     bg=CARD, fg=SUCCESS if val >= 0.7 else WARN).pack()
            tk.Label(card, text=name.capitalize(),
                     font=('Segoe UI', 9),
                     bg=CARD, fg=SUBTEXT).pack()

        # Chart
        fig = plot_metrics_bar(self.metrics)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_metrics)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=30, pady=(0, 20))

    # ── Dataset Tab ───────────────────────────────────────────────────────────

    def _render_dataset_tab(self):
        for w in self.tab_dataset.winfo_children():
            w.destroy()

        # Summary bar
        summ = tk.Frame(self.tab_dataset, bg=CARD, pady=12)
        summ.pack(fill='x', padx=30, pady=(20, 0))
        total_cats = len(self.category_counts)
        total_res  = sum(self.category_counts.values())
        for label, val in [("Total Categories", total_cats),
                           ("Total Resumes", total_res),
                           ("Avg per Category", f"{total_res // max(total_cats,1)}")]:
            col = tk.Frame(summ, bg=CARD)
            col.pack(side='left', expand=True)
            tk.Label(col, text=str(val), font=('Segoe UI', 18, 'bold'),
                     bg=CARD, fg=ACCENT2).pack()
            tk.Label(col, text=label, font=('Segoe UI', 8),
                     bg=CARD, fg=SUBTEXT).pack()

        top = dict(sorted(self.category_counts.items(),
                          key=lambda x: x[1], reverse=True)[:15])
        fig = plot_category_distribution(top)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_dataset)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=30, pady=(12, 20))

    # ── Clear ─────────────────────────────────────────────────────────────────

    def _clear(self):
        self.resume_input.delete("1.0", "end")
        self.wc_var.set("0 words")
        self._build_result_placeholder()
        for w in self.tab_score.winfo_children():
            w.destroy()
        self._set_status("Cleared — ready for new input")
