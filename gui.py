"""
gui.py
Tkinter-based desktop GUI for the Resume Parser AI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import ResumeModel
from visualizer import plot_score_gauge, plot_metrics_bar, plot_category_distribution


# ── Colour palette ──────────────────────────────────────────────────────────
BG        = '#1e1e2e'
PANEL     = '#2a2a40'
ACCENT    = '#7c6af7'
TEXT      = '#cdd6f4'
SUBTEXT   = '#a6adc8'
SUCCESS   = '#2ecc71'
WARN      = '#f39c12'
DANGER    = '#e74c3c'
BTN_FG    = '#ffffff'
ENTRY_BG  = '#313244'


class ResumeParserGUI:
    """Main application window."""

    def __init__(self, root):
        self.root = root
        self.root.title("Resume Parser AI")
        self.root.geometry("1100x750")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.model = ResumeModel(n_neighbors=5)
        self.dataset_path = None
        self.metrics = {}
        self.category_counts = {}

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_status_bar()

    def _build_header(self):
        header = tk.Frame(self.root, bg=ACCENT, height=60)
        header.pack(fill='x')
        tk.Label(header, text="  Resume Parser AI",
                 font=('Segoe UI', 18, 'bold'),
                 bg=ACCENT, fg=BTN_FG).pack(side='left', padx=20, pady=12)
        tk.Label(header, text="Powered by KNN + TF-IDF | NLTK Preprocessing",
                 font=('Segoe UI', 10),
                 bg=ACCENT, fg='#ddd').pack(side='right', padx=20)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill='both', expand=True, padx=15, pady=10)

        # Left panel
        left = tk.Frame(body, bg=PANEL, width=380)
        left.pack(side='left', fill='y', padx=(0, 8))
        left.pack_propagate(False)
        self._build_left_panel(left)

        # Right panel (tabs)
        right = tk.Frame(body, bg=BG)
        right.pack(side='left', fill='both', expand=True)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        tk.Label(parent, text="Step 1 – Load Dataset",
                 font=('Segoe UI', 11, 'bold'),
                 bg=PANEL, fg=ACCENT).pack(anchor='w', padx=15, pady=(15, 5))

        tk.Label(parent,
                 text="Load the Kaggle Resume Dataset CSV\n(columns: Category, Resume)",
                 font=('Segoe UI', 9), bg=PANEL, fg=SUBTEXT,
                 justify='left').pack(anchor='w', padx=15)

        self.dataset_label = tk.Label(parent, text="No dataset loaded",
                                      font=('Segoe UI', 9, 'italic'),
                                      bg=PANEL, fg=WARN, wraplength=330)
        self.dataset_label.pack(anchor='w', padx=15, pady=4)

        self._btn(parent, "Browse & Load Dataset", self._load_dataset).pack(
            fill='x', padx=15, pady=4)

        tk.Frame(parent, bg='#3a3a5c', height=1).pack(fill='x', padx=15, pady=10)

        tk.Label(parent, text="Step 2 – Paste Resume",
                 font=('Segoe UI', 11, 'bold'),
                 bg=PANEL, fg=ACCENT).pack(anchor='w', padx=15)

        tk.Label(parent, text="Paste the candidate's resume text below:",
                 font=('Segoe UI', 9), bg=PANEL, fg=SUBTEXT).pack(anchor='w', padx=15, pady=3)

        self.resume_input = scrolledtext.ScrolledText(
            parent, height=14, font=('Consolas', 9),
            bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
            relief='flat', wrap='word')
        self.resume_input.pack(fill='x', padx=15, pady=4)

        self._btn(parent, "Analyse Resume", self._analyse_resume,
                  color=SUCCESS).pack(fill='x', padx=15, pady=4)
        self._btn(parent, "Clear", self._clear).pack(
            fill='x', padx=15, pady=2)

        tk.Frame(parent, bg='#3a3a5c', height=1).pack(fill='x', padx=15, pady=10)

        tk.Label(parent, text="Step 3 – View Results",
                 font=('Segoe UI', 11, 'bold'),
                 bg=PANEL, fg=ACCENT).pack(anchor='w', padx=15)
        tk.Label(parent,
                 text="Results and charts appear in the\ntabs on the right.",
                 font=('Segoe UI', 9), bg=PANEL, fg=SUBTEXT,
                 justify='left').pack(anchor='w', padx=15, pady=4)

    def _build_right_panel(self, parent):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=TEXT,
                        font=('Segoe UI', 10), padding=[12, 6])
        style.map('TNotebook.Tab',
                  background=[('selected', ACCENT)],
                  foreground=[('selected', BTN_FG)])

        self.notebook = ttk.Notebook(parent, style='TNotebook')
        self.notebook.pack(fill='both', expand=True)

        # Tab 1: Result
        self.tab_result = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.tab_result, text=' Result ')
        self._build_result_tab()

        # Tab 2: Score Chart
        self.tab_score = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.tab_score, text=' Score Chart ')

        # Tab 3: Model Metrics
        self.tab_metrics = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.tab_metrics, text=' Model Metrics ')

        # Tab 4: Dataset Info
        self.tab_dataset = tk.Frame(self.notebook, bg=BG)
        self.notebook.add(self.tab_dataset, text=' Dataset Info ')

    def _build_result_tab(self):
        self.result_frame = tk.Frame(self.tab_result, bg=BG)
        self.result_frame.pack(fill='both', expand=True, padx=20, pady=20)

        tk.Label(self.result_frame,
                 text="Load a dataset and paste a resume to see results.",
                 font=('Segoe UI', 12), bg=BG, fg=SUBTEXT).pack(expand=True)

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg='#12121e', height=30)
        bar.pack(fill='x', side='bottom')
        self.status_var = tk.StringVar(value="Ready. Please load a dataset first.")
        tk.Label(bar, textvariable=self.status_var,
                 font=('Segoe UI', 9), bg='#12121e', fg=SUBTEXT).pack(
            side='left', padx=10, pady=5)

    # ── Helper widgets ───────────────────────────────────────────────────────

    def _btn(self, parent, text, command, color=ACCENT):
        return tk.Button(parent, text=text, command=command,
                         bg=color, fg=BTN_FG,
                         font=('Segoe UI', 10, 'bold'),
                         relief='flat', cursor='hand2',
                         activebackground=PANEL, activeforeground=TEXT)

    # ── Actions ──────────────────────────────────────────────────────────────

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _load_dataset(self):
        path = filedialog.askopenfilename(
            title="Select Resume Dataset CSV",
            filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        self.dataset_path = path
        self._set_status("Training model… please wait.")
        self.dataset_label.config(text="Training… ⏳", fg=WARN)

        def train():
            try:
                X_test, y_test = self.model.load_and_train(path)
                self.metrics = self.model.evaluate(X_test, y_test)

                # Category distribution
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                # Find correct category column (handles hidden characters too)
                cat_col = None
                for col in df.columns:
                    if 'job_position' in col.lower() or col == 'Category':
                        cat_col = col
                        break
                if cat_col is None:
                    cat_col = df.columns[0]
                counts = df[cat_col].value_counts().to_dict()
                self.category_counts = counts

                self.root.after(0, self._on_training_done, path)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.dataset_label.config(
                    text="Failed to load dataset.", fg=DANGER))

        threading.Thread(target=train, daemon=True).start()

    def _on_training_done(self, path):
        fname = path.split('/')[-1]
        self.dataset_label.config(text=f"✓ Loaded: {fname}", fg=SUCCESS)
        self._set_status(
            f"Model trained. "
            f"Accuracy: {self.metrics['accuracy']*100:.1f}%  |  "
            f"F1: {self.metrics['f1']*100:.1f}%")

        self._render_metrics_chart()
        self._render_dataset_chart()

    def _analyse_resume(self):
        if not self.model.is_trained:
            messagebox.showwarning("No Model", "Please load and train on a dataset first.")
            return

        resume_text = self.resume_input.get("1.0", "end").strip()
        if len(resume_text) < 50:
            messagebox.showwarning("Input Too Short",
                                   "Please paste a resume with at least 50 characters.")
            return

        self._set_status("Analysing resume…")

        def analyse():
            category = self.model.predict_category(resume_text)
            score = self.model.score_resume(resume_text)
            self.root.after(0, self._display_result, category, score, resume_text)

        threading.Thread(target=analyse, daemon=True).start()

    def _display_result(self, category, score, resume_text):
        # ── Result tab ──
        for w in self.result_frame.winfo_children():
            w.destroy()

        # Score colour
        if score >= 7:
            score_color = SUCCESS
            verdict = "Strong Candidate ✓"
        elif score >= 4:
            score_color = WARN
            verdict = "Moderate Candidate ~"
        else:
            score_color = DANGER
            verdict = "Weak Candidate ✗"

        tk.Label(self.result_frame, text="Analysis Result",
                 font=('Segoe UI', 15, 'bold'), bg=BG, fg=TEXT).pack(pady=(10, 5))

        info_frame = tk.Frame(self.result_frame, bg=PANEL, padx=20, pady=15)
        info_frame.pack(fill='x', pady=8)

        rows = [
            ("Predicted Category", category, ACCENT),
            ("Score (0–10)", f"{score} / 10", score_color),
            ("Verdict", verdict, score_color),
        ]
        for label, value, color in rows:
            row = tk.Frame(info_frame, bg=PANEL)
            row.pack(fill='x', pady=3)
            tk.Label(row, text=f"{label}:", font=('Segoe UI', 11),
                     bg=PANEL, fg=SUBTEXT, width=22, anchor='w').pack(side='left')
            tk.Label(row, text=value, font=('Segoe UI', 11, 'bold'),
                     bg=PANEL, fg=color).pack(side='left')

        # Word count
        word_count = len(resume_text.split())
        tk.Label(self.result_frame,
                 text=f"Resume word count: {word_count}",
                 font=('Segoe UI', 9), bg=BG, fg=SUBTEXT).pack(pady=5)

        # Tip
        if score < 4:
            tip = "Tip: Resume seems sparse. Add more relevant skills and experience."
        elif score < 7:
            tip = "Tip: Moderate match. Highlight domain-specific keywords more clearly."
        else:
            tip = "Tip: Strong match! Resume is well-aligned with the predicted category."

        tk.Label(self.result_frame, text=tip, wraplength=580,
                 font=('Segoe UI', 10, 'italic'), bg=BG, fg=SUBTEXT).pack(pady=6)

        # ── Score Chart tab ──
        for w in self.tab_score.winfo_children():
            w.destroy()

        fig = plot_score_gauge(score, category)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_score)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=30)

        self.notebook.select(self.tab_result)
        self._set_status(
            f"Done. Category: {category} | Score: {score}/10 | {verdict}")

    def _render_metrics_chart(self):
        for w in self.tab_metrics.winfo_children():
            w.destroy()

        fig = plot_metrics_bar(self.metrics)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_metrics)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)

        # Numeric summary below chart
        summary = tk.Frame(self.tab_metrics, bg=BG)
        summary.pack(pady=5)
        for name, val in self.metrics.items():
            tk.Label(summary, text=f"{name.capitalize()}: {val*100:.1f}%",
                     font=('Segoe UI', 10), bg=BG, fg=TEXT).pack(side='left', padx=15)

    def _render_dataset_chart(self):
        for w in self.tab_dataset.winfo_children():
            w.destroy()

        # Top 15 categories for readability
        top = dict(sorted(self.category_counts.items(),
                           key=lambda x: x[1], reverse=True)[:15])

        fig = plot_category_distribution(top)
        canvas = FigureCanvasTkAgg(fig, master=self.tab_dataset)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)

        tk.Label(self.tab_dataset,
                 text=f"Total categories: {len(self.category_counts)} | "
                      f"Total resumes: {sum(self.category_counts.values())}",
                 font=('Segoe UI', 9), bg=BG, fg=SUBTEXT).pack(pady=3)

    def _clear(self):
        self.resume_input.delete("1.0", "end")
        for w in self.result_frame.winfo_children():
            w.destroy()
        tk.Label(self.result_frame,
                 text="Load a dataset and paste a resume to see results.",
                 font=('Segoe UI', 12), bg=BG, fg=SUBTEXT).pack(expand=True)
        for w in self.tab_score.winfo_children():
            w.destroy()
        self._set_status("Cleared. Ready for new input.")