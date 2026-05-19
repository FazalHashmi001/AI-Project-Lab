"""
Resume Parser AI - Main Entry Point
Run this file to launch the application.e"""
 
from gui import ResumeParserGUI
import tkinter as tk
 
if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeParserGUI(root)
    root.mainloop()