import tkinter as tk
from tkinter import scrolledtext, messagebox
import json

# Import your analyzer logic
from faiss_index_extractor import CMSDenialAnalyzer

analyzer = CMSDenialAnalyzer()

def analyze():
    try:
        raw_input = input_box.get("1.0", tk.END).strip()
        claim_data = json.loads(raw_input)
        result = analyzer.analyze_claim(claim_data)

        # If the result is a string (formatted LLM output), show as-is
        if isinstance(result, str):
            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, result)
        else:
            # Pretty-print the result dictionary
            formatted = json.dumps(result, indent=2)
            output_box.delete("1.0", tk.END)
            output_box.insert(tk.END, formatted)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyze claim:\n{e}")

# Setup GUI
root = tk.Tk()
root.title("CMS Denial Analyzer")
root.geometry("800x600")

tk.Label(root, text="Paste Claim JSON:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
input_box = scrolledtext.ScrolledText(root, width=90, height=10, font=("Consolas", 10))
input_box.pack(padx=10, pady=5)
input_box.insert(tk.END, '{\n  "cpt_code": "99213",\n  "diagnosis": "E11.9",\n  "modifiers": ["25"],\n  "payer": "Medicare"\n}')

tk.Button(root, text="Analyze", command=analyze, bg="#007acc", fg="white", font=("Arial", 10, "bold")).pack(pady=10)

tk.Label(root, text="Analysis Output:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
output_box = scrolledtext.ScrolledText(root, width=90, height=20, font=("Consolas", 10), bg="#f4f4f4")
output_box.pack(padx=10, pady=5)

root.mainloop()
