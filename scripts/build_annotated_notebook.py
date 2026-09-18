"""Build an annotated copy of the results notebook: identical content + an interview-prep
"why" explanation cell appended at the end of every top-level section.

    python scripts/build_annotated_notebook.py [explanations.json]

explanations.json: [{"id": <section number>, "explanation": "<markdown>"}, ...]
Output: notebooks/RESULTS_annotated.ipynb
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "RESULTS.ipynb"
EXPL = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "notebook_explanations.json"
OUT = ROOT / "notebooks" / "RESULTS_annotated.ipynb"


def sec_num(cell):
    if cell["cell_type"] != "markdown":
        return None
    for line in "".join(cell["source"]).split("\n"):
        m = re.match(r"##\s+(\d+)\.", line.strip())
        if m:
            return int(m.group(1))
    return None


def md_cell(text, cid):
    body = "---\n\n" + text.strip() + "\n"
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": [l + "\n" for l in body.split("\n")]}


def main():
    nb = json.load(open(SRC))
    expl = {int(e["id"]): e["explanation"] for e in json.load(open(EXPL)) if e.get("explanation")}

    out_cells, cur = [], None
    for c in nb["cells"]:
        n = sec_num(c)
        if n is not None and cur is not None and cur in expl:     # end of previous section
            out_cells.append(md_cell(expl[cur], f"why-{cur}"))
        if n is not None:
            cur = n
        out_cells.append(c)
    if cur is not None and cur in expl:                            # last section
        out_cells.append(md_cell(expl[cur], f"why-{cur}"))

    for i, c in enumerate(out_cells):                              # every cell needs an id (nbformat 4.5)
        c.setdefault("id", f"cell-{i}")

    nb["cells"] = out_cells
    json.dump(nb, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT}: {len(nb['cells'])} cells "
          f"(+{len(expl)} explanation sections inserted)")


if __name__ == "__main__":
    main()
