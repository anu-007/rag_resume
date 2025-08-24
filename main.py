# main.py
import argparse, os, json
from src.pipeline import build_pipeline, generate_artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", nargs="+", required=True)
    ap.add_argument("--jd", required=True)
    ap.add_argument("--question", default="Create ATS resume, cover, email, and DM for this job.")
    ap.add_argument("--model", default="gemma3:4b")
    args = ap.parse_args()

    print(args)
    p = build_pipeline(args.resume, args.jd, model=args.model)
    out = generate_artifacts(p, args.question)

    os.makedirs("out", exist_ok=True)
    for k,v in out.items():
        if isinstance(v, str) and k != "docs_used":
            ext = ".md" if "resume" in k or "cover" in k else ".txt"
            with open(f"out/{k}{ext}","w",encoding="utf-8") as f: f.write(v)
    print(json.dumps({"docs_used": out["docs_used"]}, indent=2))

if __name__ == "__main__":
    main()
