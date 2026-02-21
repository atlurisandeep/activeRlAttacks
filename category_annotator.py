"""One-time script to annotate sft_dataset.json with S1-S8 harm categories.

Usage:
    python3 category_annotator.py \
        --input prompts/sft_dataset.json \
        --output prompts/sft_dataset_categorized.json
"""
import argparse
import json
import re

# S1-S8 category definitions with keyword patterns
CATEGORIES = {
    "S1": {
        "name": "Violence/Physical Harm",
        "keywords": [
            r"\bbomb\b", r"\bexplos", r"\bweapon", r"\bgun\b", r"\bkill\b",
            r"\bmurder", r"\bassault\b", r"\bshoot", r"\bstab\b",
            r"\bpoison", r"\btortur", r"\bviolent", r"\bviolence",
            r"\bthreat", r"\bterror", r"\barson", r"\bkidnap", r"\babduct",
            r"\bpunch", r"\binjur", r"\bdestroy\b", r"\bEMP\b",
            r"\belectromagnetic pulse", r"\bdisrupt.*electronic",
            r"\bambush", r"\bbreak.*neck", r"\bmace\b",
            r"\bassassin", r"\bsniper", r"\bbludgeon", r"\bstrangle",
            r"\bchoke\b", r"\bdrown\b", r"\barmed\b", r"\bammunit",
            r"\bfirearm", r"\brifle\b", r"\bshotgun", r"\bgrenade",
            r"\blandmine", r"\bnapalm", r"\bmolotov",
        ],
    },
    "S2": {
        "name": "Non-Violent Crimes/Deception",
        "keywords": [
            r"\bhack\b", r"\bhacking", r"\bsteal\b", r"\btheft", r"\bfraud",
            r"\bscam\b", r"\bphish", r"\bforge", r"\bcounterfeit", r"\bsmuggl",
            r"\blaunder", r"\bbribe", r"\bcorrupt", r"\bextort", r"\bblackmail",
            r"\bburgl", r"\bshoplift", r"\bmalware", r"\bvirus\b",
            r"\btrojan", r"\bransomware", r"\bexploit.*vulnerab",
            r"\bidentity.*theft", r"\bcyber.*crime", r"\billegal\b",
            r"\bembezzl", r"\btake.*money.*without", r"\bcheat\b",
            r"\bmanipulat", r"\bdeceiv", r"\bdeception", r"\btrick\b",
            r"\bcon\b.*people", r"\bscheme\b", r"\bpyramid.*scheme",
            r"\bponzi", r"\bskip.*school", r"\bfake.*id",
            r"\bfake.*document", r"\bdrug.*sell", r"\bdrug.*deal",
            r"\bdrug.*market", r"\bblack.*market", r"\bmoonshine",
            r"\billicit", r"\bprohibit", r"\bcontraband",
            r"\bbreach", r"\bintrusion", r"\bbreak.*into",
            r"\bpick.*lock", r"\bdisable.*alarm", r"\bevade.*police",
            r"\bescape.*prison", r"\btrespass",
        ],
    },
    "S3": {
        "name": "Sex-Related Content",
        "keywords": [
            r"\bsexual", r"\brape\b", r"\bmolest", r"\bsex.*traffick",
            r"\bprostitut", r"\bpornograph", r"\bnon.?consensual",
            r"\bsext", r"\brevenge.*porn", r"\bnude\b", r"\bnaked\b",
            r"\bvoyeur", r"\bindecen", r"\berotic", r"\bfetish",
            r"\bintimate.*without.*consent", r"\bseduc",
            r"\babortion",
        ],
    },
    "S4": {
        "name": "Child Safety",
        "keywords": [
            r"\bchild.*abus", r"\bchild.*exploit", r"\bchild.*porn",
            r"\bpedophil", r"\bminor.*sexual", r"\bchild.*traffick",
            r"\bunderage", r"\byoung.*girl", r"\byoung.*boy",
            r"\bchild.*naked", r"\bchild.*nude", r"\bCSAM\b",
            r"\bchild.*groom", r"\bkid.*inappropriat",
        ],
    },
    "S5": {
        "name": "Defamation/Hateful Content",
        "keywords": [
            r"\bdefam", r"\bslander", r"\blibel", r"\bhate\b",
            r"\bracis[mt]", r"\bdiscriminat", r"\bbigot", r"\bslur\b",
            r"\bxenophob", r"\bhomophob", r"\btransphob", r"\bsexis[mt]",
            r"\bmisogyn", r"\bwhite.*suprem", r"\bnazi", r"\bstereo",
            r"\binsult", r"\bhateful", r"\bprejudic",
            r"\bthugs?\b", r"\bbad.*company", r"\bbad.*person",
            r"\bjew", r"\bmuslim.*bad", r"\bblack.*people",
            r"\bwomen.*inferior", r"\bmen.*superior", r"\bwhy.*stupid",
            r"\bholocaust", r"\bgenocid", r"\bethnic.*clean",
            r"\bimmigrant.*bad", r"\billegal.*resident",
            r"\brunning.*joke", r"\bmake.*fun", r"\bbully",
            r"\bridicul", r"\bmock\b", r"\bdegrad",
        ],
    },
    "S6": {
        "name": "Specialized/Dangerous Advice",
        "keywords": [
            r"\bmedical.*advice", r"\blegal.*advice", r"\bfinancial.*advice",
            r"\bdiagnos", r"\bprescri", r"\btreat.*disease",
            r"\btax.*evasi", r"\bdrug.*dos", r"\bself.*medicat",
            r"\bsuicid", r"\bself.*harm", r"\banorex", r"\bbulimi",
            r"\beating.*disorder", r"\boverdos",
            r"\bingrown", r"\bhome.*remed", r"\bhome.*treat",
            r"\bDIY.*surg", r"\bprank\b", r"\bdangerous.*challenge",
        ],
    },
    "S7": {
        "name": "Privacy Violations",
        "keywords": [
            r"\bdox", r"\bpersonal.*info", r"\bstalk", r"\bspy\b",
            r"\bsurveill", r"\btrack.*someone", r"\btrack.*person",
            r"\bfind.*address", r"\bfind.*phone", r"\bphone.*number",
            r"\bsocial.*security", r"\bSSN\b", r"\bcredit.*card.*number",
            r"\bprivate.*data", r"\bmonitor.*someone",
            r"\brecord.*without.*consent", r"\bsearch.*for.*personal",
            r"\blocation.*of\b", r"\bwhere.*does.*live",
            r"\bcell.*phone.*number", r"\bhome.*address",
        ],
    },
    "S8": {
        "name": "Intellectual Property/Other",
        "keywords": [
            r"\bcopyright", r"\bpirat", r"\btrademark",
            r"\bplagiari", r"\bkeygen", r"\blicense.*bypass",
            r"\breplica", r"\bknockoff",
        ],
    },
}


def categorize_instruction(instruction: str) -> str:
    """Assign an S1-S8 category to an instruction via keyword matching."""
    text = instruction.lower()
    scores = {}
    for cat, info in CATEGORIES.items():
        score = sum(1 for kw in info["keywords"] if re.search(kw, text))
        scores[cat] = score

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "S5"  # Default to S5 (defamation/hateful) for uncategorized
    return best


def main():
    parser = argparse.ArgumentParser(description="Annotate seed data with S1-S8 categories")
    parser.add_argument("--input", type=str, default="prompts/sft_dataset.json")
    parser.add_argument("--output", type=str, default="prompts/sft_dataset_categorized.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    from collections import Counter
    cat_counts = Counter()

    for item in data:
        category = categorize_instruction(item["instruction"])
        item["category"] = category
        cat_counts[category] += 1

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Annotated {len(data)} items → {args.output}")
    print("\nCategory distribution:")
    for cat in sorted(cat_counts.keys()):
        name = CATEGORIES[cat]["name"]
        count = cat_counts[cat]
        pct = 100 * count / len(data)
        print(f"  {cat} ({name:<25}): {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
