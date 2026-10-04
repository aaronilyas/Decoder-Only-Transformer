#!/usr/bin/env python3
"""
Compact dataset generator for a small decoder-only transformer.

This script intentionally mirrors the schema and scope of the generated corpus:
basic natural language, 2x2 systems, and two-variable word problems.
For the exact larger corpus supplied with this package, use the included JSONL files.
"""
import json, random, argparse
from pathlib import Path

def fmt_expr(a,b):
    def term(c,v,first=False):
        if c == 0: return ""
        sign = "-" if c < 0 else "+"
        n = abs(c)
        body = v if n == 1 else f"{n}{v}"
        if first: return f"-{body}" if c < 0 else body
        return f" {sign} {body}"
    out = ""
    if a != 0: out += term(a,"x",True)
    if b != 0: out += term(b,"y",not bool(out))
    return out or "0"

def system_example(rng, span=10, coeff=7):
    while True:
        x,y = rng.randint(-span,span), rng.randint(-span,span)
        a,b,d,e = [rng.randint(-coeff,coeff) for _ in range(4)]
        if (a,b)!=(0,0) and (d,e)!=(0,0) and a*e-b*d != 0:
            break
    c,f = a*x+b*y, d*x+e*y
    p = f"Solve the system:\n{fmt_expr(a,b)} = {c}\n{fmt_expr(d,e)} = {f}"
    r = f"x = {x}, y = {y}"
    return "linear_system", p, r

def language_example(rng):
    pairs = [
        ("Hello!", "Hello! How can I help?"),
        ("Thanks!", "You're welcome!"),
        ("What is a variable?", "A variable is a symbol that represents an unknown or changeable value."),
        ("What is an equation?", "An equation is a statement that two expressions are equal."),
        ("What is a synonym for 'quick'?", "fast"),
        ("What is the opposite of 'hot'?", "cold"),
    ]
    return ("natural_language",) + rng.choice(pairs)

def word_problem(rng):
    adults, children = rng.randint(2,25), rng.randint(2,25)
    ap, cp = rng.randint(8,18), rng.randint(3,7)
    total, revenue = adults+children, adults*ap+children*cp
    p = (f"Adult tickets cost ${ap} and child tickets cost ${cp}. "
         f"{total} tickets were sold for ${revenue}. How many of each were sold?")
    r = (f"Let a be adult tickets and c be child tickets. a + c = {total} and "
         f"{ap}a + {cp}c = {revenue}. Solving gives a = {adults}, c = {children}.")
    return "linear_system_word_problem", p, r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mini_dataset.jsonl")
    ap.add_argument("--examples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    gens = [language_example, system_example, word_problem]
    weights = [0.5,0.3,0.2]
    with open(args.out,"w",encoding="utf-8") as f:
        for i in range(args.examples):
            gen = rng.choices(gens, weights=weights, k=1)[0]
            cat,prompt,response = gen(rng)
            row = {
                "id": f"example-{i:06d}",
                "category": cat,
                "prompt": prompt,
                "response": response,
                "text": f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n<|end|>"
            }
            f.write(json.dumps(row, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    main()
