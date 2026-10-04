# Synthetic Transformer Training Dataset

Scope:
- Decoder-only transformer instruction corpus
- Basic natural-language interaction and comprehension
- 2x2 systems of linear equations with integer solutions
- Basic word problems that translate to 2 equations / 2 unknowns

Files:
- train.jsonl: 60,000 examples
- val.jsonl: 6,000 examples
- test.jsonl: 6,000 examples
- preview.jsonl: 18 sample records

Category mix in every split:
- 50% natural_language
- 30% linear_system
- 20% linear_system_word_problem

Recommended use:
- Train tokenizer on train.jsonl only.
- Train the LM on the `text` field.
- Use val.jsonl for checkpoint/model selection.
- Keep test.jsonl untouched until final evaluation.
- Mask loss on the user/prompt span if you want instruction-style supervised training.
- For plain next-token training, you can train on the complete `text` field.

Special tokens used in text:
<|user|>
<|assistant|>
<|end|>

Generalization design:
- Direct algebra systems use wider solution/coefficient ranges in validation and test.
- Exact prompt/response duplicates are removed across all splits.
- Word problems vary names, quantities, prices, and wording.

Important limitations:
- Natural-language coverage is intentionally basic and synthetic. It teaches format,
  short dialogue, simple comprehension, instructions, vocabulary relations, and simple arithmetic,
  not broad world knowledge.
- Math scope is intentionally limited to 2 equations / 2 unknowns.
- All generated system solutions are integers.
- Word problems are limited to common count/cost and count/attribute patterns.

Seed: 20261002
