# # NLP_TrigramVsTransformer

Assignment 8.5 – From Trigrams to Transformers

This project compares a simple **trigram language model** with a hand-built simulation of **transformer-style attention**, to show why using more context gives better next-word predictions.

## How to run

```bash
python trigram_vs_transformer.py
```

No extra libraries needed. It only uses Python's standard library.

## The trigram model

A trigram model predicts the next word using **only the previous two words**.

1. The training text is lowercased, punctuation is removed, and it's split into words.
2. Every group of three words is counted: `(word1, word2) -> next word`.
3. Counts are turned into probabilities:
   `P(next | w1, w2) = count(w1, w2, next) / count(w1, w2)`
4. To predict, the model takes the last two words of the prompt and returns the most likely next words.

## The attention simulation

A transformer can look at **every** word in the prompt, not just the last two. To simulate this:

1. Each word in the prompt gets an **attention weight** (0 to 1, adding up to 1) based on how important it is.
2. Each candidate next word gets a **relevance score** with each context word (set by hand here; a real transformer learns these).
3. Each candidate's final score is a weighted sum: `score = Σ (weight × relevance)`.
4. The highest-scoring candidate is the prediction.

## Example output

```
Prompt: The cat sat on the ___
Trigram context used: (on, the)
Top predictions:
  1. mat (0.500)
  2. rug (0.500)

Limitation check - same last two words, different subject:
  The cat sat on the ___     -> [('mat', 0.5), ('rug', 0.5)]
  The dog sat on the ___     -> [('mat', 0.5), ('rug', 0.5)]

Prompt: The cat sat on the ___
Attention weights:
  cat        0.40  ################
  sat        0.20  ########
  on         0.20  ########
  the (2nd)  0.10  ####
  the (1st)  0.10  ####
  TOTAL      1.00
Candidate scores (weighted sum):
  mat      0.870
  rug      0.750
  phone    0.330
Transformer-style prediction: mat

COMPARISON
Prompt                                Trigram           Attention
The cat sat on the ___                mat (0.50)        mat
I want to eat ___                     pizza (0.33)      pizza
She picked up the phone because ___   it (0.50)         it
```

The full output is in `output.txt`.

## What I learned

- A trigram model only sees two words, so it misses important clues earlier in the sentence. It gave the exact same answer for "The **cat** sat on the" and "The **dog** sat on the" because it never saw "cat" or "dog."
- Every trigram "top prediction" here was actually a **tie**. The model wasn't confident at all; it just picked whichever word showed up first.
- Attention fixes this by letting the model weigh every word in the sentence. Important words like *cat*, *eat*, and *phone* get high weights, and filler words like *the* and *to* get low weights.
- With attention, the predictions became clear winners instead of coin flips, and they were based on the meaning of the whole sentence.
- Attention isn't magic. If the weights or learned relationships are wrong, the prediction will be wrong too.

## Files

| File | Description |
|---|---|
| `trigram_vs_transformer.py` | Trigram model, attention simulation, and comparison |
| `output.txt` | Full program output |
| `README.md` | This file |
