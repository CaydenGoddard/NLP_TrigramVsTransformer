"""
NLP_TrigramVsTransformer
Assignment 8.5 - From Trigrams to Transformers

This program:
  1. Builds a trigram language model from a small training corpus.
  2. Uses it to predict the next word for three test prompts.
  3. Simulates "transformer thinking" by assigning attention weights to
     every word in each prompt and using those weights to score candidate
     next words.
  4. Shows a side-by-side comparison of the two approaches.
"""

import string
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# PART 1: TRIGRAM MODEL
# A trigram model predicts the next word using ONLY the previous two words.
# ---------------------------------------------------------------------------

def preprocess_text(text):
    """Lowercase the text, remove punctuation, and split it into tokens."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


def build_trigram_counts(tokens):
    """
    Count how often each word follows each pair of words.
    Result looks like: (word1, word2) -> Counter({next_word: count})
    """
    trigram_counts = defaultdict(Counter)

    for i in range(len(tokens) - 2):
        context = (tokens[i], tokens[i + 1])
        next_word = tokens[i + 2]
        trigram_counts[context][next_word] += 1

    return trigram_counts


def build_trigram_probabilities(trigram_counts):
    """
    Turn raw counts into probabilities.
    P(next | w1, w2) = count(w1, w2, next) / count(w1, w2, anything)
    """
    trigram_probs = {}

    for context, counter in trigram_counts.items():
        total = sum(counter.values())
        trigram_probs[context] = {word: count / total for word, count in counter.items()}

    return trigram_probs


def predict_next_words(word1, word2, trigram_probs, top_n=3):
    """Return the top N most likely next words for a two-word context."""
    context = (word1.lower(), word2.lower())

    if context not in trigram_probs:
        return []

    predictions = sorted(trigram_probs[context].items(), key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


def get_context_words(prompt):
    """
    Return the final two words of a prompt - the only context a trigram sees.
    Example: 'The cat sat on the ___' -> ('on', 'the')
    """
    cleaned = prompt.replace("___", "").strip().lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    words = cleaned.split()

    if len(words) < 2:
        return None

    return words[-2], words[-1]


# ---------------------------------------------------------------------------
# PART 2: SIMULATED TRANSFORMER ATTENTION
# A transformer can look at EVERY word in the prompt. Attention weights say
# how much each word should influence the prediction. Weights add up to 1.
# ---------------------------------------------------------------------------

# Hand-assigned attention weights for each prompt.
# Repeated words are labeled with their position so each one gets its own weight.
ATTENTION_WEIGHTS = {
    "The cat sat on the ___": {
        "cat": 0.40,          # WHO is sitting - the most important clue
        "sat": 0.20,          # the action - we need something you sit on
        "on": 0.20,           # tells us the answer is a surface
        "the (2nd)": 0.10,    # tells us a noun comes next
        "the (1st)": 0.10,    # little meaning on its own
    },
    "I want to eat ___": {
        "eat": 0.50,          # the answer must be food
        "want": 0.25,         # a desire, so a specific food or meal
        "I": 0.15,            # the speaker
        "to": 0.10,           # grammar word, low meaning
    },
    "She picked up the phone because ___": {
        "phone": 0.35,        # the object - phones ring
        "because": 0.20,      # a REASON is coming next
        "picked": 0.20,       # the action that needs explaining
        "she": 0.15,          # the subject
        "up": 0.05,
        "the": 0.05,
    },
}

# How strongly each candidate word relates to each context word (0 to 1).
# In a real transformer these relationships are learned from huge amounts of
# text. Here they are set by hand to show the idea.
CANDIDATE_RELEVANCE = {
    "The cat sat on the ___": {
        "mat":   {"cat": 0.9, "sat": 0.9, "on": 0.9, "the (2nd)": 1.0, "the (1st)": 0.5},
        "rug":   {"cat": 0.6, "sat": 0.9, "on": 0.9, "the (2nd)": 1.0, "the (1st)": 0.5},
        "phone": {"cat": 0.1, "sat": 0.2, "on": 0.5, "the (2nd)": 1.0, "the (1st)": 0.5},
    },
    "I want to eat ___": {
        "pizza":  {"eat": 1.0, "want": 0.8, "I": 0.5, "to": 0.5},
        "pasta":  {"eat": 1.0, "want": 0.7, "I": 0.5, "to": 0.5},
        "dinner": {"eat": 0.9, "want": 0.6, "I": 0.5, "to": 0.5},
        "phone":  {"eat": 0.0, "want": 0.5, "I": 0.5, "to": 0.5},
    },
    "She picked up the phone because ___": {
        "it":  {"phone": 1.0, "because": 0.9, "picked": 0.8, "she": 0.4, "up": 0.5, "the": 0.5},
        "she": {"phone": 0.5, "because": 0.9, "picked": 0.7, "she": 1.0, "up": 0.5, "the": 0.5},
        "the": {"phone": 0.3, "because": 0.6, "picked": 0.3, "she": 0.2, "up": 0.5, "the": 0.3},
    },
}


def attention_predict(prompt):
    """
    Score each candidate word as a weighted sum:
        score(candidate) = sum over context words of (weight * relevance)
    The candidate with the highest score is the prediction.
    """
    weights = ATTENTION_WEIGHTS[prompt]
    candidates = CANDIDATE_RELEVANCE[prompt]

    scores = {}
    for candidate, relevance in candidates.items():
        scores[candidate] = sum(weights[w] * relevance.get(w, 0) for w in weights)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def print_attention_simulation(prompt):
    """Print the attention weights, candidate scores, and final prediction."""
    weights = ATTENTION_WEIGHTS[prompt]
    ranked = attention_predict(prompt)

    print(f"\nPrompt: {prompt}")
    print("Attention weights:")
    for word, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        bar = "#" * int(weight * 40)
        print(f"  {word:<10} {weight:.2f}  {bar}")
    print(f"  {'TOTAL':<10} {sum(weights.values()):.2f}")

    print("Candidate scores (weighted sum):")
    for word, score in ranked:
        print(f"  {word:<8} {score:.3f}")

    print(f"Transformer-style prediction: {ranked[0][0]}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # Small training corpus
    text = """
    The cat sat on the mat.
    The dog sat on the rug.
    I want to eat pizza.
    I want to eat pasta.
    I want to eat dinner.
    She picked up the phone because it rang.
    She picked up the phone because she was expecting a call.
    The student opened the book because the assignment was difficult.
    The chef cooked dinner because the guests were hungry.
    """

    tokens = preprocess_text(text)
    trigram_counts = build_trigram_counts(tokens)
    trigram_probs = build_trigram_probabilities(trigram_counts)

    prompts = list(ATTENTION_WEIGHTS.keys())

    # ----- Trigram results -----
    print("=" * 50)
    print("TRIGRAM MODEL RESULTS")
    print("=" * 50)

    trigram_top = {}
    for prompt in prompts:
        context = get_context_words(prompt)
        print(f"\nPrompt: {prompt}")

        if not context:
            print("Not enough words for trigram context.")
            continue

        word1, word2 = context
        predictions = predict_next_words(word1, word2, trigram_probs, top_n=3)
        print(f"Trigram context used: ({word1}, {word2})")

        if not predictions:
            print("No trigram predictions found.")
            trigram_top[prompt] = None
        else:
            print("Top predictions:")
            for i, (word, prob) in enumerate(predictions, start=1):
                print(f"  {i}. {word} ({prob:.3f})")
            trigram_top[prompt] = predictions[0]

    # ----- Limitation demo: the trigram can't tell cat from dog -----
    print("\nLimitation check - same last two words, different subject:")
    for prompt in ["The cat sat on the ___", "The dog sat on the ___"]:
        w1, w2 = get_context_words(prompt)
        print(f"  {prompt:<26} -> {predict_next_words(w1, w2, trigram_probs)}")
    print("  The trigram gives identical answers because it never sees 'cat' or 'dog'.")

    # ----- Simulated attention -----
    print("\n" + "=" * 50)
    print("SIMULATED TRANSFORMER ATTENTION")
    print("=" * 50)
    for prompt in prompts:
        print_attention_simulation(prompt)

    # ----- Side-by-side comparison -----
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"{'Prompt':<38}{'Trigram':<18}{'Attention'}")
    for prompt in prompts:
        tri = trigram_top.get(prompt)
        tri_str = f"{tri[0]} ({tri[1]:.2f})" if tri else "none"
        att = attention_predict(prompt)[0][0]
        print(f"{prompt:<38}{tri_str:<18}{att}")


if __name__ == "__main__":
    main()
