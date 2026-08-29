# Why we do not use `user_profile`

**Status:** closed, with numbers. Written for a reader who wants the reasoning
checked, not just the conclusion.

Every sample in `data/public_set.jsonl` ships a `user_profile` dict. Our agent
receives it, stores it, and deliberately never lets it influence what we
recommend. This document is the justification, because "we ignored one of the
inputs" is a decision that should be defended rather than assumed.

The short version: we tested it, one of the tests came back statistically
significant, and we still say no — because significance is not the same thing
as usefulness, and the significant result did not survive a robustness check.

---

## A note on the statistical terms used below

We use a handful of standard terms. Each is defined here once so the tables
later are readable without a statistics background.

**Sample size (`n`).** How many observations a number is computed from. The
public set has 200 samples, so every claim here has at most n = 200 behind it.
Small `n` inside a subgroup is the most common way a real-looking pattern turns
out to be nothing.

**Correlation coefficient (`r`).** A number between −1 and +1 measuring how
strongly two quantities move together. `r = 0` means knowing one tells you
nothing about the other. `r = 1` means knowing one tells you the other exactly.
It measures *straight-line* association only, which is why we also inspect the
data broken into groups.

**Null hypothesis.** The assumption that there is no real relationship, and
that any pattern seen is coincidence. Statistical testing works by asking how
surprising the observed data would be if the null hypothesis were true.

**Permutation test and `p`-value.** Instead of trusting a formula, we shuffle
the data to destroy any real relationship, recompute the correlation, and
repeat 20,000 times. That gives us the range of correlations coincidence alone
produces — the *null distribution*. The `p`-value is the fraction of those
shuffles that matched or beat what we actually observed. A small `p` means
coincidence rarely produces a result this large. The usual convention is that
`p < 0.05` counts as "statistically significant".

**Effect size, and why it is the number that matters.** A `p`-value answers
"is this relationship real?". It does not answer "is this relationship big
enough to act on?". A very small relationship can be certainly real and still
be useless. The second question is the one an engineering decision turns on.

**Variance explained (`r²`).** The correlation squared. It is the share of the
variation in one quantity that the other accounts for. This converts a
correlation into a plain statement about how much predictive power you actually
gained.

**Marginal vs conditional probability.** The *marginal* is the overall base
rate — how often something is true across all samples. The *conditional* is the
rate within a subgroup. A feature is only informative if the conditional
differs from the marginal. If knowing the subgroup does not move the rate, the
feature carries no information about the outcome.

**Lift.** The conditional rate divided by the marginal rate. Lift of 1.0 means
no information. Lift of 14 means the subgroup is 14 times more concentrated
than the base rate.

**Multiple comparisons.** If you run twenty independent tests at the `p < 0.05`
threshold, roughly one will come back "significant" purely by chance. Testing
many fields and then reporting the one that passed is a well-known way to
manufacture a false finding. The defence is to decide in advance what would
count, and to treat a lone pass among many tests with suspicion.

---

## What the field actually contains

Before testing anything, we counted what is in the field across all 200
samples. Two of the five keys carry no information at all.

| key | values observed | information content |
|---|---|---|
| `purchase_frequency` | `"3-4 prior purchases"` on **200 of 200** | none — it is a constant |
| `summary` | template restatement of `preference_tags` + `rating_style` | none — it is a derived duplicate |
| `preference_tags` | 9-value vocabulary, 2–3 tags per sample | possibly |
| `average_prior_rating` | 1.0–5.0, 134 of 200 are 5.0 | possibly |
| `rating_style` | `usually positive` / `critical` / `mixed` | possibly |

A constant field cannot discriminate between products by definition. If every
customer has the same value, the value cannot explain why customers want
different things. That removes two of the five keys before any statistics.

That leaves three fields worth testing.

---

## Test 1 — does `average_prior_rating` predict the target's rating?

The idea being tested: a customer who rates things highly might be shopping for
highly-rated products. If so, we could push well-rated candidates up the list
for generous raters.

We correlated each sample's `average_prior_rating` against the `average_rating`
of that sample's hidden target product.

```
observed r          = 0.1824
permutation p       = 0.0094   (188 of 20,000 shuffles reached |r| >= 0.1824)
null |r| 95th pct   = 0.1389
```

**This is a statistically significant result.** `p = 0.0094` is well under the
0.05 convention. Coincidence produced a correlation this large in under 1% of
shuffles. Taken at face value, the field predicts the target's rating.

We did not ship it, for two independent reasons.

**Reason 1: it is not robust.** Breaking the correlation out by group shows it
is not a trend at all.

```
average_prior_rating    n     mean target rating
        1.0            14           4.393
        2.0             9           4.067    <- the entire effect
        3.0            22           4.300
        4.0            21           4.305
        5.0           134           4.413
```

The relationship is supposed to be "higher prior rating, higher target rating".
The most critical raters (1.0) land at 4.393 and the most generous (5.0) land
at 4.413 — effectively identical. The four other cells span 4.30 to 4.41,
against a standard deviation of 0.261. The whole correlation is carried by one
group of **nine samples** sitting low.

Removing that one cell collapses the result:

```
r with the 9-sample cell dropped = 0.0929   (n = 191)
null |r| 95th percentile         = 0.1389
```

0.093 is now *below* the level coincidence routinely produces. A finding that
depends on 9 of 200 observations, and that reverses the expected ordering
between its largest groups, is not a finding we are willing to put in a scored
pipeline.

**Reason 2: even if it were real, it is far too small to use.**

```
r          = 0.1824
r squared  = 0.0333    -> explains 3.3% of the variance in target rating
target rating spread (sd)            = 0.261
remaining spread after using profile = 0.257
improvement in predictive precision  = 1.7%
```

This is the effect-size point made concrete. Knowing the customer's prior
rating narrows our estimate of the target's rating by **1.7%**. Our ranking
decisions are not made at that resolution — our own benchmark cannot reliably
distinguish two configurations less than one session apart out of 200. A 1.7%
sharpening of one weak feature cannot move a decision the system is not
sensitive enough to notice.

---

## Test 2 — do `preference_tags` predict what the customer wants?

The idea being tested: a customer tagged `warmth` or `performance` might be
shopping in a different part of the catalog than one tagged `style`.

We compared, for each tag, the coarse category of the target products of
customers carrying that tag (the *conditional*) against the category
distribution across all 200 samples (the *marginal*).

The marginal is 59.0% `Women`.

| tag | n | most common target category | conditional | marginal |
|---|---|---|---|---|
| fit | 163 | Women | 0.577 | 0.590 |
| material | 154 | Women | 0.617 | 0.590 |
| comfort | 144 | Women | 0.625 | 0.590 |
| style | 101 | Women | 0.644 | 0.590 |
| durability | 47 | Women | 0.574 | 0.590 |
| performance | 26 | Women | 0.462 | 0.590 |
| warmth | 18 | Women | 0.444 | 0.590 |
| weather | 12 | Women | 0.417 | 0.590 |

Every conditional sits within noise of the marginal. Knowing the tag does not
change what we should expect the customer to be shopping for, which is the
definition of a feature carrying no information about the outcome.

There is a second, structural problem visible in the `n` column. The top four
tags appear on 50% to 82% of all samples. A feature present on four customers
in five cannot separate those customers from each other, however well it
correlates with anything. The two tags that lean furthest from the marginal,
`warmth` (0.444) and `weather` (0.417), are exactly the two with the smallest
groups — 18 and 12 samples — which is the pattern noise makes, not signal.

---

## Test 3 — is the profile an identity signal?

The remaining idea is personalization proper: recognise a returning shopper and
carry their history forward.

The API contract exposes no customer ID, so the only thing to key on is the
profile dict's own contents. We hash it and treat two sessions presenting an
identical profile as the same shopper. Across the public set that gives 125
distinct profiles over 200 samples, with 30 profiles seen more than once.

We then asked whether same-profile sessions are actually shopping for similar
things.

```
same-profile-key sessions whose targets share a coarse category   0.5%
random pairs of sessions whose targets share a coarse category    1.2% +/- 0.5%
```

Sessions matching on profile agree **below** the rate of two sessions picked at
random. A profile match is therefore not evidence of a shared shopper. There is
no correct history to carry forward, so the personalization mechanism has
nothing correct to do.

We built the persistence machinery anyway (`starter/user_profile.py`) and it
runs correctly — it is populated, corroborated across sessions, and survives
process restarts. It simply is not consulted, on the evidence above.

---

## On multiple comparisons, stated against ourselves

We ran roughly a dozen tests across three profile fields. Exactly one came back
under `p < 0.05`. That is close to what pure chance produces at that threshold
with that many tests, which is the honest way to read a single pass among many.

We flag this deliberately, because the tempting move at this point is to report
the one significant result as a discovered feature. Our conclusion runs the
other way: the single pass is what a null field looks like when you test it
enough times, and it failed its robustness check besides.

---

## What we did instead

The same analysis surfaced a much stronger signal in the catalog, and it has
nothing to do with the user profile.

`rating_number` — the count of reviews a product has — is the only field with
100% catalog coverage. Material is 70.9%, colour 39.9%, price about 21%. And
target products are drawn overwhelmingly from the popular tail.

| review-count threshold | % of catalog above it | % of targets above it | lift |
|---|---|---|---|
| 100 | 18.8% | 95.0% | 5.1x |
| 500 | 5.9% | 83.0% | 14.1x |
| 1,000 | 3.1% | 74.5% | 24.3x |
| 5,000 | 0.7% | 54.0% | 76.7x |

The catalog's median product has 12 reviews. The median target has 7,078.

This is the contrast that justifies the whole document. A usable signal looks
like a 14x lift that needs no significance test to see. A signal we should
decline looks like `r = 0.18` that needs a significance test to defend and then
fails a robustness check. We report both honestly and act only on the first.

The popularity prior is implemented as an optional third retrieval leg in
`scripts/sweep_prior_leg.py` and is **not yet measured end-to-end**. It is a
candidate, not a shipped result, and we hold it to the same standard: it must
reproduce the shipped score at weight zero before any swept point is believed.

One caveat we state rather than hide. The popularity concentration is a
property of how the evaluation samples were drawn, not a fact about shopping.
It transfers to the hidden set only if that set was drawn the same way. That is
a real dependency and we would not want a judge to discover it for us.

---

## Summary

We ignore `user_profile` because we measured it and it does not carry usable
information.

- Two of its five fields are constant or duplicated, so they cannot discriminate.
- `average_prior_rating` reaches statistical significance but explains 3.3% of
  the variance, improves prediction by 1.7%, and collapses to nothing when a
  9-sample group is removed.
- `preference_tags` conditionals are indistinguishable from the base rate, and
  the common tags are too near-universal to separate customers anyway.
- Profile-key matches agree on target category *below* the random-pair rate, so
  the field is not an identity signal.

Using it anyway would have added a personalization story to the report and
noise to the ranking. We would rather be able to show the measurement.
