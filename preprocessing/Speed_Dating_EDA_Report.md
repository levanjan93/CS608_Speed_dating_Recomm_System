# Speed Dating Dataset — Exploratory Data Analysis Report

**Dataset:** Columbia Business School Speed Dating Experiment (Fisman & Iyengar, 2002–2004)  
**Source:** Kaggle — [Speed Dating Experiment](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment/data)  
**Prepared for:** Group Project Proposal — Reciprocal Recommender System  

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Structural Layout](#2-structural-layout)
3. [Column-by-Column Analysis](#3-column-by-column-analysis)
   - 3.1 Identity & Structure Columns
   - 3.2 Outcome Columns
   - 3.3 Date Ratings (Given & Received)
   - 3.4 Pair Context Columns
   - 3.5 Demographics
   - 3.6 Academic Background (Text Fields)
   - 3.7 Lifestyle & Interests
   - 3.8 Stated Preference Surveys (All Time Points)
   - 3.9 Self-Perception & Perceived-by-Others
   - 3.10 Follow-Up Survey Columns (Time 2 & Time 3)
4. [Data Dirtiness Assessment](#4-data-dirtiness-assessment)
5. [Recommended Actions by Column Group](#5-recommended-actions-by-column-group)
6. [Usability Tier Summary](#6-usability-tier-summary)
7. [Key EDA Findings for the Proposal](#7-key-eda-findings-for-the-proposal)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Total rows | 8,378 |
| Total columns | 195 |
| Unique participants | 551 |
| Unique date pairs | 4,184 |
| Unique waves (events) | 21 |
| Date range | October 2002 – April 2004 |
| File encoding | cp1252 (Windows Latin-1) |
| Line endings | CR only (old Mac style — `\r`) |
| Core dtype split | 174 float64, 13 int64, 8 string object |

**The fundamental unit of a row is a directional interaction:** one person's evaluation of one partner during one event. Because every pair is evaluated in both directions, the 8,378 rows represent 4,184 unique pairs × 2 directions, giving exactly **8,368 directional rows + 10 rows with null `pid`** (orphaned or data-entry errors).

Each participant appears in multiple rows — one per partner they met. The minimum is 5 dates (tiny wave), the median is 16, and the maximum is 22.

---

## 2. Structural Layout

### 2.1 Wave Summary

The experiment ran 21 events. Match rates varied substantially across waves, partly driven by the number of participants and the experimental condition.

| Wave | Date | Participants | Males | Females | Round Size | Condition | Match Rate |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Oct 16 '02 | 20 | 10 | 10 | 10 | Limited | 31.0% |
| 2 | Oct 23 '02 | 35 | 16 | 19 | 16 | Extensive | 10.2% |
| 3 | Nov 12 '02 | 20 | 10 | 10 | 10 | Limited | 13.0% |
| 4 | Nov 12 '02 | 36 | 18 | 18 | 18 | Extensive | 20.1% |
| 5 | Nov 20 '02 | 19 | 10 | 9 | 10 | Limited | 28.4% |
| 6 | Mar 26 '03 | 10 | 5 | 5 | 5 | Limited | 20.0% |
| 7 | Mar 26 '03 | 32 | 16 | 16 | 16 | Extensive | 16.8% |
| 8 | Apr 2 '03 | 20 | 10 | 10 | 10 | Limited | 18.0% |
| 9 | Apr 2 '03 | 40 | 20 | 20 | 20 | Extensive | 15.5% |
| 10 | Sep 24 '03 | 18 | 9 | 9 | 9 | Limited | 18.5% |
| 11 | Sep 24 '03 | 42 | 21 | 21 | 21 | Extensive | 14.5% |
| 12 | Oct 7 '03 | 28 | 14 | 14 | 14 | Extensive | **10.7%** ⚠️ |
| 13 | Oct 8 '03 | 19 | 9 | 10 | 9 | Limited | 17.8% |
| 14 | Oct 8 '03 | 38 | 18 | 20 | 18 | Extensive | 17.5% |
| 15 | Feb 24 '04 | 37 | 19 | 18 | 19 | Extensive | 18.4% |
| 16 | Feb 25 '04 | 14 | 8 | 6 | 8 | Limited | 25.0% |
| 17 | Feb 25 '04 | 24 | 14 | 10 | 14 | Extensive | 17.1% |
| 18 | Apr 6 '04 | 12 | 6 | 6 | 6 | Limited | **8.3%** ⚠️ |
| 19 | Apr 6 '04 | 30 | 15 | 15 | 15 | Extensive | 16.4% |
| 20 | Apr 7 '04 | 13 | 7 | 6 | 7 | Limited | 11.9% |
| 21 | Apr 7 '04 | 44 | 22 | 22 | 22 | Extensive | 14.9% |

> ⚠️ **Wave 12** had an explicit budget constraint: participants were only allowed to say "yes" to ≤50% of their partners. Decision rate (33.9%) and match rate (10.7%) are structurally suppressed. **This wave should be excluded or flagged** in any model that uses `dec` as the target.

### 2.2 Preference Scale Heterogeneity

**This is the most important structural data quality issue in the dataset.** The six preference attributes (attractiveness, sincerity, intelligence, fun, ambition, shared interests) were elicited on two completely different scales:

| Scale Type | Waves | Participants | Interactions |
|---|---|:---:|:---:|
| 100-point allocation (sum = 100) | 1–5, 10–21 | 449 | 6,816 |
| 1–10 importance rating (independent) | 6–9 | 102 | 1,562 |

A value of `20` on `attr1_1` means completely different things depending on the wave: in 100-point waves it is 20% weight allocated to attractiveness; in 1–10 waves it is out of range. **These cannot be compared directly and must be normalized before any cross-wave analysis.**

### 2.3 Two Experimental Conditions

| Condition | Label | Waves | Interactions | Match Rate |
|---|---|---|:---:|:---:|
| 1 | Limited choice | 1,3,5,6,8,10,13,16,18,20 | 1,434 | 20.2% |
| 2 | Extensive choice | 2,4,7,9,11,12,14,15,17,19,21 | 6,944 | 15.7% |

Limited choice events (fewer people) have a notably higher match rate. Any model must control for or stratify on condition.

---

## 3. Column-by-Column Analysis

### 3.1 Identity & Structure Columns

These columns identify who is interacting with whom and the experimental context.

| Column | Type | Missing | Description | Notes |
|---|---|---|---|---|
| `iid` | int | 0% | Unique global participant ID | Key join field; values are group(wave × gender) |
| `id` | float | 0.01% | Subject number within wave | 1 null; use `iid` as the primary key |
| `gender` | int | 0% | 0 = Female, 1 = Male | Perfectly balanced: 4,184 female rows, 4,194 male rows |
| `idg` | int | 0% | Subject number within gender group | Redundant with `iid` for most purposes |
| `condtn` | int | 0% | 1 = Limited choice, 2 = Extensive choice | Confounds match rate; model as covariate |
| `wave` | int | 0% | Event number (1–21) | Drives preference scale, group size, match rate |
| `round` | int | 0% | Number of dates that night | Equals the number of partners met in that wave |
| `position` | int | 0% | Station number where this date occurred | Useful for order-effect studies |
| `positin1` | float | **22.0%** | Starting station number | Null for all waves 1–5 (not collected); clean for waves 6–21 |
| `order` | int | 0% | Sequence number of this date within the night | 1 = first date, max = last date; strong first-date effect (see §7) |
| `partner` | int | 0% | Partner's ID number within the wave | Use `pid` (global iid) for cross-wave joins |
| `pid` | float | 0.1% | Partner's global `iid` | 10 null values — likely data-entry errors; drop these rows |

**Action:** `positin1` is structurally missing for waves 1–5, not randomly missing — it was simply not collected. Do not impute. Use `position` (0% missing) as the positional indicator. Drop the 10 rows where `pid` is null.

---

### 3.2 Outcome Columns

These are the three columns that directly define what the recommender system is trying to predict.

| Column | Type | Missing | Value Range | Stats |
|---|---|---|---|---|
| `dec` | int | 0% | 0 or 1 | P(yes) = 41.99%; males say yes more often (47.4%) than females (36.5%) |
| `dec_o` | int | 0% | 0 or 1 | P(yes) = 41.95%; mirror of `dec` from the partner's row |
| `match` | int | 0% | 0 or 1 | P(match) = 16.47%; perfectly consistent with `dec==1 & dec_o==1` (0 inconsistencies) |

**Key reciprocity finding:** Pearson(dec, dec_o) = **−0.047** (p < 0.001). Decisions are essentially independent of each other. The expected match rate under independence is 41.99% × 41.95% = 17.6%, and the actual rate is 16.5% — slightly below independence, meaning people slightly *avoid* mutual attraction. This makes the problem genuinely two-sided: predicting one direction does not predict the other.

**Match rate by wave:** Ranges from 8.3% (wave 18) to 31.0% (wave 1). Wave effects are real and must be addressed in evaluation — train/test splits should be wave-based, not row-based, to prevent data leakage.

---

### 3.3 Date Ratings (Given & Received)

These are filled out on the scorecard during the event. Each person rates their partner on six attributes plus two global questions. The `_o` suffix means the same column from the partner's perspective (i.e., how the partner rated you).

#### Ratings Given (`attr`, `sinc`, `intel`, `fun`, `amb`, `shar`, `like`, `prob`)

| Column | Missing | Mean | Std | Min | Median | Max | Notes |
|---|---|---|---|---|---|---|---|
| `attr` | 2.4% | 6.19 | 1.95 | 0 | 6.0 | 10 | Strongest single predictor of `dec` (r = +0.49) |
| `sinc` | 3.3% | 7.18 | 1.74 | 0 | 7.0 | 10 | Weakest predictor (r = +0.21) |
| `intel` | 3.5% | 7.37 | 1.55 | 0 | 7.0 | 10 | Weakest predictor (r = +0.22); inflation at 7–8 |
| `fun` | 4.2% | 6.40 | 1.95 | 0 | 7.0 | 10 | Strong predictor (r = +0.41) |
| `amb` | **8.5%** | 6.78 | 1.79 | 0 | 7.0 | 10 | Weak predictor (r = +0.18); highest missingness |
| `shar` | **12.7%** | 5.47 | 2.16 | 0 | 6.0 | 10 | Strong predictor (r = +0.40); most missing |
| `like` | 2.9% | 6.13 | 1.84 | 0 | 6.0 | 10 | Global "like" rating; strongest predictor (r = +0.51) |
| `prob` | 3.7% | 5.21 | 2.13 | 0 | 5.0 | 10 | "Probability they'll say yes to you"; r(dec) = +0.31 |

**Data quality issues in rating columns:**
- **Zero values:** Ratings of 0 exist for `attr` (8), `sinc` (9), `intel` (5), `fun` (14), `amb` (5), `shar` (59), `like` (8), `prob` (49). The scale is 1–10, so 0 is out-of-range. These are almost certainly "did not rate" entries that should be treated as missing, not as the lowest possible rating. This is especially pronounced for `shar` (59 zeros = 0.8% of all rows) and `prob` (49 zeros = 0.6%). **Action: replace zeros with NaN before any analysis.**
- **One value of 10.5** exists in `attr_o`, and **one value of 11.0** in `fun_o`. These are entry errors and should be capped or replaced with 10 and NaN respectively.
- **`shar` and `amb`** have the most missing data of any per-date ratings (12.7% and 8.5% respectively). This is partially structural: some earlier waves may not have included these questions, and participants may have felt these were harder to assess in 4 minutes. Use caution when using these as features — imputing with the column mean is the most defensible simple approach.

#### Ratings Received (`attr_o`, `sinc_o`, `intel_o`, `fun_o`, `amb_o`, `shar_o`, `like_o`, `prob_o`)

These mirror the "Given" columns exactly, but represent the partner's ratings of the participant. Missing rates are nearly identical.

**Person-level aggregates of received ratings are extremely informative for a recommender:**
- Mean `attr_o` received correlates **r = +0.79** with the percentage of people who said yes to a person (their "desirability score").
- The popularity distribution is highly concentrated: top quintile receives yes 77.5% of the time; bottom quintile only 11.7%.

---

### 3.4 Pair Context Columns

| Column | Missing | Description | Key Finding |
|---|---|---|---|
| `int_corr` | 1.9% | Pearson correlation between the two people's 17 interest ratings | Mean = 0.20, range [−0.83, 0.91]. Correlates with `match` at only r = +0.031 and with `dec` at r = +0.019. Statistically significant but practically negligible. |
| `samerace` | 0% | 1 if same race, 0 if different | Match rate: 17.1% same race vs 16.1% different. Small effect. Decision rate: 43.4% vs 41.1%. |
| `met` | 4.5% | Whether they had met before (1=yes, 2=no) | Use with caution — counterintuitively coded (1=yes, 2=no, unlike most binary flags). |
| `met_o` | 4.6% | Partner's report of whether they had met before | Should agree with `met`; small discrepancies may reflect memory asymmetry. |

**Action:** `int_corr` has theoretical appeal (shared interests should matter) but the empirical signal is nearly zero. It can be included as a feature but should not be expected to drive model performance. For `met`/`met_o`, re-encode to 0/1 (0=no, 1=yes) to match the convention of every other binary in the dataset.

---

### 3.5 Demographics

All demographic columns are person-level attributes — they repeat across every row for the same participant. Use `drop_duplicates('iid')` to analyze at the person level (551 rows).

| Column | Missing | Description | Notes |
|---|---|---|---|
| `age` | 1.1% | Participant age | Mean 26.1 (F) / 26.6 (M); range 18–55. One female outlier at 55. |
| `age_o` | 1.2% | Partner's age | Derived from the partner's `age`; can be reconstructed if needed. |
| `race` | 0.8% | Race (1=Black, 2=White, 3=Latino, 4=Asian, 5=NativeAm, 6=Other) | 304 White (55%), 136 Asian (25%), 42 Latino (8%), 37 Other (7%), 26 Black (5%). **Skewed toward White and Asian**, reflecting the Columbia graduate school population. |
| `race_o` | 0.9% | Partner's race | Derived from partner's `race`. |
| `imprace` | 0.9% | Importance of same-race partner (1–10) | Mean 3.7; heavily right-skewed; median = 3. Most participants place little importance on this. |
| `imprelig` | 0.9% | Importance of same-religion partner (1–10) | Mean 3.6; similar distribution to `imprace`. |
| `from` | 0.9% | State/country of origin | Free text. 259+ unique values. Useful only if geocoded; recommend dropping in favor of `zipcode`-derived income. |
| `zipcode` | **12.7%** | Zip code of area grown up in | High missingness; likely from international participants. `income` is derived from this. |
| `income` | **48.9%** | Median household income (Census, based on `zipcode`) | 49% missing — almost exclusively international participants or those who didn't provide zip codes. Of the 281 non-null values: mean = $45,286, range = $8,607–$109,031. **Recommend dropping** for the recommender — too sparse and the missing pattern is non-random (proxies for being international). |

**Age difference effect on matching:**
| Age gap | Match rate |
|---|---|
| 0 years | 20.4% |
| 1–2 years | 18.6% |
| 3–5 years | 15.9% |
| 6–10 years | 13.1% |
| 11+ years | 8.0% |

Age difference is a useful pair-level feature to derive: `age_diff = |age - age_o|`.

**Race data note:** The race distribution is highly non-representative of the general population. Most participants are Columbia graduate students, and the dataset over-represents White and Asian participants. Any analysis of racial preference effects (as in the original Fisman/Iyengar paper) must account for this sampling bias.

---

### 3.6 Academic Background (Text / Semi-Structured)

| Column | Missing | Description | Recommendation |
|---|---|---|---|
| `field` | 0.8% | Free-text field of study | 259 unique values including case variants ("Law" / "law", "MBA" / "mba"). Use `field_cd` instead. |
| `field_cd` | 1.0% | Coded field (1–18) | Clean and usable. Business/Finance (130), Bio/Chem/Physics (61), Engineering (56), Law (48), Poli Sci (46), Social Sci (46) are the top fields. Treat as categorical. |
| `undergra` | **41.3%** | Undergraduate institution (free text) | 241 unique values; 41% missing. Elite schools dominate non-null values (Columbia, Harvard, Yale, Berkeley). Too sparse and noisy for modeling. **Drop.** |
| `mn_sat` | **62.6%** | Median SAT of undergrad institution (from Barron's) | Stored as a string with comma formatting (e.g., "1,400.00"). Parseable, but 62.6% missing. Can be converted to numeric; median = 1,310. **Low utility given missingness — drop or use only as a proxy feature.** |
| `tuition` | **57.2%** | Annual tuition of undergrad institution | Also string-formatted; parseable. 57% missing. Mean = $21,036. Same recommendation as `mn_sat` — too sparse. **Drop.** |
| `career` | 1.1% | Free-text intended career | 367 unique values including duplicates by case ("Professor" / "professor", "Lawyer" / "lawyer"). Use `career_c` instead. |
| `career_c` | 1.6% | Coded career (1–17) | Usable. Academic/Research (152) and Business/Finance (144) dominate. Treat as categorical. |

---

### 3.7 Lifestyle & Interests

#### Goals, Dating Habits

| Column | Missing | Description | Notes |
|---|---|---|---|
| `goal` | 0.9% | Primary goal: 1=Fun, 2=Meet people, 3=Get a date, 4=Serious relationship, 5=Say I did it, 6=Other | Most common: Fun (228), Meet people (189). Only 22 report seeking a serious relationship. |
| `date` | 1.1% | General dating frequency (1=sev/week … 7=almost never) | Most participants date infrequently: "several times/year" (136) and "twice/month" (131) most common. |
| `go_out` | 0.9% | How often go out socially (same scale as `date`) | Similar distribution to `date`. |

#### Interest Ratings (17 activities, 1–10 scale)

All 17 interest columns have **identical missingness of 0.9%** (79 null rows) — confirming these are from the same 79 participants who skipped this section of the signup survey entirely. This is block missingness (MCAR at the person level), so the same imputation strategy applies to all 17.

| Column | Mean | Std | Gender M/F | Notes |
|---|---|---|---|---|
| `sports` | 6.40 | 2.63 | 7.05 / 5.72 | Largest gender gap |
| `tvsports` | 4.55 | 2.80 | 4.97 / 4.12 | Moderate gender gap |
| `exercise` | 6.29 | 2.45 | 6.20 / 6.38 | No gender gap |
| `dining` | 7.78 | 1.78 | 7.41 / 8.15 | High overall; slight F > M |
| `museums` | 6.97 | 2.06 | 6.51 / 7.45 | Slight F > M |
| `art` | 6.69 | 2.27 | 6.17 / 7.22 | F > M |
| `hiking` | 5.76 | 2.57 | 5.58 / 5.94 | Near-equal |
| `gaming` | 3.84 | 2.61 | 4.43 / 3.23 | M > F; **contains outliers: 5 entries with value 14** (out of range) |
| `clubbing` | 5.75 | 2.49 | 5.59 / 5.91 | Near-equal |
| `reading` | 7.65 | 2.00 | 7.39 / 7.91 | **Contains outliers: 3 entries with value 13** (out of range) |
| `tv` | 5.33 | 2.54 | 4.93 / 5.73 | F slightly higher |
| `theater` | 6.76 | 2.27 | 6.03 / 7.51 | F > M |
| `movies` | 7.90 | 1.72 | 7.65 / 8.15 | Highest mean; narrow range |
| `concerts` | 6.84 | 2.15 | 6.54 / 7.15 | Slight F > M |
| `music` | 7.88 | 1.79 | 7.71 / 8.04 | Near-equal; 2nd highest mean |
| `shopping` | 5.60 | 2.62 | 4.74 / 6.49 | F > M |
| `yoga` | 4.42 | 2.76 | 3.77 / 5.08 | F > M |

**Out-of-range outliers:** `gaming` has 5 entries with value 14 (all the same person, appearing 5 times); `reading` has 3 entries with value 13. Cap these at 10.

**Predictive value of interests:** When individual interest scores are correlated with the decision `dec`, almost none reach r > 0.05. Only `gaming` (r = +0.083) and `exercise` (r = −0.056) show weak signals. The 17-item interest vector is most useful in constructing the `int_corr` pair-level feature (which is pre-computed in the dataset) rather than using individual items as model features.

#### Expectations

| Column | Missing | Description | Notes |
|---|---|---|---|
| `exphappy` | 1.2% | Expected happiness with the people you'll meet (1–10) | Mean ≈ 5.5; useful as a control or user-level feature. |
| `expnum` | **78.5%** | Expected number of matches (out of ~20) | 78.5% missing — only collected in limited waves. **Recommend dropping.** |
| `match_es` | **14.0%** | Estimated number of matches (mid-event) | 14% missing; could be a useful calibration feature but marginal. |

---

### 3.8 Stated Preference Surveys (All Time Points)

Participants were asked to distribute importance across six attributes (attr, sinc, intel, fun, amb, shar) at multiple time points. These constitute 60+ columns. Due to the **scale heterogeneity problem** (100-pt vs 1–10), waves 6–9 must be handled separately.

#### Naming Convention Recap

| Suffix | Time | Instrument | Description |
|---|---|---|---|
| `_1` | Signup (Time 1) | Pre-event survey | Before seeing any partners |
| `_s` | Mid-event | Half-way through | After seeing ~half their partners |
| `_2` | Day-after (Time 2) | Next-day survey | Required to receive their matches |
| `_3` | Follow-up (Time 3) | 3–4 weeks later | Optional |

#### Signup Preferences (`attr1_1` … `shar1_1`)

Missing: 0.9–1.4% (the same 79 participants who skipped the signup block).

**Mean stated preferences by gender (100-pt waves only):**
| Attribute | Female | Male |
|---|---|---|
| Attractiveness | 18.8 | **29.4** |
| Sincerity | 18.2 | 16.1 |
| Intelligence | 21.5 | 19.4 |
| Fun | 17.2 | 17.5 |
| Ambition | 12.0 | 7.8 |
| Shared interests | 12.3 | 10.3 |

Men state attractiveness as far more important (29.4 vs 18.8 percentage points). Women weight intelligence and sincerity more.

**Data quality issues in 100-pt waves:** Most waves sum precisely to 100, but waves 2 and 3 contain some participants whose allocations sum to 0, 90, 101, 120, or 148. These are data-entry errors affecting ~7 participants. Rows where the sum is outside [95, 105] should be normalized to sum to 100 or excluded from preference analyses.

#### What the Opposite Sex Looks For (`attr2_1` … `shar2_1`)

Missing: 0.9–1.1%. These are beliefs about what the other gender wants, not the participant's own preferences. Useful for studying perception gaps between what you say you want, what you think others want, and what actually drives decisions.

#### Peer Preferences (`attr4_1` … `shar4_1`)

Missing: **22.5%**. These were not collected in waves 1–5, explaining the large gap. The 22% missing is structural, not random — do not impute across the board. Use only for wave-stratified analyses.

#### Mid-Event Preferences (`attr1_s` … `shar1_s`)

Missing: **51.1%**. This block appears only in certain waves and was answered mid-event. The theoretical interest is whether preferences shift after meeting real people, but the 51% missingness makes this a secondary analysis tool, not a primary modeling feature.

#### Post-Event Revealed Preferences (`attr7_2`, `attr7_3`)

Missing: **76.3–76.7%** (Time 2 retrospective), **75.9%** (Time 3 retrospective). These are the most theoretically interesting preference columns — participants reflect on what *actually* drove their yes/no decisions after the fact. However, 76% missingness makes them unsuitable as model features.

**Stated (attr1_2) vs Revealed (attr7_2) at Time 2 — 100pt waves:**
| Attribute | Female Stated | Female Revealed | Male Stated | Male Revealed |
|---|---|---|---|---|
| Attractiveness | 20.8 | **27.5** | 33.3 | 35.5 |
| Sincerity | 17.3 | 15.1 | 14.5 | 12.9 |
| Intelligence | 19.5 | 15.7 | 16.6 | 15.6 |
| Fun | 17.5 | 17.6 | 16.6 | **18.8** |
| Ambition | 10.5 | 8.3 | 7.7 | 6.4 |
| Shared interests | 15.1 | 13.7 | 11.0 | 10.5 |

After reflecting, both genders assign **more weight to attractiveness** than they stated up front (the "attraction gap"), and less to sincerity and intelligence. This aligns with the empirical finding that `attr` (r = +0.49) is by far the strongest per-date predictor of `dec`.

---

### 3.9 Self-Perception & How Others See You

#### Self-Ratings (`attr3_1` … `amb3_1`)

Missing: 1.3%. Participants rate themselves on five attributes (attractiveness, sincerity, intelligence, fun, ambition) on a 1–10 scale at signup.

| Attribute | Corr(self, received) | Mean Self | Mean Received | Over-rating |
|---|---|---|---|---|
| Attractiveness | +0.289 | 7.09 | 6.19 | +0.90 |
| Sincerity | −0.011 | ~8.2 | ~7.2 | ~+1.0 |
| Intelligence | +0.052 | ~8.4 | ~7.4 | ~+1.0 |
| Fun | +0.280 | ~6.9 | ~6.4 | +1.27 |
| Ambition | +0.158 | ~7.5 | ~6.8 | +0.76 |

**Participants systematically overrate themselves across all attributes, on average by approximately 1 point.** The correlation between self-rating and received rating is weak — especially for sincerity (r ≈ 0) and intelligence (r = +0.05). Self-ratings are a noisy proxy for actual perceived quality, but they are available at prediction time with 0% practical missingness, which makes them valuable as features despite the noise.

#### Perceived-by-Others Ratings (`attr5_1` … `amb5_1`)

Missing: **41.4%**. Participants guess how others rate them. Over 40% of the data is missing here because this block was not collected in certain waves. Use only in wave-stratified analyses; do not impute.

---

### 3.10 Follow-Up Survey Columns (Time 2 & Time 3)

These columns capture behavior and attitudes after the event.

#### Time 2 (Day After) — Event Evaluation

| Column | Missing | Description | Notes |
|---|---|---|---|
| `satis_2` | 10.9% | Overall satisfaction (1–10) | ~11% of participants did not submit day-after survey |
| `length` | 10.9% | Was 4 min: 1=Too little, 2=Too much, 3=Just right | Same 11% |
| `numdat_2` | 11.3% | Was the number of dates: 1=Too few, 2=Too many, 3=Just right | Same ~11% |

**Non-response bias check:** Participants who filled out Time 2 had slightly more matches (2.57 vs 2.00) and slightly higher yes-rate given (43.8% vs 35.4%). This suggests mild non-random attrition — people who did better in the event were more likely to return the day-after survey to get their matches.

#### Time 3 (3–4 Weeks Later) — Follow-Up Behavior

| Column | Missing | Description | Notes |
|---|---|---|---|
| `you_call` | 52.6% | How many matches did you contact? | Only ~half of participants returned the follow-up |
| `them_cal` | 52.6% | How many matches contacted you? | Same |
| `date_3` | 52.6% | Had a date with any match? (1=yes, 2=no) | Same |
| `numdat_3` | **82.1%** | How many matches dated? | Conditional on having any date |
| `num_in_3` | **92.0%** | Uh oh... | Extremely sparse; **drop** |

All Time 3 behavioral columns are too sparse to be useful as modeling features. They are potentially useful as **validation ground truth** — e.g., did our recommender's top matches actually result in real-world dates? This reframes them as an extension or X-factor component for the project, not as training features.

---

## 4. Data Dirtiness Assessment

### 4.1 Summary Table

| Issue | Severity | Columns Affected | Count |
|---|---|---|---|
| Preference scale heterogeneity (100pt vs 1–10) | 🔴 Critical | All `attr1_1`…`shar1_1` and Time 2/3 equivalents | Waves 6–9 (1,562 rows) |
| Wave 12 budget constraint (forced ≤50% yes) | 🔴 Critical | `dec`, `match` | 392 rows |
| Zero ratings on 1–10 scale (should be NaN) | 🟠 High | `attr`, `sinc`, `intel`, `fun`, `amb`, `shar`, `like`, `prob` (and `_o` equivalents) | ~250 zero values total |
| Out-of-range ratings (>10) | 🟡 Medium | `attr_o` (10.5), `fun_o` (11.0), `gaming` (14), `reading` (13) | ~10 values |
| Preference allocation not summing to 100 | 🟡 Medium | `attr1_1`…`shar1_1` in waves 2, 3, 16, 21 | ~10 participants |
| Free-text fields with case inconsistency | 🟡 Medium | `field`, `career` | Hundreds of duplicate-meaning strings |
| Structural missingness (not collected) | 🟢 Low | `positin1` (waves 1–5), `attr4_1` (waves 6–9), mid-event `_s` cols | By design |
| High random missingness | 🟠 High | `income` (49%), `mn_sat` (63%), `tuition` (57%), `undergra` (41%) | Columns to drop |
| `pid` nulls | 🟢 Low | `pid` | 10 rows |
| `met`/`met_o` reverse coding | 🟡 Medium | `met`, `met_o` | All non-null rows |

### 4.2 Missingness by Category

| Category | Avg Missingness | Pattern | Treatment |
|---|---|---|---|
| Identity/Structure | <1% | MCAR | Forward-fill or drop 10 rows |
| Outcomes (`dec`, `dec_o`, `match`) | 0% | None | Ready to use |
| Date ratings (`attr`…`shar`, `_o` versions) | 2–13% | Partially block, partially random | Impute with person-level mean or median |
| Demographics (`age`, `race`) | <2% | MCAR | Mean/mode impute |
| Interests (17 cols) | 0.9% | Block (same 79 people) | Impute with group median |
| Signup preferences (`attr1_1`…) | 0.9–1.4% | Same block | Impute or drop those 79 rows |
| Peer preferences (`attr4_1`…) | 22.5% | Structural (not collected in waves 1–5) | Wave-stratify; do not impute |
| Mid-event preferences (`_s`) | 51% | Structural | Exclude from main model |
| Day-after survey | 11% | Mild non-random attrition | Exclude from features; use as side analysis |
| Day-after retrospective prefs (`attr7_2`) | 76% | Structural | Drop from modeling |
| Follow-up behavior (Time 3) | 52–92% | Non-random attrition | Use as qualitative validation only |
| Text fields (`undergra`, `mn_sat`, `tuition`) | 41–63% | Non-random (international students) | Drop |
| `income` | 49% | Non-random (international + no zip code) | Drop |
| `expnum` | 78.5% | Structural | Drop |

---

## 5. Recommended Actions by Column Group

### 5.1 Cleaning Steps (in order)

**Step 1 — Fix the file.**
Load with `encoding='cp1252'` (Windows Latin-1). Standard UTF-8 will silently corrupt special characters. Line endings are CR-only (`\r`); pandas handles these automatically but raw byte reading does not.

**Step 2 — Drop or flag Wave 12 rows.**
The 50%-yes budget constraint structurally suppresses `dec`. Either exclude wave 12 entirely (392 rows), or add a binary flag `wave12_constrained` and model it separately.

**Step 3 — Replace zeros with NaN in rating columns.**
All 1–10 rating columns: `attr`, `sinc`, `intel`, `fun`, `amb`, `shar`, `like`, `prob` (and `_o` equivalents). Zero is not a valid rating on the 1–10 scale.

```python
rating_cols = ['attr','sinc','intel','fun','amb','shar','like','prob',
               'attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','like_o','prob_o']
df[rating_cols] = df[rating_cols].replace(0, np.nan)
```

**Step 4 — Cap out-of-range ratings at 10 (or NaN).**
One value of 10.5 in `attr_o`, one of 11.0 in `fun_o`. Cap or replace with NaN.

**Step 5 — Fix interest column outliers.**
Cap `gaming` at 10 (5 entries with value 14 are from the same person and appear to be typos). Cap `reading` at 10 (3 entries with value 13).

**Step 6 — Create a `scale_type` flag.**
```python
df['scale_type'] = np.where(df['wave'].isin([6,7,8,9]), '1-10', '100pt')
```
For any cross-wave analysis using `attr1_1`…`shar1_1`, normalize 100-pt values to [0, 1] by dividing by 100, and normalize 1–10 values by subtracting 1 and dividing by 9. Always report results stratified by `scale_type` to validate.

**Step 7 — Fix preference allocation outliers.**
For 100-pt waves where the six signup preferences do not sum to 100 (roughly 10 participants in waves 2, 3, 16, 21), either normalize each allocation to sum to 100, or drop those participants from preference analyses.

**Step 8 — Re-code `met` and `met_o`.**
Currently coded as 1=yes, 2=no. Re-encode to 1=yes, 0=no to match all other binary columns.

**Step 9 — Drop `pid`-null rows.**
10 rows with null `pid` are unmatched and cannot be used for reciprocal modeling.

**Step 10 — Engineer derived pair features.**
```python
df['age_diff'] = (df['age'] - df['age_o']).abs()
df['race_match'] = (df['race'] == df['race_o']).astype(int)  # more nuanced than samerace
df['attr_gap'] = df['attr'] - df['attr_o']                   # who rated whom higher
```

### 5.2 Columns to Drop

| Column | Reason |
|---|---|
| `id` | Redundant with `iid`; 1 null |
| `idg` | Redundant with `iid` |
| `partner` | Redundant with `pid`; within-wave only |
| `positin1` | Structurally missing for waves 1–5; `position` suffices |
| `from` | Free text, 259+ unique values; no clean encoding |
| `undergra` | 41% missing, free text, 241 unique values |
| `mn_sat` | 63% missing |
| `tuition` | 57% missing |
| `income` | 49% missing; non-random (international) |
| `zipcode` | 13% missing; only useful to derive `income` which is already present |
| `expnum` | 78% missing; only in limited waves |
| `attr7_2`…`shar7_2` | 76% missing; retrospective revealed preferences |
| `attr5_1`…`amb5_1` | 41% missing; not collected in many waves |
| `attr1_s`…`shar1_s` | 51% missing; mid-event only |
| `attr3_s`…`amb3_s` | 52% missing; mid-event only |
| `attr7_3`…`shar7_3` | 76% missing |
| `attr5_3`…`amb5_3` | 76% missing |
| `attr2_3`…`shar2_3` | 65–76% missing |
| `attr4_3`…`shar4_3` | 65% missing |
| `you_call`, `them_cal`, `date_3`, `numdat_3`, `num_in_3` | 52–92% missing; use as qualitative validation only |
| `field`, `career` | Free text; encoded versions (`field_cd`, `career_c`) exist |

### 5.3 Columns to Keep and How to Use Them

| Column Group | Keep | Treatment |
|---|---|---|
| Identity | `iid`, `pid`, `gender`, `wave`, `condtn`, `order`, `round`, `position` | Use as-is |
| Outcomes | `dec`, `dec_o`, `match` | Use as-is; target is `match` |
| Date ratings given | `attr`, `sinc`, `intel`, `fun`, `amb`, `shar`, `like`, `prob` | Fix zeros; impute missing with person-mean |
| Date ratings received | `attr_o`…`prob_o` | Same treatment; aggregate to person-level popularity features |
| Pair context | `int_corr`, `samerace`, `met` | Fix `met` coding; treat as pair features |
| Demographics | `age`, `age_o`, `race`, `race_o`, `imprace`, `imprelig` | Impute <2% missing; derive `age_diff` |
| Field & career | `field_cd`, `career_c` | Encode as categorical (one-hot or embedding) |
| Goal | `goal`, `date`, `go_out` | Impute <2% missing; encode as ordinal/categorical |
| Interests | `sports`…`yoga` (17 cols) | Fix outliers; impute with median; use for `int_corr` reconstruction |
| Expectations | `exphappy` | Keep; minor missingness |
| Signup preferences | `attr1_1`…`shar1_1` | Normalize by scale type; keep for stated preference features |
| Signup self-ratings | `attr3_1`…`amb3_1` | Keep; 1.3% missing; good proxy for user quality |
| Signup opp-sex beliefs | `attr2_1`…`shar2_1` | Keep; 0.9% missing |
| Day-after stated prefs | `attr1_2`…`shar1_2` | Keep with 11% caveat; scale-normalize |
| Day-after satisfaction | `satis_2`, `length` | Keep as context; 11% missing |

---

## 6. Usability Tier Summary

| Tier | Columns | Avg Missing | Use Case |
|---|---|---|---|
| **🟢 Core — always use** | 27 | 3.1% | Model features and targets; primary EDA |
| **🔵 High value — minor cleaning** | 54 | 2.1% | Rich features after cleaning |
| **🟡 Moderate — stratify carefully** | 13 | 33.9% | Wave-stratified analyses; secondary features |
| **🔴 Low value — drop** | ~65 | 63.5% | Not recommended for modeling |

---

## 7. Key EDA Findings for the Proposal

1. **Reciprocal decisions are independent.** Pearson(dec, dec_o) = −0.047. Predicting one direction does not predict the other. A one-sided model will over-recommend popular people to everyone and cannot improve mutual match rates. This is the core empirical justification for a reciprocal recommender architecture.

2. **Popularity is extremely concentrated.** The top quintile of participants receives yes on 77.5% of their dates; the bottom quintile receives yes on only 11.7%. Mean attractiveness received correlates r = 0.79 with yes-rate received. A pure popularity baseline (rank everyone by their average `attr_o`) is the hardest benchmark to beat on raw mutual-match@k, and must be the primary comparison model.

3. **Attractiveness dominates decisions.** Correlations with `dec`: attr (+0.49) > like (+0.51 — but `like` is partly circular) > fun (+0.41) > shar (+0.40) > intel (+0.22) > sinc (+0.21) > amb (+0.18). The stated preference surveys show people systematically understate how much attractiveness drives their decisions and overstate sincerity/intelligence.

4. **Self-perception is weakly calibrated.** People rate themselves ~1 point higher than they are rated by others across all attributes. Correlation between self-rated attractiveness and received attractiveness is only r = +0.29. Self-ratings are available at prediction time (pre-event) but should be used with the knowledge that they are systematically inflated.

5. **First date gets a large premium.** Order 1 has a yes-rate of 49.9% vs an average of ~42% overall, declining to ~39% late in the evening. This is a nuisance variable to control for, not a modeling signal.

6. **Age difference is a clean pair-level feature.** Match rate declines monotonically from 20.4% (same age) to 8.0% (11+ years apart). Derive `age_diff = |age - age_o|` and include as a feature.

7. **Shared race barely matters.** Match rate is 17.1% same-race vs 16.1% different-race — a 1 percentage point gap. Despite the original paper's focus on racial preferences, race is not a strong predictor of matching in this data. The dataset is also highly non-representative racially (55% White, 25% Asian).

8. **Interest correlation (`int_corr`) is nearly useless as a predictor.** r(int_corr, dec) = +0.019 and r(int_corr, match) = +0.031. Theoretically appealing, empirically negligible. Include but do not expect it to carry much weight.

9. **Wave effects are large and real.** Match rate ranges from 8.3% to 31.0% across waves. Any train/test split must be wave-based (hold out complete waves), not row-based, to prevent information leakage and to respect the event structure.

10. **Follow-up behavior is largely missing but valuable as ground truth.** 52% of participants filled out the Time 3 survey reporting real-world date behavior. This could serve as an extension validation layer: did the recommender's matches lead to actual dates? This is the X-factor for the project.

---

*End of EDA Report*
