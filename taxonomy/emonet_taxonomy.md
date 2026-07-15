# EmoNet Emotion Taxonomy (40 categories)

The **EmoNet** emotion taxonomy defines **40 fine-grained emotion categories**. Each category
is an umbrella for a **cluster of near-synonymous affect terms** — the cluster both *defines*
the category's meaning and served as the label vocabulary during data annotation. The taxonomy
is deliberately broader than the classic "basic six": it includes appetitive, epistemic,
social, and low-arousal/altered states that matter for real speech.

This taxonomy is the target space of LAION's **Empathic-Insight-Voice** speech-emotion models.
The **[Empathic-Insight-Voice-Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus)**
model predicts **all 40 of these EmoNet emotions plus two continuous affect dimensions —
Valence and Arousal — for 42 emotion/affect outputs in total**, alongside additional voice
attribute and audio-quality scores (see
[`../guides/emonet_empathic_insight.md`](../guides/emonet_empathic_insight.md)).

> **EmoNet vs. VoiceNet.** EmoNet answers *which emotion is being expressed*; the
> [VoiceNet taxonomy](voicenet_taxonomy.md) answers *how the speech is delivered* (57
> perceptual voice/prosody dimensions). They are complementary layers of the same stack.

---

## Summary

- **Categories:** 40
- **Each category:** one canonical name + a synonym cluster
- **Predicted by Empathic-Insight-Voice-Plus:** these 40 categories **+ Valence + Arousal** (42 total)
- **Source:** LAION EmoNet / EMONET-VOICE line of work

---

## Positive / Approach

| # | Emotion category | Synonym cluster |
|--:|---|---|
| 1 | **Amusement** | lighthearted fun, amusement, mirth, joviality, laughter, playfulness, silliness, jesting |
| 2 | **Elation** | happiness, excitement, joy, exhilaration, delight, jubilation, bliss, cheerfulness |
| 3 | **Pleasure/Ecstasy** | ecstasy, pleasure, bliss, rapture, beatitude |
| 4 | **Contentment** | contentment, relaxation, peacefulness, calmness, satisfaction, ease, serenity, fulfillment, gladness, lightness, tranquility |
| 5 | **Thankfulness/Gratitude** | thankfulness, gratitude, appreciation, gratefulness |
| 6 | **Affection** | sympathy, compassion, warmth, trust, caring, clemency, forgiveness, devotion, tenderness, reverence |
| 7 | **Infatuation** | infatuation, having a crush, romantic desire, fondness, butterflies in the stomach, adoration |
| 8 | **Hope/Optimism** | hope, enthusiasm, optimism, anticipation, courage, encouragement, zeal, fervor, inspiration, determination |
| 9 | **Triumph** | triumph, superiority |
| 10 | **Pride** | pride, dignity, self-confidently, honor, self-consciousness |
| 11 | **Interest** | interest, fascination, curiosity, intrigue |
| 12 | **Awe** | awe, awestruck, wonder |

## Cognitive / Epistemic

| # | Emotion category | Synonym cluster |
|--:|---|---|
| 13 | **Astonishment/Surprise** | astonishment, surprise, amazement, shock, startlement |
| 14 | **Concentration** | concentration, deep focus, engrossment, absorption, attention |
| 15 | **Contemplation** | contemplation, thoughtfulness, pondering, reflection, meditation, brooding, pensiveness |
| 16 | **Relief** | relief, respite, alleviation, solace, comfort, liberation |
| 17 | **Longing** | yearning, longing, pining, wistfulness, nostalgia, craving, desire, envy, homesickness, saudade |
| 18 | **Teasing** | teasing, bantering, mocking playfully, ribbing, provoking lightly |
| 19 | **Doubt** | doubt, distrust, suspicion, skepticism, uncertainty, pessimism |
| 20 | **Confusion** | confusion, bewilderment, flabbergasted, disorientation, perplexity |

## Negative / Distress

| # | Emotion category | Synonym cluster |
|--:|---|---|
| 21 | **Impatience and Irritability** | impatience, irritability, irritation, restlessness, short-temperedness, exasperation |
| 22 | **Fear** | fear, terror, dread, apprehension, alarm, horror, panic, nervousness |
| 23 | **Distress** | worry, anxiety, unease, anguish, trepidation, concern, upset, pessimism, foreboding |
| 24 | **Embarrassment** | embarrassment, shyness, mortification, discomfiture, awkwardness, self-consciousness |
| 25 | **Shame** | shame, guilt, remorse, humiliation, contrition |
| 26 | **Disappointment** | disappointment, regret, dismay, letdown, chagrin |
| 27 | **Sadness** | sadness, sorrow, grief, melancholy, dejection, despair, self-pity, sullenness, heartache, mournfulness, misery |
| 28 | **Bitterness** | resentment, acrimony, bitterness, cynicism, rancor |
| 29 | **Helplessness** | helplessness, powerlessness, desperation, submission |
| 30 | **Pain** | physical pain, suffering, torment, ache, agony |

## Hostile / Aversive

| # | Emotion category | Synonym cluster |
|--:|---|---|
| 31 | **Contempt** | contempt, disapproval, scorn, disdain, loathing, detestation |
| 32 | **Disgust** | disgust, revulsion, repulsion, abhorrence, loathing |
| 33 | **Anger** | anger, rage, fury, hate, irascibility, enragement, vexation, wrath, peevishness, annoyance |
| 34 | **Malevolence/Malice** | spite, sadism, malevolence, malice, desire to harm, schadenfreude |
| 35 | **Sourness** | sourness, tartness, acidity, acerbity, sharpness |
| 36 | **Jealousy & Envy** | jealousy, envy, covetousness |

## Low-arousal / Altered / Appetitive

| # | Emotion category | Synonym cluster |
|--:|---|---|
| 37 | **Fatigue/Exhaustion** | fatigue, exhaustion, weariness, lethargy, burnout |
| 38 | **Emotional Numbness** | numbness, detachment, insensitivity, emotional blunting, apathy, existential void, boredom, stoicism, indifference |
| 39 | **Intoxication/Altered States** | being drunk, stupor, intoxication, disorientation, altered perception |
| 40 | **Sexual Lust** | sexual lust, carnal desire, lust, feeling horny, feeling turned on |

---

## Machine-readable form

The taxonomy is also available as a Python dict / JSON mapping each category to its synonym
cluster (`EMOTIONS = { "Amusement": ["lighthearted fun", "amusement", ...], ... }`). The
canonical category names above are the keys; the lists are the clusters.

