# V40 Training Gaps (observed during V39 app testing, 2026-05-06)

## 1. Direction confusion — lowest FP rate

**Query:** `Which rule has the lowest FP rate?`
**Bad output:** Returns the rule with the HIGHEST FP rate (same direction error as "highest SAR" in V38).
**Expected:** Call `list_rules`, identify the rule with minimum FP count or minimum FP%, name it and state the value.
**Note:** "lowest FP rate" = fewest false positives = highest precision. Model must distinguish min vs max direction.
**Training group:** ~4 examples covering "lowest FP", "fewest false positives", "best precision rule", "least alerts that are false".

---

## 2. Precision synthesis — highest precision rule

**Query:** `Which rule has the highest precision rate?`
**Bad output:** Calls `list_rules` but generates no follow-up answer (empty response after tool call). OR returns wrong rule.
**Expected:** Call `list_rules`, compute precision = TP/(TP+FP) per rule from the tool result, name the winner with the value.
**Note:** Precision is not a pre-computed column — model must derive it from TP and FP counts in the tool result. Do NOT hallucinate; only compute from verbatim tool output.
**Training group:** ~4 examples covering "highest precision", "most precise rule", "lowest FP rate" (synonym), "best true positive ratio".

---

## 3. Multi-turn data isolation — do not echo prior tool results

**Query:** `Show SAR backtest for Activity Deviation ACH rule` (issued after a 2D sweep for Elder Abuse)
**Bad output:** Response describes Elder Abuse 2D sweep numbers (TP=30, FP=163, floor_amount=8000, age_threshold=66) — copied from conversation history — instead of calling `rule_sar_backtest` for Activity Deviation ACH.
**Expected:** Ignore prior tool result in history. Call `rule_sar_backtest` with `risk_factor="Activity Deviation ACH"`. Summarise the AD ACH result only.
**Root cause:** Model pattern-matches on prior tool result numbers in the history context and echoes them.
**Training group:** ~4 examples where history contains a prior tool result but the new query asks about a different rule/tool. Model must call the correct tool fresh and NOT reference the prior result.

---

## 4. Cluster stats hallucination — read from context, never invent

**Query:** `How about cluster 4?` (issued after a clustering response describing clusters 1-4)
**Bad output:** Model describes cluster 4 with invented statistics ($18,487 balance, 0.8 weekly transactions, etc.) that do not match the actual K-Means output shown in the sidebar drilldown table.
**Expected:** Read cluster 4 stats VERBATIM from the `[PREVIOUS CLUSTERING RESULT]` block injected in the context. Do NOT call any tool. Do NOT invent numbers.
**Note:** Rule 15 of the segmentation system prompt already states this. Model is ignoring it. Needs training reinforcement.
**Training group:** ~4 examples where `[PREVIOUS CLUSTERING RESULT]` is in context and user asks about a specific cluster. Response must quote the real stats from that block.

---

## 5. Non-AML OFAC/sanctions context → out_of_scope

**Query:** `My dog OFAC met a cat called sanctions`
**Bad output:** Routes to PolicyAgent, gives a full OFAC/sanctions compliance overview as if the query were legitimate.
**Expected:** Classify as `out_of_scope`. Return the standard OOS refusal. Do NOT treat OFAC or "sanctions" as AML keywords when used as proper names in a clearly non-AML sentence.
**Root cause:** Fine-tuned model strongly associates "OFAC" + "sanctions" → compliance query, overriding the few-shot OOS example in the classifier prompt.
**Training group:** ~4 examples where OFAC, sanctions, AML, or similar terms appear as names/objects in nonsensical non-AML sentences. Classifier label: `out_of_scope`.

---

## 6. Empty response after tool call (persistent — L01 benchmark failure)

**Query:** Any threshold tool call (e.g. `Which rule has the highest precision rate?` → calls `list_rules`)
**Bad output:** Tool executes correctly, returns results, but the model generates no follow-up text. Agent returns `(No response)` or empty string.
**Expected:** After receiving the tool result, model MUST generate at least one sentence summarising the finding.
**Note:** This may be a max_tokens or inference-side issue rather than a training issue. Investigate whether increasing `MAX_TOKENS_TOOL` or adding a forced-continuation prompt resolves it before adding training data.
**Training group:** If confirmed as a training gap — ~5 examples where tool result is provided and model must generate a non-empty summary. The training response after the tool turn must be substantive.

---

## Summary

| # | Gap | Priority | Est. examples |
|---|---|---|---|
| 1 | Lowest FP direction | High | 4 |
| 2 | Precision synthesis | High | 4 |
| 3 | Multi-turn data isolation | High | 4 |
| 4 | Cluster stats hallucination | Medium | 4 |
| 5 | Non-AML OFAC/sanctions → OOS | Low | 4 |
| 6 | Empty response after tool call | Investigate first | 5 |

**Total new examples for V40:** ~20-25 (to be combined with continued testing gaps before committing to a training run)
