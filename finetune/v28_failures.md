# V28 Training Failures (observed on V26 / to verify on V27)

## 1. "tool response" prefix in output
**Query:** Any tool call (e.g. SAR backtest)  
**Bad output:** `tool response  - Rule: Elder Abuse ...`  
**Expected:** Clean summary starting directly with the insight, no prefix.

## 2. Missing one-liner summary after rule_sar_backtest
**Query:** `Show SAR backtest for Activity Deviation ACH rule`  
**Bad output:** Chart shown but no text summary above it.  
**Expected:** One sentence describing the rule, sweep param, and key finding before the chart.

## 4. list_rules truncates to 11 of 16 rules
**Query:** `List all AML rules`  
**Bad output:** Summary mentions only 11 rules, omits 5.  
**Expected:** All 16 rules listed with correct stats copied verbatim from the pre-computed tool result.

## 5. list_rules hallucinates precision/alert numbers
**Query:** `List all AML rules`  
**Bad output:** Precision % and alert counts differ from the pre-computed table (e.g. Human Trafficking precision wrong, Structuring counts wrong).  
**Expected:** Numbers copied exactly from `list_rules_text()` output — no paraphrasing or rounding.

## 3. No response after rule_sar_backtest with cluster filter
**Query:** `Show Elder Abuse SAR backtest for Cluster 4`  
**Bad output:** `(No response)` — empty agent_text after tool executes.  
**Expected:** Tool call with `risk_factor="Elder Abuse", cluster=4`, followed by a one-liner summary of the filtered results.
