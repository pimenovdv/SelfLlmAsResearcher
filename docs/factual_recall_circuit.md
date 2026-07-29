# Factual Recall Circuit in GPT-2

## Overview
This document outlines the findings from our activation patching experiments focused on the Factual Recall task in the GPT-2 small model. The objective was to identify the specific components (layers and attention heads) responsible for retrieving and transmitting factual knowledge (e.g., retrieving "Paris" when prompted with "The capital of France is").

## Methodology
We used **Activation Patching** on individual attention heads. We ran two forward passes:
*   **Clean Run:** `The capital of France is` (Expected target: ` Paris`)
*   **Corrupted Run:** `The capital of Russia is` (Expected target: ` Moscow`)

By patching the activations from the corrupted run into the clean run at specific layers and heads, we observed how the model's output shifted. Specifically, we intercepted the input to the `c_proj` (output projection) for each attention head and patched it. If patching a specific head caused a significant drop in the logit difference (Paris - Moscow), it indicates that head plays a key role in transmitting the correct fact.

## Baseline Metrics
*   **Clean Logit Diff (Paris - Moscow):** 4.7550
*   **Corrupted Logit Diff (Paris - Moscow):** -5.2923

## Key Findings: Factual Recall Heads
Through granular attention head patching, we identified several heads that are critical for factual recall in this context. The most significant drops occurred in the later layers (Layers 8-11).

The most impactful heads are:

1.  **Layer 9, Head 8 (L9H8):** This is the most dominant head for this specific task. Patching it caused the logit difference to plummet from `4.7550` to `-1.4119` (a massive drop of `6.1669`). This head likely acts as a direct conduit for moving the retrieved factual information to the final output position.
2.  **Layer 10, Head 0 (L10H0):** Another highly significant head. Patching it resulted in a logit difference of `1.6837` (a drop of `3.0714`).
3.  **Layer 8, Head 11 (L8H11):** Shows a moderate but notable effect with a logit diff of `3.4794` (a drop of `1.2756`).
4.  **Layer 10, Head 10 (L10H10):** Logit diff dropped to `4.1675` (drop of `0.5875`).
5.  **Layer 11, Head 2 (L11H2):** Logit diff dropped to `3.9111` (drop of `0.8439`).

## Information Transmission Pathway
Based on these findings and previous MLP patching experiments (which localized the factual processing primarily at the Subject token position in the mid-to-late MLP layers), we can hypothesize the following pathway:

1.  **Subject Processing:** Early and mid-layer MLPs process the subject ("France").
2.  **Retrieval:** The factual knowledge is retrieved or strongly associated within the mid-to-late MLPs.
3.  **Transmission (The Factual Recall Heads):** Attention heads, predominantly **L9H8** and **L10H0**, attend to the subject token position and move the retrieved factual information (the concept of "Paris") to the final token position (`is`), preparing it for the final unembedding step.

## Conclusion
The Factual Recall circuit in GPT-2 is highly localized in the later layers. **Layer 9 Head 8** stands out as the primary mechanism for transmitting factual knowledge to the output in this specific task setup.
