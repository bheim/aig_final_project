# Limitations

## 1. GPT-3.5 adoption may not predict GPT-4o adoption patterns

**Problem:** Our treatment intensity is based on field-level marker word adoption during the GPT-3.5 era (Jan–Oct 2022 vs. Apr 2023–Apr 2024). There is no guarantee that fields which adopted GPT-3.5 most heavily will also be the heaviest adopters of GPT-4o. Adoption patterns could shift as the technology matures and different fields discover new use cases.

**Response:** The propensity measure captures something structural about fields — how amenable their writing and research workflows are to LLM assistance — rather than adoption of a specific model. Fields like Computer Science that showed high marker word uptake under GPT-3.5 did so because their workflows are inherently receptive to AI tools, and there is little reason to expect that structural openness to reverse under a newer model. Furthermore, the propensity ratio is measured during a period when there was no reputational cost to exhibiting AI-influenced writing, making it a less contaminated signal of genuine adoption intensity. We attempted a direct first-stage validation (testing whether GPT-3.5 propensity predicts post-4o marker word rates) and found a null result, but this is explained by researchers actively editing out AI-sounding language as awareness of marker words grew — which actually reinforces the value of the pre-gaming propensity measure.

## 2. Concurrent release of other LLMs confounds the GPT-4o effect

**Problem:** GPT-4o was not the only major LLM released during our study period. Claude, Gemini, Llama, and other models all improved substantially over the same timeframe. We cannot attribute changes in citation behavior specifically to GPT-4o as opposed to the broader ecosystem of increasingly capable models.

**Response:** Our identification strategy does not require isolating the effect of GPT-4o specifically. The November 2024 cutoff (reflecting a 6-month publication lag from GPT-4o's May 2024 release) marks a step change in the availability of free, high-quality LLM tools, with GPT-4o as the most prominent trigger. The identifying variation comes from cross-field differences in AI receptivity interacted with this post-period indicator — fields structurally more open to LLMs would adopt any sufficiently capable model more readily. Our estimand is therefore best interpreted as the effect of crossing a broad AI capability and accessibility threshold, not the effect of one product. This is arguably a more policy-relevant quantity than a single-model effect.
