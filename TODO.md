4.1 Background
Traditional e-commerce search engines heavily rely on static keyword matching, failing to capture the fluid shifts of genuine consumer psychology and the distinction between open-ended browsing and high-intent buying. In modern conversational commerce, constructing an intelligent agent that leverages dynamic context programming is critical to bridging the gap between ambiguous user queries and complex product catalogs. Solving this challenge directly impacts core industrial metrics.
4.2 Problem Statement
Participants are challenged to architect an intelligent, next-generation shopping agent capable of navigating real-world customer dynamics. Moving beyond rigid search filters, the engineered system must demonstrate deep cognitive understanding, runtime architectural agility, and commercial efficiency using the provided Amazon dataset.
Specifically, the system should be built upon the following four core pillars:
I. Core Architecture: Intent Routing & Hybrid Pipeline
- Dual-Track Routing: Instantly detect the user's underlying intent—triggering a high-precision filter track for targeted "Buying" to lock hard constraints, and a diverse dense retrieval track for open-ended "Browsing" to unlock cross-category scenario matching.
- Pipeline Base: Construct an in-memory data stream featuring "Multi-Route Retrieval $$\rightarrow$$ LLM Semantic Ranking" (combining keyword, category, and vector similarity).
II. Dialog Strategy: Multi-Turn Scenario Evolution
- Dynamic State Machine: Build a robust conversational state tracker to gracefully handle dynamic Information Accumulation (incremental slots) and abrupt Intent Override (slot erasure and rewriting).
- Proactive Guidance: Trigger an immediate retrieval cutoff when facing Over-Generality (candidate pool overload) to actively generate structured, proactive clarification prompts that guide user convergence.
III. Self-Evolution: Dynamic Context Programming
- Runtime Adaptation: Leverage accumulated dialog history to perform Personalized Context Distillation, continuously updating short-term session states and long-term user profiles.
- Adaptive Orchestration: Utilize dynamic Context Programming to achieve runtime workflow re-orchestration and strategy alignment, ensuring the agent iteratively refines its own guidance logic.
IV. Evaluation Matrix: Product & Efficiency Metrics
Anchored on the final purchased record within the Amazon dataset, performance is quantified across three dimensions:
- Coverage (Hit Rate@K): Measures the catalog recall and boundary capability during the retrieval stage.
- Precision (MRR / Top-K Hit Rate): Evaluates the LLM's accuracy in pushing the exact purchased item to the absolute top of the recommendation list.
- Efficiency (MTTC - Mean Turns to Conversion): Heavy rewards systems that guide the user to the correct product in fewer interaction rounds, penalizing unnecessary conversational cognitive load.
4.3 Constraints & Scope
Category
Constraints & Scope Details
In scope
Designing highly sensitive intent-detection modules to split traffic into "Buying" and "Browsing" tracks.
Implementing heterogeneous retrieval routing (weights, custom dynamic truncation, and slot decay over time).
Engineering runtime-adaptive memory layers for personalized context distillation.
Fine-tuning prompt strategies or local scoring logic for the LLM ranking stage to compress decision paths.
Out of scope
UI/UX Development (evaluated purely via automated backend APIs and headless pipelines).
Training or full-parameter fine-tuning of base foundational LLMs.
Deploying heavy external industrial vector DB clusters (must run entirely in-memory for light execution).
Multi-Modal Processing (restricted strictly to text catalogs, structured metadata, and text dialogs).
Limits
Max Turns: Hard limit of 10 turns per session (forced termination and zero score if exceeded).
Catalog Mutation: The Amazon product dataset is strictly read-only; no structural mutations or mock ASIN injections are allowed.
Allowed assumptions
Inputs are pre-cleaned text strings (no need to account for spelling correction, typos, or ASR noise).
Product catalog, pricing, and category trees are static for the duration of the hackathon.
Each session is simulated as an isolated single-user interaction (no multi-user concurrency stress needed).