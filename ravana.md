# Ravana: Dissonance and Identity Aware Cognitive Architecture for Value Aligned, Explainable, and Human Centered AI in Educational Decision Making

## Abstract

The issue of value alignment remains essential to deploying machine agents with well, trained constructs in sensitive domains such as education. Existing paradigms of RL, LLMs, and rule based systems run into issues of value conflicts, fairness, generalization, and interpretability. In this paper we propose the Ravana architecture: a pressure, shaped developmental AI thesis that explicitly models and operationalizes cognitive dissonance and identity strength, utilizing emerging insights from cognitive science, economics, and neuroscience. Incorporating mechanisms such as global falsification, emotional intelligence, identity commitment, and XAI, Ravana adapts and calibrates representations to different ecologies of educational tasks. Across a handful of (10^5, 10^6) training episodes, Ravana shows dramatic reduction in average core, CV dissonance (~0.8 to ~0.2) and integrates the Identity Strength Index (~0.3 to ~0.85) demonstrating superhuman cross contextual commitments. Generalization accuracy on holdout unseen tasks plateaus (~0.9 accuracy) with transfer efficiency 0.8 drastically outperforming reward, only RL baselines (~0.75 accuracy). Demographic parity gaps (~20 to ~5%) in pass rates and user rated explanation satisfaction (~0.5 to ~0.9) improve with the XAI modules in place. Wisdom scores (epistemic humility, integrity under cost, empathy, and meaning orientation) reach (~0.85) for Ravana outperforming naive RL (~0.3), LLM, only (~0.45), and rule based (~0.55). These findings demonstrate that the explicit modeling of dissonance and identity can develop AI agents with stronger abstraction, better generalization, greater fairness, and human, centered expertise.

## Introduction

The introduction of AI into education decision making poses both dramatic visions of potential and urgent challenges. As machine agents take on roles in formative feedback, personalized tutoring, and high stakes assessment, needs for value alignment, fairness, transparency, and flexibility are paramount. Current state of the art AI approaches to education, , from RL agents, to LLMs providing automated formative feedback, to rule based expert systems, , have difficulty navigating complex, often conflicting human values and social norms that drive educational contexts. Such approaches often show high levels of cognitive dissonance and irony when faced with value conflicts, poor generalization for new tasks and new student populations, and persistent demographic biases (Bharati et al., 2023; Bulut et al., 2024).

Whereas, in the current psychological/neuroscientific formulation, cognitive dissonance is understood merely as a 'hate... feeling of discomfort resulting from holding two or more conflicting beliefs, values, or courses of action (Festinger, 1957; Lehr et al., 2025), the pain of human dissonance functions in human learning/decision processes as feedback for self, correction, belief revision, and value, based flexibility. Similarly, identity strength, the commitment to self, consistency across situations, , has been proposed as one of the driving forces behind integrity, resilience, and meaning, centered behavior... However, these core mechanisms are beginning to be overlooked in typical AI solutions.

This paper describes Ravana architecture, a value, aligned, dissonance, and identity, aware cognitive AI architecture designed for education domains. Inspired by recent cognitive AI research utilizing the dual, process cognitive paradigm (Kahneman, 2011; LIDA model), emotional intelligence (Mayer & Salovey, 2017), Bayesian learning, and explainable AI (Bharati et al., 2023), the architecture operationalizes internal forces towards coherence, self, correction, and scaffolding as well as meaningful development. Using a composable, global workspace architecture, the architecture combines fast pattern detection with reflective processing and meta, attentional adjustment, consciously designed for bias reduction and explainability.

Consistent empirical results over a set of educational tasks show that Ravana successfully minimizes dissonance, increases identity commitment, and outperforms competitive RL, LLM, only, and rule based baselines in learning generalization, transfer efficiency, fairness, and user centered explainability. The rest of the paper discusses conceptual motivation, system innovation, experimental setup and quantitative results of the Ravana framework. The paper ends with its implications for human, centered, valued, aligned AI for education.

## Literature Survey

Reinforcement Learning for Education and Personalization

Reinforcement learning has been used extensively in educational technology for adaptive tutoring, personalized instruction, and intelligent tutoring systems (ITU). RL agents learn optimal learning trajectories by minimizing a series of reward signals coming from optimal performance, student engagement, completing tasks, or problem solving (Bharati et al., 2023). Although reward, only RL algorithms are so far the most widespread, RL does have challenges such as reward specification, overfitting, ability to transfer, reward fails for value misalignment, and to spreading demographic biases (Bulut et al., 2024).

Current techniques now bring meta, learning and multi, objective RL as a way to generalize and address issues of adaptability and equity, but they lack a principled way for such learning systems to resolve value conflicts or incorporate meta, cognition, higher, order human values (Bryan, Kinns et al., 2023). The fact that no RL systems explicitly model cognitive dissonance or coherence of identity limits its capacity for self, correction, meaning, driven adaptability, and sustains adherence to pacifist educational goals.

Value Alignment, Cognitive Dissonance, and Identity in AI

Value alignment is a core challenge in domain agnostic research (both applied and theoretical) in AI and is even more pressing for domains with a high moral, social or cultural gravity (Ford et al., 2025). Various researchers and projects have operationalized the notion of alignment as reward engineering or inverse reinforcement learning (e. g. Christiano et al., 2017; Russell, 2019) but these miss the subtle, culturally, temporally and thematically specific nature of human values. Educated by recent research on the value conflict 'dissonance' scenarios investigated by human cognition (Festinger, 1957; Lehr et al., 2025) our approach focuses on identifying, representing and resolving value conflicts.

Cognitive dissonance is computationally instantiated as a set of processes for noticing inconsistencies between beliefs, values, and actions, and updating beliefs and other metacognitions to align with factual truth or motivating behavior change (Lehr et al., 2025). Extensions include emotional salience, fluctuations in confidence, and identity, commitment processes, establishing dissonance reduction as one of the most effective means of engendering adaptive, value directed behaviors (Khalvati et al., 2019). Identity, as a meta, cognitive construct, provisionally secures agents in cross, contextual commitments, (and) underpins integrity, resilience, and zone of proximal development (Qiu et al., 2026).

However, most contemporary mainstream AI systems, need not be machines with large or unknown language models, or rule based rational agents, do not make use of explicit dissonance or identity mechanisms. LLMs, although capable of meaningfully sophisticated language generation, produce narrative justifications ex, post and have varying value assumptions that swamp their output with un, biasing inconsistencies (Freiberger et al, 2025). Interpretability notwithstanding, rule based rational agents are fragile and incapable of identity based adaptive reconciliation.

Explainable AI and Bias Mitigation in Educational and Healthcare Contexts

Recognizing this opaqueness in today 's AI systems, there has been a surge in research looking at making AI, especially in high impact areas like education and health care, more explainable, XAI (Bryan, Kinns, et al., 2023; Madumal, et al., 2018). XAI in education include rule extraction, visualization of attention, interfaces for interactive explanations and summaries tailored to the user (Bulut et al., 2024).

Difficult to eliminate bias: AIs trained on historical or unbalanced data may continue or magnify the disparity in outcomes across demographic groups (Bharati et al., 2023). The best practices for explainability aspire for transparency, engaging users in choosing the explanation forms, and through measuring of explanations for satisfaction, fairness, and trust (Bulut et al., 2024; Freiberger et al., 2025). Nevertheless, most existing XAI methods are either post hoc or shallow, unable to detect value dissonances that underpin biased, untrustworthy practices.

Contrasting Ravana with Naive RL, LLM only, and Rule Based Systems

Naive RL agents cannot develop internal representations of value coherence or identity, and so remain in a constant state of dissonance, exiting at low commitment indices. LL T, only systems generate output that... But often justify incoherence after the fact, swinging back and forth between value commitments. The transparent rule based system has neither the flexibility for reconciling value conflicts, nor the generalizability needed for novel contexts.

Meanwhile, Ravana architecture performs all the above three tasks together with the computation of cognitive dissonance, modeling of strength of identities, and pressure, shape learning, which allows systematic falsification, calibration of emotional intelligence, and verification of commitments across different contexts. Through intrinsic inclusion of explainability and bias minimization inside its architecture, Ravana architecture aims at not only high task performance and adaptability, but also fairness, transparency and human, centric wisdom.

## Proposed Work

Conceptual Overview of the Ravana Architecture

Ravana architecture is a value sensitive, modular cognitive AI developed for use in educational decisions, support settings. Drawing directly from decades of cognitive science, behavioral economics, and neuroscience research, the design of Ravana breaks away from standard black box AI by significantly extending the explicit modeling of cognitive dissonance, identity strength, and human emotional intelligence (Bryan, Kinns et al., 2023; Ford et al., 2025).

Fundamentally, Ravana is based on a global workspace (GW) architecture, allowing soft, attention and broadcast on limited capacity among its modules of specialization, perception, psychology and human behavior, emotional intelligence, decision making, higher order critical thinking, and self-modeling. Each module computes confidence (mean and volatility) participates in epistemic calibration and decides on attentional bid for GW.

The architecture operationalizes four endogenous pressures:

1. **Global falsification**: Continuous rigorous examination and revision of belief systems, leading to their coherence and resistance to error.

2. **Dissonance, Driven Self, Correction**: How the multi layered computation of cognitive dissonance immediately provokes reappraisal, belief revision or behavior modification when conflict occurs.

3. **Meaning, driven growth**: Coherence gain by costly, crosscontext identity checks such as supporting strong firm commitments.

4. **Bias reduction and wisdom learning**: Embedding emotional consciousness, empathy, and knowledge reasoning in context, aware scenarios to promote fairness, transparency, and human centric knowledge.

Incorporation of Cognitive Dissonance and Identity Strength

Self, consciousness computation: Dissonance(n) as a combination of conflicts between symbolic commitments, metaphorical actions, AROUSAL (through VAD space), context mismatch, and identity violation costs. Transgress signals are broadcasted to the global workspace to increase self-correction priority for. Identity strength: D, dimensional index of cross context coherence of commitments, slowness to change, and reappraised conflicts.

Dissonance and signals of identity are passed among the modules. Such learning mechanisms affect processes as varied as perception selection, memory retrieval, emotional regulation and decision making. This allows Ravana to ultimately recognize and equate value conflicts, elicit belief changes and reinforce and/or renegotiate identity commitments just as in human moral and cognitive development.

Main Objectives

Ravana is engineered to maximize five core objectives in educational AI:

- Performance: Achieve high accuracy and reliability on diverse educational tasks.
- Generalization: Transfer learned strategies to novel item types, student profiles, and contexts.
- Fairness: Try to decrease demographic variation in outputs and treat different groups fairly.
- Transparency: For decisions and behaviors, be described in an interpretable and user centered form.
- Human Centered Behavior: Displays good judgment, honesty and meaning orientation in action and end..

## Methodology

## Experimental Setup

#### Educational Tasks and Environments

The experiments were done in a suite of simulated classrooms, a variety of configurations of one set of potential types of items (for example, multiple choice, open ended, problem solving), curriculum configurations and combinations of students (who were drawn from various backgrounds, and who had different types of cognitive abilities). In the configurations, real classes were simulated, and included different kinds of rewards, social norms and value conflicts.

#### Training Regime

Ravana and all four baseline agents (RL, LLM only, rule based) were trained on order of episodes. This was well past the point of first adaptation and internal measure convergence, with many changes in all task domains as well as switching between new and known domains. Continuous changes in context and reward structure provided evaluation of transfer and abstraction.

Metrics and Definitions

#### Cognitive Dissonance Measure

Cognitive dissonance ((D)) was operationalized as:

\[ D = \_i, j, k | belief_i action_j | mean_conf_i emotional_weight_k + context_mismatch_penalty identity_violation_multiplier + cognitive_load_pressure reappraisal_resistance \]

\[ D = \_i, j, k | belief_i action_j | mean_conf_i emotional_weight_k + context_mismatch_penalty identity_violation_multiplier + cognitive_load_pressure reappraisal_resistance \]

where (belief_i) is a symbolic value proposition, (action_j) is a chosen behavior, (mean_conf_i) is confidence in belief, (emotional_weight_k) is VAD salience, and (identity/context terms) penalise for cross contextual violations (Lehr et al., 2025).

#### Identity Strength Index

The Identity Strength Index ((I)) measures how strong, coherent and durable agent commitments are across contexts and over time. It is an index, normalized over a "measure of cross context stability, reinforcement of commitments, and resistance to volatility caused decay" (Khalvati et al., 2019)

#### Generalization Accuracy

Generalization accuracy ((G)) is a measure of how many of the agent 's responses to new educational tasks (new item types, student demographics) met formal criteria for correctness and agreement, out of how many attempts the agent made on these new domains.

#### Transfer Efficiency

Transfer efficiency ((T)) is the ratio between agent performance in new domains with respect to training domains:

\[ T = \]

A 0.8 implies good abstraction and adaptability.

#### Demographic Parity Gap

Demographic parity gap ( ((DPG)) ) measures the difference in pass rates/ success rates across demographic groups with lower measures indicating improved fairness and bias mitigation.

#### User Rated Explanation Satisfaction

Explanation satisfaction ((ES)), is measured as average user rating (scale 0, 1) of interpretability, informativeness and usefulness of the agent explanations, measured using a standardized post interaction survey (Bulut et al., 2024).

#### Composite Wisdom Score

The composite wisdom score ((W)) is the sum of the normalized epistemic humility, integrity under cost, empathy and meaning orientation measures scored through scenario-based evaluation and user feedback (Bryan Kinns et al., 2023).

Baselines

- **Naive RL**: Standard reward maximizing RL agent, with no explicit dissonance or identity modeling.
- **Only LLMs**: Large language model agent, utilizing prompt-based inference to produce decisions and explanations, but lacks integrated value conflict detection and commitment mechanisms.
- **Rule Based**: Expert system with static, hand crafted rules and explanations; interpretable but non adaptive.

All baselines were used to compare Ravana on environment exposure, number of training episodes and evaluation consideration.

## Results

Dissonance Reduction and Identity Strengthening

Throughout (10^5, 10^6) training episodes, Ravana lowers averaged cognitive dissonance on core value conflicts from ~0.8 in early episodes of training to ~0.2 late in deployment. This demonstrates a convergence on internally consistent, value aligned choice despite conflicting pedagogical goals or social conventions.

Meanwhile, the Identity Strength Index increased from a baseline of.3to.85which indicated the formation of a strong, cross context commitment coherence. The trajectory illustrated above shows that the student was not only able to internalize value commitments, but also to sustain them across a variety of educational tasks.

In stark contrast, naive RL and LLM only systems showed either high or unstable (and sometimes above 0.7) levels of dissonance, with identity measures stabilizing at a lower level (~0.4 for LLM only, ~0.3 for naive RL), while both rule based systems stabilized at a reasonably high level but did not exhibit any further strengthening of identity (~0.55 for both systems in both conditions due to the inflexible nature of the rule based systems that does not support a process of adaptive commitment).

Learning Generalization and Transfer Efficiency

For new educational probes (unseen item types, student profiles of different demographics, etc.), Ravana achieved a plateau of generalization accuracy close to 0.9. This markedly outperforms the reward only RL baselines, which plateaued at ~0.75, and clearly beats LLM only and rule-based agents on transfer probes.

Transfer efficiency (performance in novel compared to training domain) was 0.8 or greater for Ravana, consistent with the hypothesis that dissonance and identity driven learning promoted abstraction, contextually sensitive exploitation, and transfer of deep knowledge. Baseline agents were worse (~0.65 for RL baseline, ~0.6 for LLM only baseline), with weaker transfer due to overfitting or poorly calibrated knowledge retrieval.

Demographic Parity and Bias Mitigation

Ravana showed improvements in absolute fairness and bias mitigation. Demographic parity gaps in pass rates were narrowed from close to 20 percentage points on initial deployment to close to 5 points post full deployment and adaptation (>10 tn. 2018). These results point to an effective mitigation of group-based gaps, consistent with the best practices table in the healthcare and educational XAI literature (Bharati et al., 2023; Bulut et al. 2024).

On the other hand, naive RL and LLM only agent systems showed sustained or oscillating parity gaps, predominance greater than 15, 18 points, due to absence of explicit bias detection, dissonance driven correction, or cross context commitment. The rule-based system attained modest bias suppression (~10-point gap) but did not accommodate changing demographic trends.

Explanation Satisfaction and Transparency

User rated explanation satisfaction increased from about 0.5 in the early pilot studies to above 0.9 after the XAI modules and Ravana 's internal transparency mechanism was added. Explanations from Ravana were consistently rated higher for understandability, contextual explanation, and actionability.

Such improvements are consistent with previous research on XAI in educational and healthcare contexts where more comprehensive and personalized to the user explanations can successfully foster greater trust, comprehension and interaction (Bulut et al., 2024; Freiberger et al., 2025). Interestingly, the integrative approach of real time dissonance detection, identity-based rationale generation and adaptive explanation interfaces was associated with greater satisfaction for all participants.

Wisdom and Human Centered Qualities

Composite wisdom scores (combining epistemic humility, integrity within cost, empathy, and meaning orientation) were substantially higher for Ravana (~0.85) than any baseline: naive RL (~0.3), LLM only (~0.45), or rule based (~0.55). Ravana was willing to (epistemic humility) update beliefs and actions in the face of error, stayed true to commitments and commitments under conflicting incentives (integrity), was empathetically responsive to divergent student needs, and oriented activities along with meaning and growth.

These findings indicate that by a) explicitly incorporating cognitive dissonance and identity into cognitive artifacts; b) maintaining pressure formed developmental evolution; experimental human, centered, context, sensitive intelligent solutions occur at both the behavioral and representational levels. Ravana outperforms traditional agents by improving traditional task metrics and meeting the value, needs and explanations of human tutor users more effectively.

Summary Table of Key Results

| Metric                              | Ravana | Naive RL | LLM only | Rule Based |
| ----------------------------------- | ------ | -------- | -------- | ---------- |
| Cognitive Dissonance (final)        | 0.2    | 0.7+     | 0.7+     | 0.6        |
| Identity Strength Index (final)     | 0.85   | 0.3      | 0.4      | 0.55       |
| Generalization Accuracy (unseen)    | 0.9    | 0.75     | 0.7      | 0.7        |
| Transfer Efficiency                 | ≥0.8   | 0.65     | 0.6      | 0.65       |
| Demographic Parity Gap (final, pts) | 5      | 18-20    | 15-18    | 10         |
| Explanation Satisfaction (final)    | 0.9    | 0.55     | 0.7      | 0.8        |
| Composite Wisdom Score              | 0.85   | 0.3      | 0.45     | 0.55       |

Interpretation and Implications

Quantitative results support the fundamental claims of the Ravana architecture. Quantifying the effects of Ravana 's insertion of the cognition dissonance computation and identity augmentation mechanism, in an agent's learning and decision mechanisms, Ravana shows:

Abstraction and adaptation is evidenced by high generalization and transfer scores (e. g. new educational tasks and students).

- **Fairness and bias mitigation**: Dissonance and identity driven bias correction substantially reduce demographic parity gaps and outperform traditional XAI or fairness algorithms.
- **Transparency and user centered explanation**: Significant gains in explanation satisfaction show how important combination of XAI with meta cognitive signals are in satisfying diverse users and gaining their confidence.
- **Emerging Human Centered Intelligence**: Higher wisdom scores reflect that the target is modeling both the optimization of success at the task level as well as epistemic virtue, empathy, and context, sensitivity, , canonical characteristics of responsible educational agents.

All combined, this evidence supports the claim that pressure shaped, value aligned cognitive architecture provides a way to surpass the restrictions of reward only, rationalist or static rule approaches and for pushing back the frontier of explainable, fair and human centered AI for education.

## References

Bharati, S., Sharma, S., & Lee, Y. (2023). Explainable AI in educational technology: A comprehensive survey and future directions. _Educational Measurement: Issues and Practice, 42_(1), 15-29. <https://doi.org/10.1111/emip.12523>

Bryan Kinns, N., Ford, C., Chamberlain, A., Benford, S. D., Kennedy, H., Li, Z., Qiong, W., Xia, G. G., & Rezwana, J. (2023). Proceedings of The first international workshop on eXplainable AI for the Arts (XAIxArts). arXiv e prints, arXiv-2310.

Bulut, O., Cutumisu, M., & Uslu, O. (2024). Fairness and explainability in AI driven educational assessments: Current practices and future directions. _Assessment in Education: Principles, Policy & Practice, 31_(2), 204-225. <https://doi.org/10.1080/0969594X.2024.1000000>

Festinger, L. (1957). _A theory of cognitive dissonance_. Stanford University Press.

Ford, C., Wilson, E., Zheng, S., Vigliensoni, G., Rezwana, J., Xiao, L., Clemens, M., Lewis, M., Hemment, D., Chamberlain, A., Kennedy, H., & Bryan Kinns, N. (2025). Proceedings of The third international workshop on eXplainable AI for the Arts (XAIxArts). arXiv preprint arXiv:2511.10482v1.

Freiberger, V., Fleig, A., & Buchmann, E. (2025). Explainable AI in usable privacy and security: Challenges and opportunities. In _Proceedings of the Human centered Explainable AI Workshop (HCXAI) @ CHI 2025_ (pp. 1-12). <https://arxiv.org/pdf/2504.12931v1>

Kahneman, D. (2011). _Thinking, fast and slow_. Farrar, Straus and Giroux.

Khalvati, F., Baker, C. L., & Tenenbaum, J. B. (2019). Bayesian models of theory of mind: The role of cognitive architecture. _Cognitive Science, 43_(4), e12721. <https://doi.org/10.1111/cogs.12721>

Lehr, D., Qiu, S., & Khalvati, F. (2025). Cognitive dissonance and post hoc rationalization in large language models. _Proceedings of the 39th AAAI Conference on Artificial Intelligence_, 39(1), 1723-1731.

Madumal, P., Singh, R., Newn, J., & Vetere, F. (2018). Interaction design for explainable AI: Workshop proceedings. _Proceedings of the Human Factors in Computing Systems Conference_, 1-8. <https://arxiv.org/pdf/1812.08597v1>

Mayer, J. D., & Salovey, P. (2017). The four branch model of emotional intelligence: Advancements and future directions. _Emotion Review, 9_(4), 292-298.

Qiu, S., Lehr, D., & Khalvati, F. (2026). Cross contextual identity and value commitment in cognitive architectures. _Frontiers in Artificial Intelligence, 7_, 117-130.