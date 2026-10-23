# project_a

## 1. What this project is

**A learning project.** We are studying the _artificial_ aspects of AI agents - the engineered scaffolding around a model, as opposed to the model's intelligence itself. The parts a human has to design, measure, and get right: how a model is post-trained, how it is served, how its context and tools are managed, how its behavior is constrained.
We do this **with the help of AI agents**, which act as collaborators in the work - for implementation, review, investigation, and the unglamorous parts of the pipeline.
What we are actually building, as the vehicle for that study:

1.  **A post-training pipeline** - supervised fine-tuning, preference optimization, and reinforcement learning on a small model.
    
2.  **An inference engine** - a runtime that serves that model efficiently, including on low-power embedded hardware.
    

The work is deliberately first-principles: we do not delegate the core of either to `llama.cpp`, `vLLM`, `TRL`, or `Axolotl`. Where we reuse a library, we say why and what it does. The reasoning is that a framework hides exactly the parts we came here to learn.
**The distinction that shapes everything below:** a model's capability is largely given to us. The agent built on top of it is not. We are studying the second part.

* * *

## 2. Why we're doing it

**The learning objective** is the primary one. Everyone here wants to understand post-training and inference engineering well enough to build them, not just configure them. The parts we keep are the parts existing frameworks don't package: data curation, reward design, KV-cache economics, agent behavior - the artificial decisions.
**The shipping objective** is secondary but real: a quantized, efficient runtime serving a fine-tuned small model on an embedded device. It exists to keep the learning honest. A working artifact is the only reliable proof that the understanding is real.
**Non-goals**, stated plainly so nobody expects them:

*   Training a frontier model
    
*   Beating published benchmarks
    
*   Building a general-purpose serving product
    
*   Rebuilding autograd, GPU kernels, or tokenizer training from scratch
    

* * *

## 3. Where we are right now

**Stage 1, preliminary.** The plan is set; no implementation has started.
We are deliberately **not** documenting technical design yet. Model configuration, hardware targets, performance targets, and interface details are still being investigated. Those documents come later, as findings land - not as guesses now.
If you're joining, expect to help shape the design rather than implement against a finished spec.

* * *

## 4. Stage 1 scope

**In scope:** the two foundation repositories.
| Repo | Purpose |
| --- | --- |
| `infra-models` | Describes what a model is - its configuration, structure, and inputs |
| `infra-infer` | Runs a model - generation, caching, batching, sampling |

Everything else is explicitly deferred. Application repositories (the training recipes, the evaluation harness, the serving package, the agent runtime) and the remaining infrastructure repositories come in later stages, once Stage 1 proves the foundation works.
**The central deliverable of Stage 1 is not the model stack itself. It is a foundation that accepts a _second_ model without rework.** One model working proves we built something that runs. Two models working, without changing the engine, proves we built something that _generalizes_.

* * *

## 5. Repository naming

Two categories, and the naming tells you which you're in:
| Category | Naming | Meaning |
| --- | --- | --- |
| **Infrastructure** | `infra-` prefix | Depended on by other repositories. Must stay stable. |
| **Application** | no prefix | Standalone app, service, or demo. Free to change. |

The prefix is a stability promise, not just a label. Breaking an `infra-` repo is a coordination event; breaking an unprefixed one is just your afternoon.
Application repos planned for later stages: `posttrain`, `eval`, `serve`, `agents`.

* * *

## 6. Milestones

Stage 1 runs about **10–12 weeks**. Each milestone ends at a gate - we don't proceed until the gate's outcome is achieved.
| # | Milestone | Weeks | Gate: what must be true to move on |
| --- | --- | --- | --- |
| **M0** | Model reconnaissance complete | 1 | We have a verified, first-hand description of our first model, and the team trusts it more than any public writeup |
| **M1** | Foundation contracts in place | 2 | The structure other people build against exists and is testable, before any model is added to it |
| **M2** | Reference inference path | 3–4 | We can run the model ourselves and prove the output is correct against an independent reference |
| **M3** | Working engine | 3–4 | Generation works, is cached correctly, and performance is measured and recorded |
| **M4** | Second model added | 2 | A second, different model runs through the same path - **with no changes to the engine** |

M0 and M1 run in parallel. M4 is serial and is the real exit test for the stage.
**What we are NOT doing before M4:** training, evaluation scoreboards, quantization, device deployment, agents. Pulling any of these forward is the main scheduling risk we've identified.

* * *

## 7. How the stages connect

| Stage | Deliverable | Depends on |
| --- | --- | --- |
| **1 - Foundation** | Model and inference infrastructure | - |
| 2 - Post-training | Training pipeline + evaluation | Stage 1 |
| 3 - Deployment | Quantized runtime on target hardware | Stage 2 |
| 4 - Agents | Tool use, planning, memory | Stage 1's serving contract |

Stages 2 and 3 each split into a framework (infrastructure) and a set of recipes or packaging (application). Stage 4 is largely independent of 2 and 3 and can begin as soon as Stage 1's serving contract is defined.
Note that Stage 4 is where the project's subject matter - the artificial scaffolding of agents - becomes the work itself rather than the thing being enabled. Stages 1–3 exist to make Stage 4 possible.

* * *

## 8. How we work

Small team, part-time, with AI agents assisting throughout. The conventions exist to protect limited joint hours:

*   **One maintainer per repository.** Shared ownership stalls on reviews.
    
*   **Small, short-lived branches and PRs.** No long-running per-person branches.
    
*   **Exact version pins between repos.** Upgrades are explicit PRs, never automatic.
    
*   **Decisions get written down.** Written context is how a part-time team survives gaps between sessions - and how agent collaborators get context they weren't present for.
    
*   **Scope discipline.** Deferred means deferred. The list in §6 is a commitment, not a suggestion.
    

* * *

## 9. To be documented later

The following are **open and under investigation**. They will be written up as they're resolved, not before:

*   The specific model chosen as the first target, and its verified configuration
    
*   The embedded hardware target and its constraints
    
*   Performance and quality targets
    
*   Interface and contract details between repositories
    
*   The choice of second model that proves the foundation generalizes
    
*   Evaluation criteria and benchmark selection

## 10. Use of AI

**Is this AI slop?** - **Yes, it is** would be the short answer. In fact, most of the code including this README document will be generated by AI. Therefore, you should not be running this in any kind of production environment. ~~Although at least one human should review the code before pushing it to GitHub~~
