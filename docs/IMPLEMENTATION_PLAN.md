## Plan: Customer Support Ticket Triage Environment

Build an OpenEnv-compliant, production-oriented customer support ticket triage environment with deterministic scoring, dense reward shaping, a reproducible baseline inference pipeline, and deployment readiness for Hugging Face Spaces.

### TL;DR
Implement a SaaS support-operations simulator where an agent triages tickets (category, priority, route, next action, response draft). Use static scenario fixtures and deterministic grader rules to produce auditable scores in [0.0, 1.0]. Prioritize strict OpenEnv API compliance (`step`, `reset`, `state`), reproducibility, and fast validation.

### Scope
- Included:
  - Real-world support triage simulation
  - 3 tasks with easy/medium/hard progression
  - Deterministic graders and shaped rewards
  - Root `inference.py` using OpenAI client and env vars
  - Containerized deployment artifacts and docs
- Excluded:
  - Non-deterministic LLM-based grading
  - External live ticket APIs
  - Human-in-the-loop labeling during evaluation

### Assumptions
- Python 3.10+
- OpenEnv core APIs available locally
- Docker available locally
- HF Space uses container deployment path

### Phases and Steps
1. Phase A: Domain and policy specification
2. Phase B: Data model and action protocol design
3. Phase C: Environment mechanics and state transitions
4. Phase D: Task fixtures and deterministic graders
5. Phase E: Reward shaping and anti-loop controls
6. Phase F: Baseline inference pipeline
7. Phase G: OpenEnv metadata, validation, and deployment
8. Phase H: Testing, documentation, and submission hardening

### Phase A: Domain and Policy Specification
1. Define support domain boundaries and operating assumptions.
   - Product context: SaaS productivity platform.
   - Teams/queues: billing, technical, account-management, trust-and-safety.
   - Ticket channels: email/web form (normalized in fixtures).
2. Define deterministic triage policy matrix.
   - Mapping of issue category to routing queue.
   - Mapping of severity and SLA indicators to priority.
   - Rules for escalation vs request-for-information.
3. Define safety and compliance rules for response drafts.
   - Required response components by issue type.
   - Prohibited phrases and policy violations.
   - Maximum response length and formatting constraints.

### Phase B: Data Model and Action Protocol Design
1. Define typed enums and value sets.
   - Categories: billing_dispute, access_issue, outage_report, abuse_report, feature_request, bug_report, other.
   - Priorities: low, medium, high, urgent.
   - Queues: billing_ops, tech_support, csm, trust_safety.
   - Next actions: request_info, provide_steps, escalate_engineering, escalate_security, refund_review, close.
2. Define core models.
   - `TicketRecord` immutable input + mutable decisions.
   - `TicketDecision` with agent-assigned triage outputs.
   - `TicketTriageAction`, `TicketTriageObservation`, `TicketTriageState`.
3. Define action contract and validation rules.
   - Action types and required payload fields.
   - Rejection behavior for malformed or out-of-order actions.
   - No hidden-answer fields exposed in observations.

### Phase C: Environment Mechanics and State Transitions
1. Implement episode lifecycle semantics.
   - `reset` loads one scenario fixture and initializes counters.
   - `step` validates action, mutates state, computes reward increment.
   - `state` returns current episode metadata and progress snapshot.
2. Define ticket workflow states.
   - untriaged -> in_progress -> submitted.
   - Submitted ticket can be reopened with penalty if changed.
3. Define done conditions.
   - done true when all tickets submitted, or max step budget exhausted.
4. Define info payload conventions.
   - Return machine-readable per-step diagnostics for grader-relevant errors.

### Phase D: Task Fixtures and Deterministic Graders
1. Create fixed scenario fixtures.
   - Easy: 1-2 tickets, explicit cues, minimal ambiguity.
   - Medium: 3-4 tickets, mixed categories, SLA pressure and distractors.
   - Hard: 6-8 tickets, overlapping incidents, VIP + abuse + outage interactions.
2. Attach hidden answer key metadata in fixtures.
   - Correct category, priority, route, next action.
   - Required response rubric tags.
3. Implement per-ticket grader.
   - Weighted score components:
     - category correctness: 0.25
     - priority correctness: 0.20
     - route correctness: 0.20
     - next action correctness: 0.20
     - deterministic response compliance: 0.15
4. Implement batch grader.
   - Average per-ticket score
   - SLA-risk penalty for unresolved urgent tickets
   - Clamp final score to [0.0, 1.0]
5. Determinism guarantees.
   - No random branches in grader path.
   - Stable ordering for ticket iteration.

### Phase E: Reward Shaping and Anti-Loop Controls
1. Dense progress rewards.
   - +0.05 for each correct triage field set for first time.
   - +0.02 for correcting previously incorrect field.
   - +0.03 for response draft passing compliance subset.
2. Penalties.
   - -0.03 invalid action schema.
   - -0.05 contradictory reassignment loop on same ticket field.
   - -0.01 per step after soft budget threshold.
   - -0.08 high-risk misroute for abuse/security/billing-critical cases.
3. Terminal signal.
   - +0.20 if final batch score meets threshold target.
4. Reward safety checks.
   - Clip per-step reward bounds to avoid exploding trajectories.

### Phase F: Baseline Inference Pipeline
1. Root `inference.py` requirements.
   - Must exist at repository root.
   - Use OpenAI client.
   - Read `OPENAI_API_KEY` (required), and optionally `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` for compatibility.
2. Baseline execution logic.
   - Iterate deterministic easy/medium/hard scenarios.
   - Prompt model with policy + observation summary.
   - Parse constrained action outputs.
   - Stop on done or max steps.
3. Output format.
   - Per-task score table and aggregate average.
   - Print reproducibility metadata: model, seed, temperature, max steps.
4. Reproducibility controls.
   - Fixed seed in scenario ordering and any sampling path.
   - Fixed model params (temperature, max tokens).

### Phase G: OpenEnv Metadata, Validation, and Deployment
1. OpenEnv metadata.
   - Create `openenv.yaml` with name, description, tags, task metadata, and entrypoints.
2. Containerization.
   - Add Dockerfile compatible with local build and HF Space runtime.
   - Include health endpoint expectations.
3. Validation pipeline.
   - `openenv validate` must pass.
   - Docker build/run smoke test must pass.
4. HF Space deployment plan.
   - Container deployment settings.
   - `openenv` tag inclusion.
   - reset endpoint checks for HTTP 200.

### Phase H: Testing, Documentation, and Submission Hardening
1. Unit tests.
   - Model validation tests for all action payload variants.
   - Grader determinism tests (repeat-run equality).
   - Reward tests for key branches.
2. Integration tests.
   - reset-step-state lifecycle sanity test.
   - End-to-end task run for each difficulty fixture.
3. README completion.
   - Motivation and real-world utility.
   - Action/observation/state definitions.
   - Task details and grader formula.
   - Setup, run, validate, deploy instructions.
   - Baseline score results.
4. Runtime budget audit.
   - Verify under 20 minutes end-to-end under 2 vCPU / 8 GB assumptions.

### File Blueprint
- `README.md`
- `openenv.yaml`
- `inference.py`
- `Dockerfile`
- `requirements.txt`
- `ticket_triage_env/models.py`
- `ticket_triage_env/client.py`
- `ticket_triage_env/server/environment.py`
- `ticket_triage_env/server/app.py`
- `ticket_triage_env/graders/rules.py`
- `ticket_triage_env/graders/aggregate.py`
- `ticket_triage_env/data/scenarios/easy.json`
- `ticket_triage_env/data/scenarios/medium.json`
- `ticket_triage_env/data/scenarios/hard.json`
- `tests/test_models.py`
- `tests/test_graders.py`
- `tests/test_env_lifecycle.py`

### Verification Gates
1. `openenv validate` passes.
2. Docker builds and runs.
3. reset/step/state functional checks pass.
4. Graders are deterministic and bounded in [0.0, 1.0].
5. `inference.py` runs all tasks and prints reproducible results.
6. README satisfies submission checklist.
