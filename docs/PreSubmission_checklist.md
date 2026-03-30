# ✅ OpenEnv Submission Checklist

## 🚀 Core Task

- [ ] Environment simulates a real-world task (not a game/toy)
- [ ] Implements OpenEnv APIs:
  - [ ] step()
  - [ ] reset()
  - [ ] state()

---

## 📌 OpenEnv Spec Compliance

- [ ] Typed models implemented:
  - [ ] Observation
  - [ ] Action
  - [ ] Reward
- [ ] step(action) returns observation, reward, done, info
- [ ] reset() returns initial observation
- [ ] state() returns current state
- [ ] openenv.yaml is present and valid
- [ ] openenv validate passes

---

## 🧠 Tasks & Evaluation

- [ ] At least 3 tasks implemented
- [ ] Tasks cover Easy → Medium → Hard
- [ ] Each task has:
  - [ ] Clear objective
  - [ ] Deterministic grader
- [ ] Graders return scores between 0.0 – 1.0

---

## 🎯 Reward Design

- [ ] Reward function provides continuous feedback
- [ ] Rewards partial progress
- [ ] Penalizes bad behavior (loops, destructive actions)

---

## 🤖 Baseline Inference

- [ ] inference.py exists in root directory
- [ ] Uses OpenAI client
- [ ] Reads required environment variables
- [ ] Runs without errors
- [ ] Produces reproducible scores

---

## 🐳 Deployment & Infra

### Hugging Face Space

- [ ] Deployed as Hugging Face Space
- [ ] Returns HTTP 200
- [ ] reset() endpoint works

### Docker

- [ ] Dockerfile exists
- [ ] docker build succeeds
- [ ] docker run succeeds

---

## 📦 Runtime Constraints

- [ ] Runs under 20 minutes
- [ ] Works with 2 vCPU, 8GB RAM

---

## 📚 Documentation

- [ ] README includes:
  - [ ] Environment description
  - [ ] Action space
  - [ ] Observation space
  - [ ] Task descriptions
  - [ ] Setup instructions
  - [ ] Usage instructions
  - [ ] Baseline results

---

## 🧪 Pre-Submission Validation

- [ ] HF Space deploys successfully
- [ ] OpenEnv spec validated
- [ ] Docker build passes
- [ ] Inference script runs successfully
- [ ] All tasks produce valid scores

---

## 🔧 Environment Variables

- [ ] API_BASE_URL set
- [ ] MODEL_NAME set
- [ ] HF_TOKEN set

---

## 🧪 Validator

- [ ] Validation script executed successfully

---

## 🏁 Final Sanity Check

- [ ] End-to-end execution works
- [ ] Clean project structure
- [ ] No missing dependencies
- [ ] Fully reproducible
