# Reinforcement Learning Agent 10-Year Evolution Simulation

This project simulates the 10-year evolution of a reinforcement learning agent, where each "year" represents a significant architectural and conceptual leap. The development timeline is modeled as a Markov chain, with each new agent (`T_n+1`) being generated from the previous one (`T_n`) by incorporating new research trends.

## Current State: Year 2 (T2) - GOAL-INV-LATENT-HIRO

The current implementation is the **Year 2 Agent (T2)**, a hierarchical reinforcement learning agent designed to operate in a latent goal space.

### T2 Architecture

The `T2` agent has a two-level hierarchical structure:

1.  **Manager (High-Level):**
    *   **Goal:** To solve the overall task (e.g., open the door in the KeyDoor environment).
    *   **Policy:** An actor-critic agent that operates in a learned latent space. It selects latent goals (`z_g`) for the `LowLevelController` to achieve.
    *   **Training:** The `Manager` is trained using an off-policy correction mechanism called Inverse-HIRO. It uses a `LatentGoalModule` to predict the "achieved" latent goal from a trajectory and uses this relabeled goal to update its policy.

2.  **LowLevelController (Low-Level):**
    *   **Goal:** To reach the latent goal (`z_g`) commanded by the `Manager`.
    *   **Policy:** A conservative Q-learning (CQL) agent that takes actions in the environment.
    *   **Training:** The `LowLevelController` is trained with Hindsight Experience Replay (HER) and Retrieval-Enhanced Replay (RER) to improve sample efficiency. The reward is the negative distance to the latent goal.

3.  **LatentGoalModule:**
    *   **State Encoder:** A VAE-style encoder that compresses the environment state (`s`) into a low-dimensional latent vector (`z`).
    *   **Inverse Model:** Predicts the latent goal (`z_g`) that connects a start state (`s_t`) and an end state (`s_t+c`).

## 10-Year Development Roadmap

This project follows a deterministic 10-year token chain, where each token represents a new generation of the agent.

*   **T0: The Seed (`HIER-HER-ICM-PER-TRIPLEQ-ADAPT-KEYDOOR`)**
    *   **State:** A hierarchical agent (HIRO) using Hindsight Experience Replay (HER), Intrinsic Curiosity (ICM), Prioritized Replay (PER), 3 Q-networks (Triple-Q), and Adaptive Epsilon.
    *   **Context:** Solves the KeyDoor maze but is sample-inefficient and prone to instability.

*   **T1: Stability & Memory (`HYBRID-QC-HER++-RER`)**
    *   **Concept:** Hybrid Q-Conservative + Retrieval-Enhanced Replay.
    *   **Leap:** Replaces simple Triple-Q with a conservative ensemble that penalizes OOD actions and augments PER with "semantic retrieval" (replaying similar past successes).

*   **T2: The Semantic Goal (`GOAL-INV-LATENT-HIRO`)**
    *   **Concept:** Latent Goal Generation + Inverse Kinematics.
    *   **Leap:** The Manager outputs a *latent* vector (`z_g`) instead of raw coordinates. A learned Inverse Model ensures goals are physically feasible.

*   **T3: Intrinsic Maturity (`ICM-EMP-INFOGAIN`)**
    *   **Concept:** Empowerment & Information Gain.
    *   **Leap:** Adds "Empowerment" (maximizing the mutual information between action and next state) to encourage controlling the environment, not just finding chaos.

*   **T4: Off-Policy Correction (`CORRECT-HIER-IMPORTANCE`)**
    *   **Concept:** Strict Off-Policy Correction.
    *   **Leap:** Implements strict importance sampling or "relabeling" to check if the current low-level policy would have reached a subgoal from the replay buffer.

*   **T5: The Meta-Controller (`META-BANDIT-AUTO-TUNE`)**
    *   **Concept:** Online Hyperparameter Bandit.
    *   **Leap:** A high-level Bandit controller adjusts hyperparameters like epsilon, ICM scaling, and learning rates based on recent performance.

*   **T6: Structured Representation (`GNN-RELATIONAL-ATTN`)**
    *   **Concept:** Graph Neural Networks & Attention.
    *   **Leap:** Replaces MLP encoders with a Graph Attention Network (GAT) where objects in the environment are nodes, allowing the agent to learn relational concepts.

*   **T7: Continuous-Stochastic (`SAC-HIER-DISTRIB`)**
    *   **Concept:** Soft Actor-Critic Hierarchy.
    *   **Leap:** Moves from Q-learning to Maximum Entropy RL (SAC), with both Manager and Controller becoming stochastic actors for improved exploration.

*   **T8: World Modeling (`DREAMER-HIER-MBRL`)**
    *   **Concept:** Model-Based Planning in Latent Space.
    *   **Leap:** The Manager uses a learned world model (Dreamer-style) to "imagine" the outcome of a subgoal before commanding it.

*   **T9: Symbolic Abstraction (`NEURO-SYM-PROGRAM`)**
    *   **Concept:** Neuro-Symbolic Skill Synthesis.
    *   **Leap:** The agent identifies recurring subgoal sequences and "compiles" them into symbolic macros (e.g., `fetch_key()`).

*   **T10: Lifelong Autonomy (`AGI-LITE-CONTINUAL`)**
    *   **Concept:** Continual Learning & Safety Constraints.
    *   **Leap:** The system uses Generative Replay to prevent catastrophic forgetting and Constrained MDPs to ensure safety while exploring new, procedurally generated worlds.

## Running the Simulation

### Setup

The project requires Python 3 and the following libraries:

*   `torch`
*   `numpy`
*   `scipy`

Install the dependencies with pip:

```bash
pip install torch scipy
```

### Execution

To run the simulation, execute the `main.py` script:

```bash
python main.py
```

The script will run the T2 agent in the KeyDoor environment and print the success rate every 10 episodes.

## Codebase Structure

*   **`main.py`:** The main entry point for the simulation. It contains the training loop and initializes the agent, environment, and other components.
*   **`agent.py`:** Contains all the core agent-related classes:
    *   `Manager`: The high-level, goal-setting agent.
    *   `LowLevelController`: The low-level, action-executing agent.
    *   `LatentGoalModule`: The state encoder and inverse model.
    *   `HERRetrievalBuffer`: The replay buffer, which implements HER and RER.
    *   `ActorNetwork`, `CriticNetwork`, `QNetwork`: The neural network architectures.
*   **`env.py`:** Defines the `KeyDoorEnv` environment for the simulation.
