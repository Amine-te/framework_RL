# 🗺️ Framework Exploration Guide

**Complete guide to exploring and running all experiments in the Reinforcement Learning Framework**

This guide helps you navigate the codebase, understand what each script does, and run every possible experiment to see all outcomes.

---

## 📁 Project Structure Overview

```
framework_RL/
├── src/                    # Core framework
│   └── gridworld.py       # GridWorld environment
│
├── scripts/
│   ├── agents/            # Individual agent implementations
│   ├── train/             # Training scripts
│   ├── experiments/       # Experimental/comparison scripts
│   ├── utils/             # Common utilities
│   └── test/              # Testing scripts
│
└── results/               # All outputs
    ├── agents/            # Trained agents (.pkl)
    ├── plots/             # Visualizations (.png)
    └── gifs/              # Animations (.gif)
```

---

## 🎯 Quick Start: What to Run First

### 1. **Classic Agents Comparison** (Start Here!)
```bash
python scripts/train/train_classic_agents.py --agent all --grid-size 5 --episodes 1000
```
**What it does:** Trains and compares 4 classic RL algorithms (Policy Iteration, Value Iteration, Monte Carlo, Q-Learning)  
**Output:** 
- Console: Performance metrics and comparison table
- `results/plots/agents_basic_grid5_ep1000_PI_VI_MC_QL_comparison.png` - Learning curves

**Expected Outcome:** You'll see how different algorithms perform:
- **Policy Iteration** & **Value Iteration**: Fast convergence (model-based)
- **Monte Carlo** & **Q-Learning**: Learn from experience (model-free)

---

## 🤖 Agent Implementations (`scripts/agents/`)

### Available Agents

| Agent | File | Type | Description |
|-------|------|------|-------------|
| **Policy Iteration** | `agents/policy_iteration.py` | Model-based | Alternates policy evaluation and improvement |
| **Value Iteration** | `agents/value_iteration.py` | Model-based | Directly computes optimal value function |
| **Monte Carlo** | `agents/monte_carlo.py` | Model-free | Learns from complete episodes |
| **Q-Learning** | `agents/q_learning.py` | Model-free | Off-policy TD learning |

**How to use agents directly:**
```python
from scripts.agents import PolicyIterationAgent, QLearningAgent
from src.gridworld import GridWorldEnv

env = GridWorldEnv(grid_size=5)
agent = QLearningAgent(env, alpha=0.1, gamma=0.9)
results = agent.train(num_episodes=1000)
```

---

## 🏋️ Training Scripts (`scripts/train/`)

### 1. **Train Classic Agents** (`train/train_classic_agents.py`)

**Purpose:** Train and compare classic RL algorithms

**Usage Examples:**
```bash
# Train all agents and compare
python scripts/train/train_classic_agents.py --agent all

# Train only Q-Learning with custom episodes
python scripts/train/train_classic_agents.py --agent QL --episodes 2000

# Train specific agents on larger grid
python scripts/train/train_classic_agents.py --agent PI VI --grid-size 7

# Custom evaluation episodes
python scripts/train/train_classic_agents.py --agent all --eval-episodes 200
```

**Parameters:**
- `--agent`: Agent(s) to run (`PI`, `VI`, `MC`, `QL`, or `all`)
- `--episodes`: Training episodes for MC/QL (default: 1000)
- `--eval-episodes`: Evaluation episodes (default: 100)
- `--grid-size`: Grid size (default: 5)
- `--seed`: Random seed (default: 42)

**Outputs:**
- Console: Training progress, evaluation metrics, comparison table
- Plot: `results/plots/agents_basic_grid{size}_ep{episodes}_{agents}_comparison.png`

**What to observe:**
- Model-based agents (PI, VI) converge quickly
- Model-free agents (MC, QL) show learning curves
- Success rates and average returns for each agent

---

### 2. **Q-Learning Parameter Studies** (`train/train_q_learning.py`)

**Purpose:** Systematic hyperparameter exploration for Q-Learning

**Usage:**
```bash
# Default: 2000 episodes per configuration
python scripts/train/train_q_learning.py

# Custom episodes
python scripts/train/train_q_learning.py --episodes 3000
```

**What it does:**
1. **Alpha (Learning Rate) Study**: Tests α = [0.05, 0.1, 0.3, 0.5] on 7x7 grid
2. **Gamma (Discount Factor) Study**: Tests γ = [0.7, 0.9, 0.95, 0.99] on 7x7 grid
3. **Grid Size Study**: Tests grid sizes [5, 7, 10] with fixed α=0.1, γ=0.9

**Outputs:**
- **Agents saved:** `results/agents/q_learning_ep{episodes}_alpha{alpha}_gamma{gamma}_grid{size}.pkl`
- **Plots:**
  - `results/plots/q_learning_ep{episodes}_alpha_comparison_grid7.png`
  - `results/plots/q_learning_ep{episodes}_gamma_comparison_grid7.png`
  - `results/plots/q_learning_ep{episodes}_grid_size_comparison.png`

**What to observe:**
- **Alpha**: Higher values learn faster but may be unstable
- **Gamma**: Higher values value future rewards more (better for long-term planning)
- **Grid Size**: Larger grids require more episodes to learn

**Expected Outcomes:**
- Alpha ~0.3 often performs best
- Gamma ~0.95-0.99 works well for goal-reaching tasks
- Larger grids show slower convergence

---

## 🧪 Experimental Scripts (`scripts/experiments/`)

### 1. **Q-Learning with Moving Goals** (`experiments/q_learning_moving_goal.py`)

**Purpose:** Compare two state representations when goal position changes every episode

**Usage:**
```bash
python scripts/experiments/q_learning_moving_goal.py --episodes 5000 --grid_size 7
```

**What it compares:**
- **Goal-Aware Agent**: State includes (agent_pos, goal_pos) - knows goal location
- **Position-Only Agent**: State only includes agent position - must learn to navigate

**Outputs:**
- Plot: `results/plots/q_learning_moving_goal_ep{episodes}_grid{size}_comparison.png`
- Shows: Rewards, steps, success rates, Q-value changes

**Expected Outcome:**
- Goal-aware agent learns faster and achieves higher success rates
- Demonstrates importance of state representation

---

### 2. **Deep Q-Learning: Fixed Goal** (`experiments/deep_q_learning_fixed_goal.py`)

**Purpose:** Compare Naive DQN vs Target DQN on fixed goal task

**Usage:**
```bash
python scripts/experiments/deep_q_learning_fixed_goal.py
```

**What it compares:**
- **Naive DQN**: Uses same network for prediction and target (unstable)
- **Target DQN**: Uses separate target network updated periodically (stable)

**Outputs:**
- Plot: `results/plots/dqn_fixed_goal_ep500_grid7_comparison.png`
- Shows: Rewards, steps, success rates, training loss

**Expected Outcome:**
- Target DQN shows more stable learning
- Naive DQN may have high variance in performance

**Requirements:** PyTorch installed

---

### 3. **Deep Q-Learning: Moving Goal** (`experiments/deep_q_learning_moving_goal.py`)

**Purpose:** Compare state representations for DQN with moving goals

**Usage:**
```bash
python scripts/experiments/deep_q_learning_moving_goal.py
```

**What it compares:**
- **Relative State**: (dx, dy) - relative position to goal
- **Absolute State**: (x_agent, y_agent, x_goal, y_goal) - absolute positions

**Outputs:**
- Plot: `results/plots/dqn_moving_goal_ep1000_grid7_state_comparison.png`

**Expected Outcome:**
- Relative state representation generalizes better to new goal positions
- Demonstrates importance of feature engineering for neural networks

**Requirements:** PyTorch installed

---

### 4. **Stable-Baselines3 DQN** (`experiments/deep_q_learning_stable_baseline3.py`)

**Purpose:** Use Stable-Baselines3 library for DQN training

**Usage:**
```bash
python scripts/experiments/deep_q_learning_stable_baseline3.py
```

**What it does:**
1. Trains DQN on fixed goal task
2. Compares relative vs absolute state for moving goals

**Outputs:**
- `results/plots/sb3_dqn_fixed_goal_grid7.png`
- `results/plots/sb3_dqn_moving_goal_grid7_state_comparison.png`

**Requirements:** 
- PyTorch
- stable-baselines3
- gymnasium

**Expected Outcome:**
- Professional RL library with optimized hyperparameters
- Typically better performance than custom implementations

---

### 5. **Function Approximation: No Goal Info** (`experiments/func_approx_comparison_no_goal_info.py`)

**Purpose:** Compare tabular Q-Learning vs function approximation when goal info is NOT available

**Usage:**
```bash
python scripts/experiments/func_approx_comparison_no_goal_info.py --episodes 5000 --grid_size 7
```

**What it compares:**
- **Tabular Q-Learning**: Traditional Q-table
- **Linear Function Approximation**: One-hot encoding or tile coding
- **Deep Q-Network (DQN)**: Neural network approximation

**Outputs:**
- Plot: `results/plots/func_approx_no_goal_ep{episodes}_grid{size}_comparison.png`

**Expected Outcome:**
- Tabular methods work well for small state spaces
- Function approximation needed for larger/complex problems
- DQN can learn complex patterns but requires more data

---

### 6. **Function Approximation: With Goal Info** (`experiments/func_approx_camparison_with_goal_info.py`)

**Purpose:** Compare function approximation methods when goal position IS available

**Usage:**
```bash
python scripts/experiments/func_approx_camparison_with_goal_info.py --episodes 5000 --grid_size 7
```

**What it compares:**
- **Linear FA with Goal**: Linear function approximation with goal features
- **DQN with Goal**: Deep Q-network with goal information

**Outputs:**
- Plot: `results/plots/func_approx_with_goal_ep{episodes}_grid{size}_comparison.png`

**Expected Outcome:**
- Goal information significantly improves learning
- Demonstrates value of informative state representations

---

## 🧪 Testing Scripts (`scripts/test/`)

### Test Q-Learning Agents (`test/test_ql_agents.py`)

**Purpose:** Load trained agents and create GIF visualizations of their behavior

**Usage:**
```bash
# Test all available agents
python scripts/test/test_ql_agents.py --all

# Test specific agents
python scripts/test/test_ql_agents.py --agents g5_a0.3_gm0.99 g7_a0.3_gm0.99

# Custom output directory
python scripts/test/test_ql_agents.py --all --output-dir custom_gifs
```

**What it does:**
- Loads trained Q-Learning agents from `results/agents/`
- Runs episodes and creates animated GIFs
- Shows agent behavior visually

**Outputs:**
- GIFs: `results/gifs/test_ql_grid{size}_alpha{alpha}_gamma{gamma}.gif`

**Requirements:** 
- Trained agents must exist in `results/agents/`
- Pillow library for GIF creation

**Expected Outcome:**
- Visual animations showing agent navigating to goal
- Success/truncated status for each episode

---

## 🔧 Utility Functions (`scripts/utils/`)

### Helpers (`utils/helpers.py`)
- `moving_average(data, window)`: Smooth data for visualization
- `get_random_goal(grid_size, avoid_positions)`: Generate random goal positions

### Visualization (`utils/visualization.py`)
- `compare_agents(agents_results, ...)`: Create comparison plots
- `print_comparison_table(agents_results)`: Print formatted comparison table

---

## 📊 Complete Experiment Workflow

### Workflow 1: Classic Algorithms Comparison
```bash
# Step 1: Train all classic agents
python scripts/train/train_classic_agents.py --agent all --episodes 2000

# Step 2: View results
# - Check console for metrics
# - View plot: results/plots/agents_basic_grid5_ep2000_*_comparison.png
```

### Workflow 2: Q-Learning Hyperparameter Tuning
```bash
# Step 1: Run parameter studies
python scripts/train/train_q_learning.py --episodes 2000

# Step 2: Analyze results
# - View alpha comparison: results/plots/q_learning_ep2000_alpha_comparison_grid7.png
# - View gamma comparison: results/plots/q_learning_ep2000_gamma_comparison_grid7.png
# - View grid size comparison: results/plots/q_learning_ep2000_grid_size_comparison.png

# Step 3: Test best agents
python scripts/test/test_ql_agents.py --all
```

### Workflow 3: Deep Learning Experiments
```bash
# Step 1: Compare DQN variants (fixed goal)
python scripts/experiments/deep_q_learning_fixed_goal.py

# Step 2: Compare state representations (moving goal)
python scripts/experiments/deep_q_learning_moving_goal.py

# Step 3: Use Stable-Baselines3
python scripts/experiments/deep_q_learning_stable_baseline3.py
```

### Workflow 4: Function Approximation Studies
```bash
# Step 1: Compare without goal info
python scripts/experiments/func_approx_comparison_no_goal_info.py --episodes 5000

# Step 2: Compare with goal info
python scripts/experiments/func_approx_camparison_with_goal_info.py --episodes 5000
```

---

## 🎯 Expected Outcomes Summary

### Classic Agents
- **Policy Iteration**: Fast convergence, optimal policy
- **Value Iteration**: Similar to PI, often faster
- **Monte Carlo**: Slower convergence, learns from episodes
- **Q-Learning**: Fast learning, good for online scenarios

### Q-Learning Parameters
- **Alpha (0.1-0.3)**: Good learning rate range
- **Gamma (0.95-0.99)**: High values for goal-reaching tasks
- **Grid Size**: Larger grids need more episodes

### Deep Q-Learning
- **Target DQN > Naive DQN**: More stable learning
- **Relative State > Absolute State**: Better generalization
- **Stable-Baselines3**: Professional implementation with good defaults

### Function Approximation
- **Tabular**: Best for small state spaces
- **Linear FA**: Good balance of simplicity and performance
- **DQN**: Powerful but requires more data and tuning

---

## 📁 Output Locations

All results are automatically saved to `results/`:

| Type | Location | Format | Naming Convention |
|------|----------|--------|-------------------|
| **Trained Agents** | `results/agents/` | `.pkl` | `{script}_{params}.pkl` |
| **Plots** | `results/plots/` | `.png` | `{script}_{params}_{type}.png` |
| **GIFs** | `results/gifs/` | `.gif` | `test_ql_grid{size}_alpha{alpha}_gamma{gamma}.gif` |

**Example filenames:**
- `q_learning_ep2000_alpha0.1_gamma0.9_grid7.pkl`
- `agents_basic_grid5_ep1000_PI_VI_MC_QL_comparison.png`
- `test_ql_grid7_alpha0.3_gamma0.99.gif`

---

## 🔍 Exploring the Codebase

### Understanding Agent Implementations
1. **Start with:** `scripts/agents/q_learning.py` (simplest model-free agent)
2. **Then check:** `scripts/agents/policy_iteration.py` (model-based approach)
3. **Compare:** How model-based vs model-free differ

### Understanding Training Scripts
1. **Read:** `scripts/train/train_classic_agents.py` (clean training loop)
2. **Study:** `scripts/train/train_q_learning.py` (parameter studies)

### Understanding Experiments
1. **Start simple:** `experiments/q_learning_moving_goal.py`
2. **Then complex:** `experiments/deep_q_learning_moving_goal.py`

---

## 🚀 Quick Reference: All Commands

```bash
# Classic Agents
python scripts/train/train_classic_agents.py --agent all
python scripts/train/train_classic_agents.py --agent QL --episodes 2000

# Q-Learning Studies
python scripts/train/train_q_learning.py --episodes 2000

# Moving Goal Experiments
python scripts/experiments/q_learning_moving_goal.py --episodes 5000

# Deep Q-Learning
python scripts/experiments/deep_q_learning_fixed_goal.py
python scripts/experiments/deep_q_learning_moving_goal.py
python scripts/experiments/deep_q_learning_stable_baseline3.py

# Function Approximation
python scripts/experiments/func_approx_comparison_no_goal_info.py --episodes 5000
python scripts/experiments/func_approx_camparison_with_goal_info.py --episodes 5000

# Testing
python scripts/test/test_ql_agents.py --all
```

---

## 💡 Tips for Exploration

1. **Start Small**: Begin with `--grid-size 5` and `--episodes 500` to see quick results
2. **Compare Parameters**: Run same script with different parameters to see effects
3. **Check Plots**: All plots are saved with meaningful names - easy to compare
4. **Read Console Output**: Scripts print detailed progress and metrics
5. **View GIFs**: Visual understanding of agent behavior

---

## 🎓 Learning Path Recommendations

### Beginner Path
1. Run `train_classic_agents.py --agent all` → Understand basic algorithms
2. Run `train_q_learning.py` → Learn about hyperparameters
3. View plots in `results/plots/` → See learning curves

### Intermediate Path
1. Run moving goal experiments → Understand state representation
2. Compare function approximation methods → See trade-offs
3. Test agents with GIFs → Visual understanding

### Advanced Path
1. Modify agent implementations in `scripts/agents/`
2. Create custom experiments in `scripts/experiments/`
3. Experiment with different grid configurations

---

## 📝 Notes

- All scripts use meaningful filenames for outputs
- Results are organized by type (agents, plots, gifs)
- Random seeds are set for reproducibility
- Most scripts have `--help` for parameter information

**Happy Exploring! 🚀**

