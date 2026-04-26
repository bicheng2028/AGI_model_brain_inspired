# AGI Demo: Brain-Inspired Architecture with Hippocampal-Prefrontal Loop
https://img.shields.io/badge/python-3.8+-blue.svg

https://img.shields.io/badge/PyTorch-1.9+-red.svg

https://img.shields.io/badge/License-MIT-yellow.svg

https://img.shields.io/badge/Environment-Crafter-green.svg

This repository contains a brain-inspired AGI (Artificial General Intelligence) prototype that implements the theoretical framework proposed in:

Bi, C. (2025). A Deductive Framework for Memory, Consciousness, and Brain-Inspired AGI Based on First Principles of Neuroscience.
DOI: 10.13140/RG.2.2.26341.56801

The agent simulates multiple interacting brain regions — including cortical memory, hippocampal indexing, prefrontal control, neuromodulation, and sleep-based consolidation — to perform autonomous exploration and learning in the Crafter environment.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Theoretical Foundations](#theoretical-foundations)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Code Structure](#code-structure)
- [Visualization Dashboard](#visualization-dashboard)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)
  


### Component Roles

| Component | Brain Analogy | Function |
|-----------|--------------|----------|
| **Cortical Memory Library** | Neocortex (L2/3, L5) | Multi-level visual feature extraction (L0-L3), multimodal memory storage |
| **Hippocampal Index Library** | Hippocampus (CA3) | Causal chain storage as pointer sequences, graph-based retrieval |
| **Prefrontal Controller** | Prefrontal Cortex | Working memory, action planning via index retrieval, spatial navigation |
| **Value Discriminator** | Neuromodulatory Systems | NE/DA/ACh/Cortisol channels, motivation generation, exploration/exploitation balance |
| **Dream Engine** | NREM/REM Sleep | Offline memory consolidation, multi-level clustering, cross-level association |


## Theoretical Foundations

This implementation is grounded in several key theoretical principles from the paper:

### 1. Index-Content Separation
The hippocampus stores **indices** (pointers to cortical memory traces), not the content itself. This separation enables rapid learning of new sequences without modifying the vast stores of content memory.

### 2. Nested Free Energy Theory
- Gamma oscillations (within cortical columns) form deep free energy troughs for stable memory traces
- Cross-level oscillatory coupling creates nested free energy landscapes
- Neuromodulatory systems (dopamine, norepinephrine) regulate transitions between free energy states
- Prediction error is reinterpreted as the energetic difference required for state transitions

### 3. Sleep as Memory Consolidation
- **NREM sleep**: Sparse association between hippocampal indices and cortical memory traces
- **REM sleep**: Lateral prefrontal cortex inhibition enables free recombination of memory fragments
- Dream engine implements both same-level clustering and cross-level creative sampling

### 4. Multi-Level Sequential Representations
Stable sequential firing patterns of specific neurons within self-organized oscillatory circuits manifest as sequential representations. Higher-order brain regions guide lower-order regions through feedback connections.

---

## Key Features

### Neuro-Inspired Design
- **Multi-level visual features** extracted via CNN hooks at different depths (L0: perceptual → L1: attribute → L2: affordance → L3: conceptual)
- **Multimodal integration**: visual features + inventory (22D) + position (2D) + health + hunger + thirst
- **Dopamine-gated causal chaining**: temporary chains (DA > 0.5) → permanent chains (on second occurrence)

### Sleep/Dream Engine
- **Same-level clustering**: L0-L3 independent clustering with adaptive similarity thresholds
- **Cross-level sampling**: Random association between different feature levels (e.g., L0↔L2) for creative insight
- **Temporal biased integration**: Recent memories weighted 10× over old memories
- **Multimodal clustering**: Inventory, position, health, hunger, thirst, and discount-based clustering

### Autonomous Decision-Making
- Motivation signals generated from predicted physiological deviations
- Exploration/exploitation balance controlled by norepinephrine (NE)
- Safety filter triggered by high cortisol (Cort)
- Spatial goal-directed navigation when no causal chain is available

### Real-Time Visualization
- Environment view with day/night detection
- Index library graph structure visualization
- Neuromodulation channel monitoring
- Dream fragment visualization (original + blended images)
- Causal chain image panel
- Motivation & physiology display

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA-capable GPU (optional, CPU fallback available)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/agi-brain-inspired.git
cd agi-brain-inspired

# Install dependencies
pip install -r requirements.txt
 ```

### Required packages:
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
matplotlib>=3.3.0
crafter
opencv-python-headless
networkx

## Pretrained Decoder (Optional)
To enable dream visualization with the feature decoder, run the pretraining step once:
from agi_demo import pretrain_decoder
pretrain_decoder(save_path="decoder_weights.pt", num_frames=500, epochs=15)

## Quick Start
from agi_demo import Config, AGIAgent, run_experiment
from agi_demo import Config, AGIAgent, run_experiment
```python
# Configure the experiment
config = Config(
    total_steps=2000,
    max_memory_images=2000,
    recent_images_for_dream=50,
    render_interval=100,
)

# Run the experiment (AGI vs Random vs PPO baselines)
histories, agents, chain_counts = run_experiment(config)
```

## Code Structure

**Data Classes**
- `Config` — Global configuration dataclass
- `MultiLevelFeatures` — Multi-level feature dataclass
- `IndexChain` / `MotivationSignal` / `CandidateChain` — Supporting dataclasses

**Core Components**

- `CorticalMemoryLibrary` — Content Library (Neocortex)
  - Multi-level CNN feature extraction (L0-L3) via MobileNetV3 hooks
  - Multimodal encoding: visual, inventory, position, health, hunger, thirst, discount
  - Memory trace creation with novelty detection and dopamine-gated thresholds
  - Feature decoder for dream visualization

- `HippocampalIndexLibrary` — Index Library (Hippocampus CA3)
  - Causal chain storage as pointer sequences
  - Graph-based retrieval with similarity edges
  - Abstract node compression and temporary chain decay

- `PrefrontalController` — Controller (Prefrontal Cortex)
  - Working memory management
  - Index library query and action selection via neuromodulation strategy
  - Spatial goal-directed navigation when no causal chain is available

- `ValueDiscriminator` — Neuromodulation System
  - Multi-channel modulation: NE (exploration), DA (reward), ACh (learning), Cortisol (safety)
  - Physiological model: hunger, thirst, health, safety with decay and thresholds
  - Motivation generation and strategy selection (greedy, safety filter, temperature)

- `PretrainedDiffusionDreamer` — Dream Engine (Sleep)
  - Same-level clustering: L0-L3 independent clustering
  - Cross-level random sampling: creative association between different feature levels
  - Temporal biased integration: recent memories weighted over old memories
  - Multimodal clustering: inventory, position, health, hunger, thirst, discount

**Agent and Baselines**

- `AGIAgent` — Complete agent orchestrator
  - Day/night detection via image gray ratio analysis
  - Sleep cycle management with automatic triggering
  - Dopamine-gated causal chaining (temporary chains → permanent chains)

- `RandomAgent` — Baseline: uniformly random action selection

- `PPOAgent` — Baseline: simplified Proximal Policy Optimization

**Visualization and Experiment**

- `Dashboard` — Real-time visualization dashboard
  - Environment view with day/night status
  - Index graph (NetworkX), neuromodulation bar chart
  - Dream fragment images (original and blended)
  - Causal chain panel with before/after scene images
  - Motivation and physiology display, reward history

- `run_experiment()` — Experiment runner with multi-agent comparison
  - Automatic death detection and life tracking
  - Periodic statistics logging (every 500 steps)
  - Final comparison plots: cumulative reward, smoothed reward, achievements, index growth

## Visualization Dashboard

The dashboard provides real-time monitoring of the agent's internal state:

| Panel | Content |
|-------|---------|
| **Environment** | Game view with day/night status and step counter |
| **Index Graph** | NetworkX visualization of the hippocampal index graph |
| **Index Stats** | Total chains, nodes, edges, abstract nodes, dream chains |
| **Neuromodulation** | Bar chart of NE, DA, ACh, Cort levels with temperature |
| **Causal Chain** | Before/after scene images with motivation tag and DA level |
| **Dream Visual** | Original and blended dream fragment images |
| **Motivations** | Active motivation goals with priority scores and physiology |
| **Sleep/Dream** | Sleep status, cycle count, discovery count |
| **Reward History** | Smoothed and raw reward curves |

## Experimental Results

The agent is evaluated against two baselines in the Crafter environment:

- **Random Agent**: Uniformly random action selection
- **PPO Agent**: Simplified Proximal Policy Optimization

### Key Metrics
- Cumulative reward over time
- Smoothed reward (window=100)
- Survival steps per life
- Achievement unlock frequency
- Index library growth rate

### Sample Output


## AGI Demo V3: Pure CNN Multi-level Integration + Dream Visualization

Step 0: AGI=0.00, Random=0.00, PPO=0.00
🏆 AGI current achievements: 0/22 (0.0%)

🎖️ Lifetime best: 0/22

💤 Sleep cycle 1 (consolidating 5 times)...

🧠 Available memory images: 2

💡 Stored 1 new associations

✅ Sleep cycle 1 complete

🏆 Achievement unlocked: wake_up!

🏆 Achievement unlocked: collect_sapling!

⏳ Chain TEMPORARY: trace_0000 → 🎯general → trace_0005

✅ Chain PERMANENT: trace_0012 → 🎯general → trace_0013

Experimental Results:
AGI: Average=0.03, Max Achievement=4/22
Random: Average=0.01
PPO: Average=0.01

## Citation
If you use this code or the associated theoretical framework in your research, please cite:
@article{bi2025deductive,
  title={A Deductive Framework for Memory, Consciousness, and Brain-Inspired AGI Based on First Principles of Neuroscience},
  author={Bi, Cheng},
  year={2025},
  doi={10.13140/RG.2.2.26341.56801},
  url={https://www.researchgate.net/publication/XXXXXXX}
}

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The author thanks the developers of the Crafter environment for providing a rich benchmark for evaluating general intelligence in agents.



