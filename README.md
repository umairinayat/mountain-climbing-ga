# mountain-climbing-ga
Evolving virtual creatures to climb mountains using advanced genetic algorithms in PyBullet.



# Evolving Climbers: Advanced Genetic Algorithm for Mountain-Climbing Creatures

This repository contains a high-performance implementation of a modular genetic algorithm designed to evolve virtual creatures capable of climbing complex 3D terrain. The work was completed as Part B of the CM3020 Artificial Intelligence coursework and includes over 4,999 lines of professional-grade Python code, systematic experimentation across 35+ configurations, and breakthrough results in evolutionary robotics.

---

## 🧠 Project Summary

- **Task**: Evolve virtual creatures to climb a simulated mountain using a physics-based environment (PyBullet).
- **Key Metrics**:
  - Max Height: **20.22 units** (400% over baseline)
  - Peak Fitness: **983.08**
  - Total Runtime: **52+ hours**
  - Total Experiments: **35+**
- **Innovation Areas**:
  - Six distinct genetic decoding strategies
  - Multi-objective fitness functions (stability, energy, climbing)
  - Adaptive mutation and convergence detection
  - Support for massive populations (up to 400 individuals)

---

## 🛠️ System Components

| Module                    | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `mountain_climbing_ga.py` | Main evolution controller with convergence checks, mutation control, logging |
| `mountain_environment.py` | PyBullet simulation environment with custom terrain                        |
| `mountain_population.py`  | Manages population creation, selection, crossover, and diversity           |
| `mountain_fitness.py`     | Multi-objective fitness evaluator with 9 weighted components               |
| `results_manager.py`      | Handles results storage, logging, and auto-visualization                   |

---

## 📈 Genetic Decoding Strategies

- **Motor Control Only**: Oscillatory patterns, fixed morphology
- **Body Shape Only**: Limb design evolution, fixed movement
- **Partial Evolution**: Targeted limb/torso mutation
- **Modular Evolution**: Evolving functional modules independently
- **Multi-Objective Optimization**: Balanced tradeoffs between stability, performance, and efficiency
- **Hybrid Evolution**: Dynamic strategy switching during run-time

---

## 📊 Key Results

| Population Size | Max Fitness | Max Height | Convergence Time |
|-----------------|-------------|------------|------------------|
| 100             | 983.08      | 20.22      | 81 generations   |
| 400             | 752.45      | 17.31      | 22.3 minutes     |
| 60              | 596.45      | 15.40      | 910 minutes      |

---

## 🔬 Experimentation Highlights

- 35+ experimental setups with varying:
  - Population sizes (20–400)
  - Mutation rates (0.05–0.50)
  - Evolution durations (60–300 generations)
- Extensive logging, automatic chart generation, and statistical summaries

---



## 🚀 Installation

```bash
git clone https://github.com/umairinayat/mountain-climbing-ga.git
cd evolving-climbers
pip install -r requirements.txt
python cw-envt.py  # or run integrated evolution script
