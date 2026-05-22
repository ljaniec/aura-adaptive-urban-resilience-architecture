# aura-adaptive-urban-resilience-architecture

The Adaptive Urban Resilience Architecture (AURA) project repository with the PoC for the "Distributed adaptive facade control for passive urban ventilation and thermal regulation" part.

## PoC Demo Commands

Run from repository root:

```bash
cd aura
source .venv/bin/activate
```

### PoC run

Rectangular building + wind from the left + tunnel-effect focused visualization.

```bash
kedro run --pipeline step1_poc --env step1
```

Output figure:

```text
data/08_reporting/viz_step1_effects.png
```

### Full Integrated Run

Runs the full adaptive experiment and all visualization artifacts.

```bash
kedro run --pipeline adaptive_facade_experiment
```

Key visual outputs:

```text
data/08_reporting/viz_dashboard.png
data/08_reporting/viz_step1_effects.png
data/08_reporting/viz_step2_thermal.png
data/08_reporting/viz_step3_facade.png
```

### Quick Runtime Commands (Laptop-Friendly)

Use reduced timesteps for a fast demo.

```bash
kedro run --pipeline step1_poc --env step1 --params="simulation.steps=70,external_cfd.lbm.steps=70"
```
