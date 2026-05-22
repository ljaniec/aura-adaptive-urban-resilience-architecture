# aura-adaptive-urban-resilience-architecture

The Adaptive Urban Resilience Architecture (AURA) project repository with the PoC for the "Distributed adaptive facade control for passive urban ventilation and thermal regulation" part

## PoC Demo Commands

Run from repository root:

```bash
cd aura
source .venv/bin/activate
```

### Step 1 PoC

Rectangular building + wind from the left + tunnel-effect focused visualization.

```bash
kedro run --pipeline step1_poc --env step1
```

Output figure:

```text
data/08_reporting/viz_step1_effects.png
```

### Step 2 PoC

Adds indoor airflow and thermal coupling.

```bash
kedro run --pipeline step2_poc --env step2
```

Output figure:

```text
data/08_reporting/viz_step2_thermal.png
```

### Step 3 PoC

Static facade strategy with flow + thermal outcome.

```bash
kedro run --pipeline step3_poc --env step3
```

Output figure:

```text
data/08_reporting/viz_step3_facade.png
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

Use reduced timesteps for fast demos.

```bash
kedro run --pipeline step1_poc --env step1 --params="simulation.steps=70,external_cfd.lbm.steps=70"
kedro run --pipeline step2_poc --env step2 --params="simulation.steps=70,simulation.internal_steps=60,simulation.thermal_steps=60,external_cfd.lbm.steps=70"
kedro run --pipeline step3_poc --env step3 --params="simulation.steps=70,simulation.internal_steps=60,simulation.thermal_steps=60,external_cfd.lbm.steps=70"
```
