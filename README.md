# SceneQueryPlanner

Official implementation of **"Memory–Environment Dual-Grounding for Embodied Agents: Active Information Seeking Beyond Passive Observation"**.

SceneQueryPlanner is an LLM-based embodied task planner that treats information acquisition as a first-class planning action. The agent interleaves three action types — `Think:` (reasoning), `Act:` (environment grounding), and `Query:` (memory grounding) — where `Query:` issues structured queries (`find_objects`, `read_node`, `get_child_node_names`, `get_edges_for_node`) against a dynamic hierarchical 3D scene graph built from partial observations.

This repository contains a unified codebase for both evaluation benchmarks used in the paper:

- **AttPlan-Bench** (ours) — long-horizon, attribute-aware household tasks in **ProcTHOR / AI2-THOR**, with two subsets: AttPlan-Base (10 task types × 50 episodes) and AttPlan-Attr (attribute-centric instructions).
- **WAH-NL** — Watch-And-Help natural-language tasks in **VirtualHome** (100 test episodes).

Implemented planners: `reactstrq` (SceneQueryPlanner, ours), `react` (ReAct), `reactwm` (ReAct + naive working memory), `sayplan` (SayPlan), `moma` (MoMa-LLM).

## Repository layout

```
core/
  conf/       Hydra configs (config_procthor.yaml, config_wah.yaml)
  env/        Simulator wrappers (procthor_env.py, wah_env.py)
  planner/    Planning loops; per-simulator variants under procthor/ and wah/
  llm_agent/  LLM interfaces (guidance-based; HF Transformers or OpenAI)
  retriever/  Structured query parsing and execution over the scene graph
  wm/         Working-memory scene graphs (dynamic 3DSG)
  utils/      Goal checking, observation rendering, visual logging
  run/        Entry points (eval, collect_human, collect_llm, embed_em_*)
dataset/      AttPlan-Bench task sets and WAH-NL splits
resource/     System prompts, object dictionaries, in-context example banks
script/       Stage-by-stage pipelines for both benchmarks
virtualhome/  Vendored VirtualHome Python package (Unity binary downloaded separately)
```

## Installation

Tested on Ubuntu 22.04, Python 3.8.

```bash
conda create -n sqp python=3.8
conda activate sqp

# Install PyTorch first (pick the wheel matching your CUDA; see https://pytorch.org)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

Credentials are read from environment variables:

```bash
export HF_TOKEN=...         # required for gated HF models (e.g., Llama 3.1)
export OPENAI_API_KEY=...   # only for OpenAI backends
```

## Evaluating on AttPlan-Bench (ProcTHOR)

AI2-THOR downloads its simulator build automatically on first run. From the repository root:

```bash
sh script/procthor_05_evaluate.sh
```

The script evaluates SceneQueryPlanner by default; baseline commands are included commented-out, along with an AttPlan-Attr (attribute-centric) section.

## Evaluating on WAH-NL (VirtualHome)

1. Download the VirtualHome Unity simulator:

    ```bash
    cd virtualhome/simulation/unity_simulator/
    wget http://virtual-home.org//release/simulator/v2.0/v2.2.2/linux_exec.zip
    unzip linux_exec.zip
    cd -
    ```

2. Run the simulator in one terminal:

    ```bash
    ./virtualhome/simulation/unity_simulator/linux_exec.x86_64
    ```

3. Evaluate in another terminal:

    ```bash
    sh script/wah_05_evaluate.sh
    ```

## Full pipeline

Evaluation uses the in-context example banks shipped under `resource/trajectory/`. To rebuild them from scratch, run the five stages in order (per benchmark):

| Stage | ProcTHOR | VirtualHome |
|---|---|---|
| 1. Collect human demos (interactive) | `script/procthor_01_collect_human.sh` | `script/wah_01_collect_human.sh` |
| 2. Embed human trajectories | `script/procthor_02_embed_human_traj.sh` | `script/wah_02_embed_human_traj.sh` |
| 3. Collect LLM trajectories | `script/procthor_03_collect_llm.sh` | `script/wah_03_collect_llm.sh` |
| 4. Embed LLM trajectories | `script/procthor_04_embed_llm_traj.sh` | `script/wah_04_embed_llm_traj.sh` |
| 5. Evaluate | `script/procthor_05_evaluate.sh` | `script/wah_05_evaluate.sh` |

All entry points are Hydra CLIs; planner, model, prompt, and example bank are selected via overrides, e.g.:

```bash
python -m core.run.eval --config-name=config_procthor \
    task_planner=reactstrq \
    llm_agent.model_name='meta-llama/Llama-3.1-8B' \
    llm_agent.em_dir='resource/trajectory/em_llm_procthor/reactstrq' \
    llm_agent.sys_prompt_path='resource/sys_prompt/procthor/reactstrq.txt'
```

Metrics reported per run: Task Success Rate (TSR), Subgoal Success Rate (SSR), and average decision steps (successful episodes).

## License

This project is released under the MIT License (see [LICENSE](LICENSE)). The vendored VirtualHome package under `virtualhome/` retains its own license (CC BY-NC-SA 4.0); see `virtualhome/LICENSE.md`.

## Citation

```bibtex
TBD (camera-ready)
```
