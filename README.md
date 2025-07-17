# Adaptive Synergy on MetaPointMassEnv

This repository contains the code and resulting hyperparameters from a rigorous tuning process for the **Adaptive Synergy** framework, a hybrid model combining Deep Reinforcement Learning (DRL), Meta-Learning, and Evolutionary Strategies (ES). The tuning was performed on `MetaPointMassEnv`, a fast and credible meta-reinforcement learning benchmark, making the analysis suitable for academic and research purposes.

The primary artifact is the `point_mass_tuning.ipynb` notebook, which provides a complete, reproducible workflow from environment setup and hyperparameter optimization to final validation and analysis against standard baselines.

## Repository Contents

*   `point_mass_tuning.ipynb`: A Jupyter Notebook detailing the entire experimental procedure. This includes the environment definition, the implementation of the Adaptive Synergy framework, the Optuna-based hyperparameter search, and the final validation and plotting code.
*   `best_params_pointmass.json`: A JSON file containing the optimal hyperparameters for the Adaptive Synergy framework as determined by the Optuna study.

## Framework Overview: Adaptive Synergy

The Adaptive Synergy framework integrates three distinct learning paradigms to enhance adaptation speed and robustness in dynamic environments:

1.  **Deep Reinforcement Learning (DRL) Layer**: Utilizes Proximal Policy Optimization (PPO) as the core learning algorithm to ensure stable policy updates.
2.  **Meta-Learning Layer**: Incorporates a Model-Agnostic Meta-Learning (MAML) inspired loss term to train a policy that can rapidly adapt to new tasks.
3.  **Evolutionary Strategy (ES) Layer**: Employs an evolutionary regularization term that guides the main policy towards parameter regions discovered by a population of agents, promoting both exploration and stability.

## Methodology

The experiment is designed to be fully reproducible and follows a structured workflow:

1.  **Benchmark Environment**: A custom `MetaPointMassEnv` is implemented using the `gymnasium` library. In this standard meta-RL task, a point-mass agent must navigate to a goal location that changes at fixed intervals, forcing the agent to continuously adapt.
2.  **Reproducibility**: A comprehensive seeding function ensures that all sources of randomness (`torch`, `numpy`, `random`, and the environment) are controlled, guaranteeing deterministic results for a given seed.
3.  **Automated Hyperparameter Tuning**: The [Optuna](https://optuna.org/) library performs a Bayesian optimization search to find the optimal values for the framework's most critical coefficients:
    *   `ppo_lr`: The learning rate for the PPO optimizer.
    *   `meta_loss_coef`: The weighting factor for the MAML-inspired meta-loss.
    *   `es_reg_coef`: The weighting factor for the evolutionary regularization loss.
4.  **Empirical Validation**: Using the tuned hyperparameters, the Adaptive Synergy framework is rigorously evaluated against two standard baselines across multiple random seeds:
    *   **PPO (DRL Only)**: A standard PPO agent without meta-learning or ES components.
    *   **MAML-PPO**: PPO augmented with the MAML meta-loss but without ES regularization.
5.  **Analysis**: Performance is evaluated through learning curves and key robustness metrics, including performance drop upon a task shift and the number of episodes required to recover to the pre-shift performance level.

## Results

### Tuned Hyperparameters

The Optuna study, configured to maximize post-shift adaptation performance, identified the following optimal hyperparameter values, which are stored in `best_params_pointmass.json`:

```json
{
    "ppo_lr": 0.00011926897548624919,
    "meta_loss_coef": 0.03606579574729759,
    "es_reg_coef": 0.000231712987243348
}
```

### Performance Analysis

As demonstrated by the plots generated in the validation phase of the notebook, the tuned **Adaptive Synergy** framework exhibits superior adaptation capabilities compared to the baselines. The quantitative analysis shows a statistically significant reduction in performance drop at task shifts and a faster recovery to baseline performance.

## Usage

### Prerequisites

The required Python libraries can be installed via pip:

```bash
pip install gymnasium torch numpy matplotlib pandas higher tqdm optuna
```

### Execution

1.  **Clone the repository** and navigate to the directory.
2.  **Open and run the `point_mass_tuning.ipynb` notebook** in a Jupyter environment.

The notebook is structured to run sequentially:
*   **Hyperparameter Tuning (Optional)**: If `best_params_pointmass.json` is not present in the directory, the notebook will first execute the Optuna hyperparameter search. This process is computationally intensive and may take a significant amount of time.
*   **Validation and Analysis**: If `best_params_pointmass.json` is present, the notebook will skip the tuning step, load the provided hyperparameters, and proceed directly to the full validation experiments and plot generation.
