# Multi-Objective Optimization of CNNs

This repository contains a dynamic, multi-objective optimization (MOO) pipeline designed to discover optimal Convolutional Neural Network (CNN) architectures for image classification on the Fashion-MNIST dataset. 

Instead of traditional single-objective optimization (which only maximizes accuracy), this project simultaneously optimizes for three conflicting physical objectives:
1. **Maximize Classification Accuracy** (Test set)
2. **Minimize Inference Latency** (Hardware execution time in ms)
3. **Minimize Model Complexity** (Total trainable parameters)

## Algorithms Compared
The project features a direct comparison between two state-of-the-art Multi-Objective optimization algorithms:
*   **NSGA-II (Evolutionary):** Uses non-dominated sorting and crowding distance to evolve a population of models.
*   **MOTPE (Bayesian):** Multi-Objective Tree-structured Parzen Estimator, which builds a probabilistic surrogate model to intelligently sample the highly complex hyperparameter space.

## Repository Structure
```text
MOML_Project/
├── moo_nsga2.ipynb            # Jupyter notebook running the NSGA-II algorithm
├── motpe/
│   └── moo_motpe.ipynb        # Jupyter notebook running the MOTPE algorithm
├── latest_report/             # Contains the final LaTeX report and trade-off analysis
│   ├── report.tex
│   └── figures/               # Contains 3D and 2D Pareto visualization plots
├── requirements.txt           # Python dependencies
└── README.md
```

## Key Insights
*   **Bayesian Sample Efficiency:** Within a strictly constrained budget of 60 trials, MOTPE significantly outperformed NSGA-II by finding a denser, more evenly spaced Pareto front with a 26% larger hypervolume.
*   **Non-Obvious GPU Trade-offs:** The Pareto fronts successfully demonstrated that shrinking a model's parameter count to the absolute extreme forces the adoption of "deep but narrow" networks. Because GPUs struggle with sequential processing compared to parallel wide-matrix operations, compressing the model size actively sacrifices inference speed.

## How to Run
1. Install the required dependencies:
   ```bash
   pip install torch torchvision pandas numpy matplotlib optuna jupyter
   ```
2. Launch Jupyter Notebook or open the `.ipynb` files in your preferred IDE (e.g., VS Code).
3. The notebooks are designed to execute the entire pipeline end-to-end: downloading the dataset, running the 60-trial optimization, extracting the Pareto front, computing metrics (Hypervolume, Spacing, Spread), and generating all 2D/3D visualizations.