# Prompt: Implement the Optimization for Data Science PDF

## Instructions

1. Do not use built-in functions for the core algorithms. Write each method from its definition (for example, gradient descent, line search, matrix operations, or any solver step) using basic Python and loops. Libraries may be used only for plotting and for loading data.
2. Implement all the code that the given PDF describes, wherever it is possible. Cover every algorithm, worked example, and exercise in the PDF, and do not skip any that can be coded.
3. Generate figures and plots wherever the work allows it. Examples include convergence curves, objective value against iteration, the path of the iterates, and comparisons between methods. Save each figure to file.
4. Keep all implementation code in one folder named `code/` at the repository root. Do not create nested or mixed folders for code. Each algorithm or worked example gets its own separate Python file inside `code/`, named by lecture or topic (for example, `code/lecture2_gradient_descent.py`). Do not put several unrelated algorithms in one file.
5. Put shared helpers (config, seeding, plotting, and the timestamped results folder) in separate files inside `code/` as well. The lecture files import these helpers, and helper code is not duplicated across lecture files.

## Output requirements

- Keep the experiment config in one place and set random seeds. Log each seed.
- Write all results to a `results/` folder with a timestamp in the folder name (`results/<timestamp>/`).
- Save every plot as a PNG in the same results folder.
- For each part of the PDF, state which question or section it answers and what it produced.
- After running the code, report what was actually run and the output. If something cannot be run, say so.
- Whenever code is updated, the report must also be updated to match the code, its results, and its figures.
- Leave author names as placeholders (for example, `[Author 1 Name]`) in the report.
