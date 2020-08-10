This project is used to simulate algorithm on Multi-Armed-Bandit problem settings.
The following two plots are generated to analyse the performance.

1. Empirical Regret accrued for different algorithms vs Time
2. Number of pulls of each arm, for different algorithms vs Time

Both the graphs also include a curve for the theoretical upper bound of each metric.

a) For running **single MAB instances**, run the unit test from test_mab_run.py
    Plots 1 and 2 are generated. When trying to run with large number of arms (>7), 
    graph 2 becomes congested with a lot of curves.
    
b) For an **average performance** of the algorithms, run the experiment_runner.py script.
    It runs multiple runs of MAB instances and averages the regret.
    Only plot 1 is generated. Plot 2 is irrelevant as the arms' means change every run.

The arms are bernoulli arms. Behaviour can be changed in arm.py
The Algorithms folder has different UCB based bandit algorithms,
 with configurable radius functions.
 
The Helper scripts are largely static classes/methods.