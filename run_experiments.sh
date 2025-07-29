#!/bin/bash
echo "Starting CM3020 Experiment Suite"
echo

echo "Phase 1: Basic Experiments"
python mountain_climbing_ga.py --group basic_experiments
echo

echo "Phase 2: Population Scaling"
python mountain_climbing_ga.py --group population_scaling
echo

echo "Phase 3: Advanced Genetic Decoding"
python mountain_climbing_ga.py --group advanced_genetic_decoding
echo

echo "All experiments completed!"