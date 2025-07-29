#!/usr/bin/env python3
"""
CM3020 AI Coursework Part B - Mountain Climbing Genetic Algorithm
Enhanced version with comprehensive experiment configurations and advanced genetic decoding
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Import the modular components
from mountain_environment import MountainEnvironment
from mountain_population import MountainPopulation
from mountain_fitness import MountainFitnessEvaluator
from results_manager import ResultsManager

class MountainClimbingGA:
    """
    Enhanced genetic algorithm controller for mountain climbing evolution
    Includes advanced genetic decoding and comprehensive experiment configurations
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.environment = None
        self.population = None
        self.fitness_evaluator = None
        self.results_manager = None
        self.generation_data = []
        
        # Setup logging with Windows-compatible format
        self.setup_logging()
        
        # Default configuration values
        self.default_config = {
            'experiment_name': 'mountain_climbing_experiment',
            'population_size': 50,
            'generations': 100,
            'gene_count_range': (3, 8),
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'mutation_amount': 0.25,
            'elite_size': 5,
            'tournament_size': 3,
            'shrink_rate': 0.1,
            'grow_rate': 0.05,
            'arena_size': 20,
            'mountain_height': 5,
            'simulation_time': 10,
            'simulation_steps': 2400,
            'gui': False,
            'save_interval': 10,
            'plot_interval': 20,
            'target_fitness': 100.0,  # Increased default target
            'mountain_type': 'gaussian_pyramid',
            'min_generations': 20,    # Minimum generations before early stopping
            'convergence_patience': 15,  # Wait before checking convergence
            'adaptive_mutation': False,
            'immigrant_rate': 0.0,
            'evolution_mode': 'full',  # For advanced genetic decoding
            'fitness_objectives': ['climbing'],
            'objective_weights': [1.0]
        }
        
        # Merge configurations with intelligent defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Auto-adjust target fitness based on generations to prevent early stopping
        self._adjust_target_fitness()
    
    def _adjust_target_fitness(self):
        """Adjust target fitness to ensure proper evolution"""
        generations = self.config['generations']
        current_target = self.config['target_fitness']
        
        # Set higher targets for longer experiments
        if generations >= 200:
            suggested_target = max(current_target, 200.0)
        elif generations >= 150:
            suggested_target = max(current_target, 150.0)
        elif generations >= 100:
            suggested_target = max(current_target, 120.0)
        else:
            suggested_target = max(current_target, 80.0)
        
        self.config['target_fitness'] = suggested_target
    
    def setup_logging(self):
        """Setup logging system with Windows-compatible formatting"""
        os.makedirs('logs', exist_ok=True)
        
        log_filename = f"logs/experiment_{self.config.get('experiment_name', 'default')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging with UTF-8 encoding and no emoji
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ],
            force=True  # Override any existing configuration
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting experiment: {self.config.get('experiment_name', 'default')}")
        self.logger.info(f"Target fitness adjusted to: {self.config['target_fitness']}")
    
    def setup_environment(self):
        """Setup the mountain climbing environment"""
        self.logger.info("Setting up mountain climbing environment...")
        
        self.environment = MountainEnvironment(
            gui=self.config['gui'],
            arena_size=self.config['arena_size'],
            mountain_height=self.config['mountain_height'],
            time_limit=self.config['simulation_time']
        )
        
        self.environment.initialize_physics()
        self.environment.create_arena()
        self.environment.load_mountain(f"shapes/{self.config['mountain_type']}.urdf")
        
        self.logger.info("Environment setup complete")
    
    def setup_population(self):
        """Setup the population"""
        self.logger.info("Setting up population...")
        
        self.population = MountainPopulation(
            population_size=self.config['population_size'],
            gene_count_range=self.config['gene_count_range']
        )
        
        self.population.initialize_random_population()
        self.logger.info(f"Population created with {self.config['population_size']} creatures")
    
    def setup_fitness_evaluator(self):
        """Setup fitness evaluation system"""
        self.fitness_evaluator = MountainFitnessEvaluator(
            simulation_steps=self.config['simulation_steps'],
            arena_size=self.config['arena_size'],
            mountain_height=self.config['mountain_height']
        )
        
        # Apply custom fitness weights if specified
        if 'fitness_objectives' in self.config and len(self.config['fitness_objectives']) > 1:
            self._setup_multi_objective_fitness()
    
    def _setup_multi_objective_fitness(self):
        """Setup multi-objective fitness evaluation"""
        objectives = self.config['fitness_objectives']
        weights = self.config.get('objective_weights', [1.0/len(objectives)] * len(objectives))
        
        # Custom weight setup for different objectives
        custom_weights = {}
        for i, objective in enumerate(objectives):
            weight = weights[i] if i < len(weights) else 1.0/len(objectives)
            
            if objective == 'climbing':
                custom_weights.update({
                    'height_climbed': 20.0 * weight,
                    'final_height': 10.0 * weight
                })
            elif objective == 'stability':
                custom_weights.update({
                    'stability': 15.0 * weight,
                    'center_proximity': 8.0 * weight
                })
            elif objective == 'efficiency':
                custom_weights.update({
                    'movement_quality': 12.0 * weight,
                    'height_efficiency': 10.0 * weight
                })
            elif objective == 'speed':
                custom_weights.update({
                    'upward_progress': 15.0 * weight,
                    'time_bonus': 5.0 * weight
                })
        
        if custom_weights:
            self.fitness_evaluator.set_custom_weights(custom_weights)
            self.logger.info(f"Multi-objective fitness setup: {objectives}")
    
    def setup_results_manager(self):
        """Setup results management system"""
        self.results_manager = ResultsManager(
            experiment_name=self.config['experiment_name'],
            config=self.config
        )
    
    def run_evolution(self) -> Dict:
        """Run the complete evolutionary experiment with enhanced features"""
        self.logger.info("Starting Mountain Climbing Evolution")
        self.logger.info(f"Experiment: {self.config['experiment_name']}")
        self.logger.info(f"Population: {self.config['population_size']}")
        self.logger.info(f"Generations: {self.config['generations']}")
        self.logger.info(f"Gene count range: {self.config['gene_count_range']}")
        self.logger.info(f"Target fitness: {self.config['target_fitness']}")
        self.logger.info(f"Min generations: {self.config['min_generations']}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Setup all components
            self.setup_environment()
            self.setup_population()
            self.setup_fitness_evaluator()
            self.setup_results_manager()
            
            # Evolution loop
            for generation in range(self.config['generations']):
                gen_start_time = time.time()
                
                self.logger.info(f"--- Generation {generation} ---")
                
                # Evaluate population
                fitness_scores = self.evaluate_population(generation)
                
                # Calculate statistics
                gen_stats = self.calculate_generation_stats(generation, fitness_scores)
                self.generation_data.append(gen_stats)
                
                # Log progress
                self.log_generation_progress(gen_stats, time.time() - gen_start_time)
                
                # Save best creature periodically
                if generation % self.config['save_interval'] == 0:
                    self.save_generation_data(generation)
                
                # Plot progress periodically
                if generation % self.config['plot_interval'] == 0 and generation > 0:
                    self.results_manager.plot_progress(self.generation_data, generation)
                
                # Enhanced early termination check
                should_terminate, reason = self._check_termination_conditions(generation, gen_stats)
                if should_terminate:
                    self.logger.info(f"Early termination: {reason}")
                    break
                
                # Evolve to next generation (skip on last generation)
                if generation < self.config['generations'] - 1:
                    self.evolve_next_generation(fitness_scores, generation)
            
            # Compile final results
            total_time = time.time() - start_time
            final_results = self.compile_final_results(total_time)
            
            # Save all results
            self.save_final_results(final_results)
            
            self.logger.info(f"Evolution completed in {total_time/60:.1f} minutes!")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Evolution failed: {str(e)}")
            raise
            
        finally:
            # Cleanup
            if self.environment:
                self.environment.cleanup()
    
    def _check_termination_conditions(self, generation: int, gen_stats: Dict) -> Tuple[bool, str]:
        """Enhanced termination condition checking"""
        # Don't terminate before minimum generations
        if generation < self.config['min_generations']:
            return False, ""
        
        # Check target fitness with patience
        if gen_stats['max_fitness'] >= self.config['target_fitness']:
            if generation >= self.config['min_generations']:
                return True, f"Target fitness {self.config['target_fitness']:.1f} reached at generation {generation}"
        
        # Check convergence with patience
        if generation >= self.config['convergence_patience']:
            if self._check_convergence():
                return True, f"Population converged at generation {generation}"
        
        return False, ""
    
    def _check_convergence(self) -> bool:
        """Check if population has converged"""
        if len(self.generation_data) < self.config['convergence_patience']:
            return False
        
        recent_fitness = [gen['max_fitness'] for gen in self.generation_data[-self.config['convergence_patience']:]]
        
        # Check if fitness has plateaued
        if len(set([round(f, 2) for f in recent_fitness])) <= 2:  # Very little variation
            fitness_variance = np.var(recent_fitness)
            if fitness_variance < 1.0:  # Low variance indicates convergence
                return True
        
        return False
    
    def evaluate_population(self, generation: int) -> List[float]:
        """Evaluate the entire population"""
        self.logger.info(f"Evaluating {len(self.population.creatures)} creatures...")
        
        fitness_scores = []
        
        for idx, creature in enumerate(self.population.creatures):
            if idx % 10 == 0:
                self.logger.info(f"  Evaluating creature {idx + 1}/{len(self.population.creatures)}")
            
            try:
                # Reset environment
                self.environment.reset_environment()
                
                # Spawn creature
                creature_id = self.environment.spawn_creature(creature)
                
                if creature_id is None or creature_id < 0:
                    fitness = 0
                    metrics = self.fitness_evaluator.get_default_metrics()
                else:
                    # Simulate and evaluate
                    metrics = self.environment.simulate_creature(
                        creature, 
                        creature_id, 
                        self.config['simulation_steps']
                    )
                    fitness = self.fitness_evaluator.calculate_fitness(metrics, creature)
                
                # Store results
                creature.fitness = fitness
                creature.climbing_metrics = metrics
                fitness_scores.append(fitness)
                
            except Exception as e:
                self.logger.warning(f"Error evaluating creature {idx}: {e}")
                creature.fitness = 0
                creature.climbing_metrics = self.fitness_evaluator.get_default_metrics()
                fitness_scores.append(0)
        
        return fitness_scores
    
    def calculate_generation_stats(self, generation: int, fitness_scores: List[float]) -> Dict:
        """Calculate statistics for the current generation"""
        if not fitness_scores:
            return {}
        
        heights = [creature.climbing_metrics.get('max_height', 0) for creature in self.population.creatures]
        distances = [creature.climbing_metrics.get('distance_travelled', 0) for creature in self.population.creatures]
        num_links = [len(creature.get_expanded_links()) for creature in self.population.creatures]
        
        stats = {
            'generation': generation,
            'max_fitness': np.max(fitness_scores),
            'mean_fitness': np.mean(fitness_scores),
            'min_fitness': np.min(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'max_height': np.max(heights),
            'mean_height': np.mean(heights),
            'max_distance': np.max(distances),
            'mean_distance': np.mean(distances),
            'num_links': num_links,
            'mean_links': np.mean(num_links),
            'max_links': np.max(num_links),
            'population_size': len(fitness_scores)
        }
        
        return stats
    
    def log_generation_progress(self, stats: Dict, gen_time: float):
        """Log progress for current generation"""
        self.logger.info(
            f"Gen {stats['generation']:3d}: "
            f"Max Fitness={stats['max_fitness']:.3f}, "
            f"Mean Fitness={stats['mean_fitness']:.3f}, "
            f"Max Height={stats['max_height']:.3f}, "
            f"Mean Links={stats['mean_links']:.1f}, "
            f"Time={gen_time:.1f}s"
        )
    
    def evolve_next_generation(self, fitness_scores: List[float], generation: int):
        """Create the next generation with enhanced features"""
        # Adaptive mutation rate
        mutation_rate = self.config['mutation_rate']
        if self.config.get('adaptive_mutation', False):
            mutation_rate = self.population.adaptive_mutation_rate(mutation_rate)
            if generation % 20 == 0:
                self.logger.info(f"Adaptive mutation rate: {mutation_rate:.3f}")
        
        # Tournament selection
        selected = self.population.tournament_selection(
            tournament_size=self.config['tournament_size']
        )
        
        new_population = []
        
        # Elitism - keep best creatures
        elite_creatures = sorted(
            self.population.creatures, 
            key=lambda x: x.fitness, 
            reverse=True
        )[:self.config['elite_size']]
        
        for elite in elite_creatures:
            new_creature = self.population.clone_creature(elite)
            new_population.append(new_creature)
        
        # Generate offspring
        while len(new_population) < self.config['population_size']:
            parent1 = np.random.choice(selected)
            parent2 = np.random.choice(selected)
            
            # Crossover
            if np.random.random() < self.config['crossover_rate']:
                child = self.population.crossover_creatures(parent1, parent2)
            else:
                child = self.population.clone_creature(parent1)
            
            # Mutations
            if np.random.random() < mutation_rate:
                child = self.population.mutate_creature(
                    child, 
                    mutation_rate,
                    self.config['mutation_amount']
                )
            
            # Structural mutations
            if np.random.random() < self.config['shrink_rate']:
                child = self.population.shrink_mutate_creature(child)
            
            if np.random.random() < self.config['grow_rate']:
                child = self.population.grow_mutate_creature(child)
            
            new_population.append(child)
        
        # Add random immigrants if specified
        immigrant_rate = self.config.get('immigrant_rate', 0.0)
        if immigrant_rate > 0:
            num_immigrants = max(1, int(self.config['population_size'] * immigrant_rate))
            if generation % 10 == 0:  # Every 10 generations
                self.population.introduce_random_immigrants(num_immigrants)
                self.logger.info(f"Introduced {num_immigrants} random immigrants")
        
        # Update population
        self.population.creatures = new_population[:self.config['population_size']]
        self.population.generation += 1
    
    def save_generation_data(self, generation: int):
        """Save generation data and best creature"""
        # Save best creature
        best_creature = max(self.population.creatures, key=lambda x: x.fitness)
        filename = f"best_creatures/elite_{self.config['experiment_name']}_gen_{generation}.csv"
        os.makedirs('best_creatures', exist_ok=True)
        
        import genome
        genome.Genome.to_csv(best_creature.dna, filename)
        
        # Save generation data
        self.results_manager.save_generation_data(self.generation_data)
    
    def compile_final_results(self, total_time: float) -> Dict:
        """Compile final experimental results"""
        best_creature = max(self.population.creatures, key=lambda x: x.fitness)
        
        results = {
            'experiment_name': self.config['experiment_name'],
            'config': self.config,
            'total_time': total_time,
            'generations_completed': len(self.generation_data),
            'best_fitness': best_creature.fitness,
            'best_metrics': best_creature.climbing_metrics,
            'generation_data': self.generation_data,
            'convergence_analysis': self.analyze_convergence(),
            'performance_summary': self.summarize_performance(),
            'population_diversity': self.population.analyze_population_diversity(),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def analyze_convergence(self) -> Dict:
        """Enhanced convergence analysis"""
        if len(self.generation_data) < 10:
            return {'convergence_generation': len(self.generation_data)}
        
        fitness_values = [gen['max_fitness'] for gen in self.generation_data]
        
        # Find convergence point
        convergence_gen = len(fitness_values)
        for i in range(10, len(fitness_values)):
            recent_improvement = (fitness_values[i] - fitness_values[i-10]) / max(fitness_values[i-10], 0.001)
            if recent_improvement < 0.01:  # Less than 1% improvement
                convergence_gen = i
                break
        
        # Additional convergence metrics
        final_variance = np.var(fitness_values[-10:]) if len(fitness_values) >= 10 else 0
        improvement_rate = self.calculate_improvement_rate()
        
        return {
            'convergence_generation': convergence_gen,
            'final_improvement_rate': improvement_rate,
            'fitness_variance': final_variance,
            'plateau_detected': final_variance < 1.0 and improvement_rate < 0.1,
            'convergence_quality': 'high' if final_variance < 0.5 else 'medium' if final_variance < 2.0 else 'low'
        }
    
    def calculate_improvement_rate(self) -> float:
        """Calculate improvement rate over last 10 generations"""
        if len(self.generation_data) < 10:
            return 0
        
        recent_fitness = [gen['max_fitness'] for gen in self.generation_data[-10:]]
        x = np.arange(len(recent_fitness))
        
        if len(recent_fitness) >= 2:
            slope, _ = np.polyfit(x, recent_fitness, 1)
            return slope
        
        return 0
    
    def summarize_performance(self) -> Dict:
        """Enhanced performance summary"""
        if not self.generation_data:
            return {}
        
        all_fitness = [gen['max_fitness'] for gen in self.generation_data]
        all_heights = [gen['max_height'] for gen in self.generation_data]
        all_mean_fitness = [gen['mean_fitness'] for gen in self.generation_data]
        
        return {
            'initial_fitness': all_fitness[0],
            'final_fitness': all_fitness[-1],
            'fitness_improvement': all_fitness[-1] - all_fitness[0],
            'relative_improvement': (all_fitness[-1] - all_fitness[0]) / max(all_fitness[0], 0.001) * 100,
            'max_fitness_achieved': max(all_fitness),
            'max_height_achieved': max(all_heights),
            'average_final_fitness': np.mean(all_fitness[-10:]) if len(all_fitness) >= 10 else all_fitness[-1],
            'population_improvement': all_mean_fitness[-1] - all_mean_fitness[0] if all_mean_fitness else 0,
            'consistency_score': 1.0 - (np.std(all_fitness[-10:]) / max(np.mean(all_fitness[-10:]), 0.001)) if len(all_fitness) >= 10 else 0
        }
    
    def save_final_results(self, results: Dict):
        """Save all final results"""
        # Save best creature
        best_creature = max(self.population.creatures, key=lambda x: x.fitness)
        best_filename = f"best_creatures/best_{self.config['experiment_name']}.csv"
        
        import genome
        genome.Genome.to_csv(best_creature.dna, best_filename)
        
        # Save all results
        self.results_manager.save_final_results(results)
        
        # Generate final plots
        self.results_manager.generate_final_plots(self.generation_data)
        
        # Generate summary report
        self.results_manager.generate_summary_report(results)


def get_experiment_configs():
    """Get comprehensive experiment configurations"""
    configs = {
        # ORIGINAL BASIC EXPERIMENTS (with improved parameters)
        'small_population': {
            'experiment_name': 'small_population',
            'population_size': 20,
            'generations': 80,  # Increased
            'gene_count_range': (3, 6),
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'simulation_time': 8,
            'target_fitness': 120.0,  # Increased
            'min_generations': 30
        },
        'medium_population': {
            'experiment_name': 'medium_population',
            'population_size': 40,
            'generations': 100,  # Increased
            'gene_count_range': (4, 8),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 10,
            'target_fitness': 150.0,  # Increased
            'min_generations': 40
        },
        'large_population': {
            'experiment_name': 'large_population',
            'population_size': 60,
            'generations': 120,  # Increased
            'gene_count_range': (5, 10),
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 180.0,  # Increased
            'min_generations': 50
        },
        'high_mutation': {
            'experiment_name': 'high_mutation',
            'population_size': 40,
            'generations': 100,  # Increased
            'gene_count_range': (4, 8),
            'mutation_rate': 0.25,
            'crossover_rate': 0.6,
            'simulation_time': 12,
            'target_fitness': 150.0,  # Increased
            'min_generations': 40
        },
        
        # NEW HIGH-VALUE EXPERIMENTS
        'very_large_population_80': {
            'experiment_name': 'very_large_population_80',
            'population_size': 80,
            'generations': 150,
            'gene_count_range': (6, 12),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 200.0,
            'elite_size': 8,
            'tournament_size': 4,
            'min_generations': 60
        },
        'very_large_population_100': {
            'experiment_name': 'very_large_population_100',
            'population_size': 100,
            'generations': 180,
            'gene_count_range': (8, 15),
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'simulation_time': 15,
            'target_fitness': 250.0,
            'elite_size': 10,
            'tournament_size': 5,
            'min_generations': 80
        },
        'massive_population_120': {
            'experiment_name': 'massive_population_120',
            'population_size': 120,
            'generations': 200,
            'gene_count_range': (10, 20),
            'mutation_rate': 0.1,
            'crossover_rate': 0.9,
            'simulation_time': 18,
            'target_fitness': 300.0,
            'elite_size': 12,
            'tournament_size': 6,
            'min_generations': 100
        },
        
        # EXTENDED GENERATION EXPERIMENTS
        'marathon_evolution_200': {
            'experiment_name': 'marathon_evolution_200',
            'population_size': 60,
            'generations': 200,
            'gene_count_range': (6, 12),
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'simulation_time': 15,
            'target_fitness': 250.0,
            'adaptive_mutation': True,
            'min_generations': 80
        },
        'marathon_evolution_300': {
            'experiment_name': 'marathon_evolution_300',
            'population_size': 80,
            'generations': 300,
            'gene_count_range': (8, 16),
            'mutation_rate': 0.1,
            'crossover_rate': 0.9,
            'simulation_time': 20,
            'target_fitness': 350.0,
            'adaptive_mutation': True,
            'immigrant_rate': 0.02,
            'min_generations': 120
        },
        
        # EXTREME MUTATION EXPERIMENTS
        'extreme_mutation_30': {
            'experiment_name': 'extreme_mutation_30',
            'population_size': 50,
            'generations': 120,
            'gene_count_range': (4, 8),
            'mutation_rate': 0.3,
            'crossover_rate': 0.7,
            'simulation_time': 12,
            'target_fitness': 180.0,
            'shrink_rate': 0.15,
            'grow_rate': 0.08,
            'min_generations': 50
        },
        'extreme_mutation_40': {
            'experiment_name': 'extreme_mutation_40',
            'population_size': 60,
            'generations': 150,
            'gene_count_range': (5, 10),
            'mutation_rate': 0.4,
            'crossover_rate': 0.6,
            'simulation_time': 15,
            'target_fitness': 200.0,
            'shrink_rate': 0.2,
            'grow_rate': 0.1,
            'min_generations': 60
        },
        'chaos_evolution_50': {
            'experiment_name': 'chaos_evolution_50',
            'population_size': 40,
            'generations': 180,
            'gene_count_range': (3, 12),
            'mutation_rate': 0.5,
            'crossover_rate': 0.5,
            'simulation_time': 15,
            'target_fitness': 220.0,
            'shrink_rate': 0.25,
            'grow_rate': 0.15,
            'immigrant_rate': 0.05,
            'min_generations': 70
        },
        
        # COMPLEX GENE RANGE EXPERIMENTS
        'complex_creatures_large': {
            'experiment_name': 'complex_creatures_large',
            'population_size': 70,
            'generations': 180,
            'gene_count_range': (12, 25),
            'mutation_rate': 0.08,
            'crossover_rate': 0.9,
            'simulation_time': 25,
            'target_fitness': 280.0,
            'elite_size': 7,
            'mutation_amount': 0.15,
            'min_generations': 80
        },
        'mega_complex_creatures': {
            'experiment_name': 'mega_complex_creatures',
            'population_size': 50,
            'generations': 250,
            'gene_count_range': (20, 40),
            'mutation_rate': 0.05,
            'crossover_rate': 0.95,
            'simulation_time': 30,
            'target_fitness': 400.0,
            'elite_size': 10,
            'mutation_amount': 0.1,
            'min_generations': 120
        },
        
        # BALANCED PARAMETER COMBINATIONS
        'balanced_high_performance': {
            'experiment_name': 'balanced_high_performance',
            'population_size': 90,
            'generations': 200,
            'gene_count_range': (8, 18),
            'mutation_rate': 0.18,
            'crossover_rate': 0.82,
            'simulation_time': 18,
            'target_fitness': 300.0,
            'elite_size': 9,
            'tournament_size': 5,
            'adaptive_mutation': True,
            'min_generations': 80
        },
        'exploration_focused': {
            'experiment_name': 'exploration_focused',
            'population_size': 100,
            'generations': 150,
            'gene_count_range': (6, 20),
            'mutation_rate': 0.35,
            'crossover_rate': 0.65,
            'simulation_time': 15,
            'target_fitness': 250.0,
            'shrink_rate': 0.2,
            'grow_rate': 0.2,
            'immigrant_rate': 0.1,
            'min_generations': 60
        },
        'exploitation_focused': {
            'experiment_name': 'exploitation_focused',
            'population_size': 120,
            'generations': 280,
            'gene_count_range': (10, 15),
            'mutation_rate': 0.05,
            'crossover_rate': 0.95,
            'simulation_time': 20,
            'target_fitness': 450.0,
            'elite_size': 20,
            'tournament_size': 8,
            'min_generations': 150
        },
        
        # ADVANCED GENETIC DECODING EXPERIMENTS
        'motor_control_only': {
            'experiment_name': 'motor_control_only',
            'population_size': 60,
            'generations': 120,
            'gene_count_range': (4, 8),
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 150.0,
            'evolution_mode': 'motor_only',
            'fixed_body_design': 'simple_quadruped',
            'min_generations': 50
        },
        'body_shape_evolution': {
            'experiment_name': 'body_shape_evolution',
            'population_size': 50,
            'generations': 180,
            'gene_count_range': (6, 12),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 15,
            'target_fitness': 200.0,
            'evolution_mode': 'body_only',
            'fixed_motor_pattern': 'oscillating',
            'min_generations': 70
        },
        'leg_design_evolution': {
            'experiment_name': 'leg_design_evolution',
            'population_size': 70,
            'generations': 150,
            'gene_count_range': (8, 16),
            'mutation_rate': 0.18,
            'crossover_rate': 0.8,
            'simulation_time': 18,
            'target_fitness': 220.0,
            'evolution_mode': 'legs_only',
            'fixed_torso_design': 'stable_base',
            'min_generations': 60
        },
        'partial_evolution_torso_legs': {
            'experiment_name': 'partial_evolution_torso_legs',
            'population_size': 80,
            'generations': 170,
            'gene_count_range': (10, 18),
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'simulation_time': 20,
            'target_fitness': 250.0,
            'evolution_mode': 'torso_legs_only',
            'fixed_components': ['head', 'tail'],
            'min_generations': 70
        },
        'modular_evolution': {
            'experiment_name': 'modular_evolution',
            'population_size': 60,
            'generations': 200,
            'gene_count_range': (12, 24),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 22,
            'target_fitness': 280.0,
            'evolution_mode': 'modular',
            'evolvable_modules': ['locomotion', 'stability', 'climbing'],
            'fixed_modules': ['core_structure'],
            'min_generations': 80
        },
        
        # MULTI-OBJECTIVE EXPERIMENTS
        'multi_objective_climbing_stability': {
            'experiment_name': 'multi_objective_climbing_stability',
            'population_size': 80,
            'generations': 150,
            'gene_count_range': (8, 16),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 20,
            'target_fitness': 200.0,
            'fitness_objectives': ['climbing', 'stability'],
            'objective_weights': [0.7, 0.3],
            'min_generations': 60
        },
        'multi_objective_full': {
            'experiment_name': 'multi_objective_full',
            'population_size': 100,
            'generations': 180,
            'gene_count_range': (10, 20),
            'mutation_rate': 0.12,
            'crossover_rate': 0.85,
            'simulation_time': 25,
            'target_fitness': 280.0,
            'fitness_objectives': ['climbing', 'stability', 'efficiency', 'speed'],
            'objective_weights': [0.4, 0.3, 0.2, 0.1],
            'adaptive_mutation': True,
            'min_generations': 80
        },
        
        # SPECIALIZED SHAPE EXPERIMENTS
        'cylindrical_parts_evolution': {
            'experiment_name': 'cylindrical_parts_evolution',
            'population_size': 50,
            'generations': 120,
            'gene_count_range': (6, 12),
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'simulation_time': 15,
            'target_fitness': 180.0,
            'part_shapes': ['cylinder', 'capsule'],
            'connection_types': ['hinge', 'universal'],
            'min_generations': 50
        },
        'mixed_shapes_evolution': {
            'experiment_name': 'mixed_shapes_evolution',
            'population_size': 70,
            'generations': 150,
            'gene_count_range': (8, 16),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 18,
            'target_fitness': 220.0,
            'part_shapes': ['box', 'cylinder', 'sphere', 'capsule'],
            'connection_types': ['hinge', 'universal', 'ball'],
            'min_generations': 60
        },
        
        # HYBRID EXPERIMENTS
        'hybrid_adaptive_complex': {
            'experiment_name': 'hybrid_adaptive_complex',
            'population_size': 100,
            'generations': 250,
            'gene_count_range': (15, 30),
            'mutation_rate': 0.25,
            'crossover_rate': 0.75,
            'simulation_time': 25,
            'target_fitness': 400.0,
            'evolution_mode': 'hybrid',
            'adaptive_mutation': True,
            'immigrant_rate': 0.05,
            'elite_size': 15,
            'min_generations': 120
        }
    }
    
    return configs


def get_experiment_groups():
    """Get experiments grouped by category for systematic analysis"""
    return {
        'basic_experiments': [
            'small_population', 'medium_population', 'large_population', 'high_mutation'
        ],
        'population_scaling': [
            'very_large_population_80', 'very_large_population_100', 'massive_population_120'
        ],
        'generation_scaling': [
            'marathon_evolution_200', 'marathon_evolution_300'
        ],
        'mutation_scaling': [
            'extreme_mutation_30', 'extreme_mutation_40', 'chaos_evolution_50'
        ],
        'complexity_scaling': [
            'complex_creatures_large', 'mega_complex_creatures'
        ],
        'parameter_combinations': [
            'balanced_high_performance', 'exploration_focused', 'exploitation_focused'
        ],
        'advanced_genetic_decoding': [
            'motor_control_only', 'body_shape_evolution', 'leg_design_evolution',
            'partial_evolution_torso_legs', 'modular_evolution'
        ],
        'multi_objective': [
            'multi_objective_climbing_stability', 'multi_objective_full'
        ],
        'shape_experiments': [
            'cylindrical_parts_evolution', 'mixed_shapes_evolution'
        ],
        'hybrid_experiments': [
            'hybrid_adaptive_complex'
        ]
    }


def get_analysis_focused_configs():
    """Configurations specifically for parameter analysis"""
    return {
        # MUTATION RATE ANALYSIS
        'mutation_analysis_low': {
            'experiment_name': 'mutation_analysis_low',
            'population_size': 60,
            'generations': 120,
            'gene_count_range': (5, 10),
            'mutation_rate': 0.05,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 150.0,
            'min_generations': 50
        },
        'mutation_analysis_medium': {
            'experiment_name': 'mutation_analysis_medium',
            'population_size': 60,
            'generations': 120,
            'gene_count_range': (5, 10),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 150.0,
            'min_generations': 50
        },
        'mutation_analysis_high': {
            'experiment_name': 'mutation_analysis_high',
            'population_size': 60,
            'generations': 120,
            'gene_count_range': (5, 10),
            'mutation_rate': 0.30,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 150.0,
            'min_generations': 50
        },
        
        # POPULATION SIZE ANALYSIS
        'population_analysis_small': {
            'experiment_name': 'population_analysis_small',
            'population_size': 30,
            'generations': 150,
            'gene_count_range': (5, 10),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 180.0,
            'min_generations': 60
        },
        'population_analysis_large': {
            'experiment_name': 'population_analysis_large',
            'population_size': 90,
            'generations': 150,
            'gene_count_range': (5, 10),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 180.0,
            'min_generations': 60
        },
        
        # GENE COMPLEXITY ANALYSIS
        'complexity_analysis_simple': {
            'experiment_name': 'complexity_analysis_simple',
            'population_size': 50,
            'generations': 120,
            'gene_count_range': (3, 6),
            'mutation_rate': 0.15,
            'crossover_rate': 0.8,
            'simulation_time': 12,
            'target_fitness': 140.0,
            'min_generations': 50
        },
        'complexity_analysis_complex': {
            'experiment_name': 'complexity_analysis_complex',
            'population_size': 50,
            'generations': 120,
            'gene_count_range': (10, 20),
            'mutation_rate': 0.10,
            'crossover_rate': 0.8,
            'simulation_time': 15,
            'target_fitness': 160.0,
            'min_generations': 50
        }
    }


def create_experiment_schedule(experiment_groups=None, max_parallel=2):
    """Create a schedule for running experiments efficiently"""
    if experiment_groups is None:
        experiment_groups = ['basic_experiments', 'population_scaling', 'mutation_scaling']
    
    groups = get_experiment_groups()
    schedule = []
    
    for group_name in experiment_groups:
        if group_name in groups:
            experiments = groups[group_name]
            
            # Split into batches for parallel execution
            for i in range(0, len(experiments), max_parallel):
                batch = experiments[i:i+max_parallel]
                schedule.append({
                    'group': group_name,
                    'batch': len(schedule) + 1,
                    'experiments': batch,
                    'estimated_time_hours': estimate_experiment_time(batch)
                })
    
    return schedule


def estimate_experiment_time(experiment_names):
    """Estimate total time for a batch of experiments"""
    configs = get_experiment_configs()
    total_time = 0
    
    for exp_name in experiment_names:
        if exp_name in configs:
            config = configs[exp_name]
            # Rough estimation: population_size * generations * simulation_time / 3600
            time_estimate = (config['population_size'] * config['generations'] * 
                           config.get('simulation_time', 10)) / 3600
            total_time += time_estimate
    
    return round(total_time, 1)


def validate_experiment_config(config):
    """Validate experiment configuration"""
    required_fields = ['experiment_name', 'population_size', 'generations', 'gene_count_range']
    
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"
    
    # Validate ranges
    if config['population_size'] <= 0:
        return False, "Population size must be positive"
    
    if config['generations'] <= 0:
        return False, "Generations must be positive"
    
    if not isinstance(config['gene_count_range'], (list, tuple)) or len(config['gene_count_range']) != 2:
        return False, "Gene count range must be a tuple/list of 2 values"
    
    if config['gene_count_range'][0] >= config['gene_count_range'][1]:
        return False, "Gene count range minimum must be less than maximum"
    
    # Validate rates
    for rate_field in ['mutation_rate', 'crossover_rate']:
        if rate_field in config:
            rate = config[rate_field]
            if not 0 <= rate <= 1:
                return False, f"{rate_field} must be between 0 and 1"
    
    return True, "Valid configuration"


def main():
    """Enhanced main function with comprehensive experiment support"""
    parser = argparse.ArgumentParser(description='CM3020 Mountain Climbing GA - Enhanced Version')
    parser.add_argument('--experiment', '-e', 
                       help='Experiment configuration to run (use --list to see all)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available experiments')
    parser.add_argument('--group', '-gr',
                       choices=['basic_experiments', 'population_scaling', 'generation_scaling', 
                               'mutation_scaling', 'complexity_scaling', 'parameter_combinations',
                               'advanced_genetic_decoding', 'multi_objective', 'shape_experiments', 
                               'hybrid_experiments', 'all'],
                       help='Run a group of experiments')
    parser.add_argument('--gui', '-g', action='store_true',
                       help='Enable PyBullet GUI')
    parser.add_argument('--population-size', '--pop', type=int,
                       help='Population size override')
    parser.add_argument('--generations', '--gen', type=int,
                       help='Number of generations override')
    parser.add_argument('--mutation-rate', '--mut', type=float,
                       help='Mutation rate override')
    parser.add_argument('--target-fitness', '--target', type=float,
                       help='Target fitness override')
    parser.add_argument('--mountain-type', '--mountain', type=str,
                       default='gaussian_pyramid',
                       help='Mountain type (gaussian_pyramid, realistic_mountain, etc.)')
    parser.add_argument('--output-dir', '--out', type=str,
                       default='.',
                       help='Output directory')
    parser.add_argument('--schedule', '-s', action='store_true',
                       help='Show experiment schedule')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('best_creatures', exist_ok=True)
    
    # Get experiment configurations
    configs = get_experiment_configs()
    analysis_configs = get_analysis_focused_configs()
    all_configs = {**configs, **analysis_configs}
    groups = get_experiment_groups()
    
    # Handle list request
    if args.list:
        print("\n=== AVAILABLE EXPERIMENTS ===")
        for group_name, experiments in groups.items():
            print(f"\n{group_name.upper().replace('_', ' ')}:")
            for exp in experiments:
                if exp in all_configs:
                    config = all_configs[exp]
                    print(f"  • {exp:30s} - Pop:{config['population_size']:3d}, "
                          f"Gen:{config['generations']:3d}, "
                          f"Mut:{config['mutation_rate']:.2f}")
        
        print(f"\n=== ANALYSIS EXPERIMENTS ===")
        for exp_name, config in analysis_configs.items():
            print(f"  • {exp_name:30s} - Pop:{config['population_size']:3d}, "
                  f"Gen:{config['generations']:3d}, "
                  f"Mut:{config['mutation_rate']:.2f}")
        
        print(f"\nTotal experiments available: {len(all_configs)}")
        return
    
    # Handle schedule request
    if args.schedule:
        print("\n=== EXPERIMENT SCHEDULE ===")
        all_group_names = list(groups.keys())
        schedule = create_experiment_schedule(all_group_names, max_parallel=2)
        
        total_time = 0
        for batch in schedule:
            print(f"\nBatch {batch['batch']} ({batch['group']}):")
            print(f"  Experiments: {', '.join(batch['experiments'])}")
            print(f"  Estimated time: {batch['estimated_time_hours']:.1f} hours")
            total_time += batch['estimated_time_hours']
        
        print(f"\nTotal estimated time: {total_time:.1f} hours ({total_time/24:.1f} days)")
        return
    
    # Determine which experiments to run
    experiment_list = []
    
    if args.group:
        if args.group == 'all':
            for group_experiments in groups.values():
                experiment_list.extend(group_experiments)
        elif args.group in groups:
            experiment_list = groups[args.group]
        else:
            print(f"Unknown experiment group: {args.group}")
            return
    elif args.experiment:
        if args.experiment in all_configs:
            experiment_list = [args.experiment]
        elif args.experiment == 'all':
            experiment_list = list(all_configs.keys())
        else:
            print(f"Unknown experiment: {args.experiment}")
            print("Use --list to see available experiments")
            return
    else:
        # Default to basic experiments
        experiment_list = ['small_population']
        print("No experiment specified, running default: small_population")
        print("Use --list to see all available experiments")
    
    # Run experiments
    all_results = {}
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"RUNNING {len(experiment_list)} EXPERIMENTS")
    print(f"{'='*80}")
    
    for i, exp_name in enumerate(experiment_list):
        if exp_name not in all_configs:
            print(f"Skipping unknown experiment: {exp_name}")
            continue
        
        config = all_configs[exp_name].copy()
        
        # Apply command line overrides
        if args.gui:
            config['gui'] = True
        if args.population_size:
            config['population_size'] = args.population_size
        if args.generations:
            config['generations'] = args.generations
        if args.mutation_rate:
            config['mutation_rate'] = args.mutation_rate
        if args.target_fitness:
            config['target_fitness'] = args.target_fitness
        if args.mountain_type:
            config['mountain_type'] = args.mountain_type
        
        # Validate configuration
        is_valid, validation_msg = validate_experiment_config(config)
        if not is_valid:
            print(f"Invalid configuration for {exp_name}: {validation_msg}")
            continue
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i+1}/{len(experiment_list)}: {config['experiment_name']}")
        print(f"{'='*80}")
        print(f"Population: {config['population_size']}")
        print(f"Generations: {config['generations']}")
        print(f"Mutation Rate: {config['mutation_rate']}")
        print(f"Target Fitness: {config['target_fitness']}")
        print(f"Min Generations: {config.get('min_generations', 'N/A')}")
        if 'evolution_mode' in config:
            print(f"Evolution Mode: {config['evolution_mode']}")
        if 'fitness_objectives' in config and len(config['fitness_objectives']) > 1:
            print(f"Objectives: {', '.join(config['fitness_objectives'])}")
        
        # Run experiment
        try:
            ga = MountainClimbingGA(config)
            results = ga.run_evolution()
            all_results[exp_name] = results
            
            print(f"\n✅ Experiment {config['experiment_name']} completed!")
            print(f"   Best Fitness: {results['best_fitness']:.3f}")
            print(f"   Best Height: {results['best_metrics'].get('max_height', 0):.3f}")
            print(f"   Generations: {results['generations_completed']}")
            print(f"   Time: {results['total_time']/60:.1f} minutes")
            
        except Exception as e:
            print(f"❌ Experiment {config['experiment_name']} failed: {e}")
            continue
    
    # Generate comparative analysis if multiple experiments
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("GENERATING COMPARATIVE ANALYSIS")
        print(f"{'='*80}")
        
        try:
            from comparative_analysis import ComparativeAnalysis
            analyzer = ComparativeAnalysis()
            analyzer.compare_experiments(all_results)
            print("✅ Comparative analysis completed!")
        except ImportError:
            print("❌ Comparative analysis module not available")
        except Exception as e:
            print(f"❌ Comparative analysis failed: {e}")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print(f"Experiments run: {len(all_results)}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Average time per experiment: {total_time/max(len(all_results), 1)/60:.1f} minutes")
    print("\nResults available in:")
    print("  📁 results/           - Experiment data and reports")
    print("  📊 graphs/            - Fitness plots and visualizations") 
    print("  📝 logs/              - Detailed execution logs")
    print("  🧬 best_creatures/    - Best evolved creature genomes")
    if len(all_results) > 1:
        print("  📈 comparative_analysis/ - Cross-experiment analysis")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()