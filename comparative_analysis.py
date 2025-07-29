#!/usr/bin/env python3
"""
CM3020 AI Coursework Part B - Comparative Analysis
Cross-experiment comparison and analysis system
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import logging


class ComparativeAnalysis:
    """
    Comprehensive comparative analysis system for comparing multiple experiments
    Generates cross-experiment visualizations and statistical analysis
    """
    
    def __init__(self, output_dir: str = "comparative_analysis"):
        self.output_dir = output_dir
    
        # Setup logging FIRST
        self.logger = logging.getLogger(__name__)
        
        # Then setup directories
        self.setup_directories()
        
        # Data storage
        self.experiments = {}
        self.comparison_data = {}
        
    def setup_directories(self):
        """Create output directories for comparative analysis"""
        self.directories = {
            'base': self.output_dir,
            'graphs': os.path.join(self.output_dir, 'comparison_graphs'),
            'data': os.path.join(self.output_dir, 'comparison_data'),
            'reports': os.path.join(self.output_dir, 'comparison_reports')
        }
        
        for directory in self.directories.values():
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Created comparative analysis directories in {self.output_dir}")
    
    def compare_experiments(self, experiment_results: Dict[str, Dict]):
        """Compare multiple experiments and generate comprehensive analysis"""
        self.experiments = experiment_results
        
        self.logger.info(f"Comparing {len(experiment_results)} experiments")
        
        # Generate comparison data
        self.comparison_data = self._extract_comparison_data()
        
        # Generate comparative visualizations
        self._generate_comparison_plots()
        
        # Generate statistical analysis
        self._generate_statistical_analysis()
        
        # Generate comprehensive report
        self._generate_comparison_report()
        
        self.logger.info("Comparative analysis completed")
    
    def _extract_comparison_data(self) -> Dict:
        """Extract and structure data for comparison"""
        comparison_data = {
            'experiment_names': list(self.experiments.keys()),
            'configs': {},
            'performance_metrics': {},
            'evolution_data': {},
            'convergence_data': {},
            'final_stats': {}
        }
        
        for exp_name, exp_data in self.experiments.items():
            # Extract configuration
            config = exp_data.get('config', {})
            comparison_data['configs'][exp_name] = {
                'population_size': config.get('population_size', 0),
                'generations': config.get('generations', 0),
                'mutation_rate': config.get('mutation_rate', 0),
                'gene_count_range': config.get('gene_count_range', (0, 0))
            }
            
            # Extract performance metrics
            comparison_data['performance_metrics'][exp_name] = {
                'best_fitness': exp_data.get('best_fitness', 0),
                'best_height': exp_data.get('best_metrics', {}).get('max_height', 0),
                'total_time': exp_data.get('total_time', 0),
                'generations_completed': exp_data.get('generations_completed', 0)
            }
            
            # Extract evolution data
            gen_data = exp_data.get('generation_data', [])
            if gen_data:
                comparison_data['evolution_data'][exp_name] = {
                    'generations': [g['generation'] for g in gen_data],
                    'max_fitness': [g['max_fitness'] for g in gen_data],
                    'mean_fitness': [g['mean_fitness'] for g in gen_data],
                    'std_fitness': [g['std_fitness'] for g in gen_data],
                    'max_heights': [g.get('max_height', 0) for g in gen_data],
                    'mean_heights': [g.get('mean_height', 0) for g in gen_data]
                }
            
            # Extract convergence data
            convergence = exp_data.get('convergence_analysis', {})
            comparison_data['convergence_data'][exp_name] = {
                'convergence_generation': convergence.get('convergence_generation', len(gen_data)),
                'improvement_rate': convergence.get('final_improvement_rate', 0),
                'fitness_variance': convergence.get('fitness_variance', 0)
            }
            
            # Extract final statistics
            if gen_data:
                final_gen = gen_data[-1]
                comparison_data['final_stats'][exp_name] = {
                    'final_max_fitness': final_gen.get('max_fitness', 0),
                    'final_mean_fitness': final_gen.get('mean_fitness', 0),
                    'final_std_fitness': final_gen.get('std_fitness', 0),
                    'final_diversity': final_gen.get('std_fitness', 0) / max(final_gen.get('mean_fitness', 1), 1),
                    'final_complexity': final_gen.get('mean_links', 0)
                }
        
        return comparison_data
    
    def _generate_comparison_plots(self):
        """Generate comprehensive comparison plots"""
        # Main performance comparison
        self._plot_performance_comparison()
        
        # Evolution trajectories
        self._plot_evolution_trajectories()
        
        # Configuration vs performance analysis
        self._plot_config_vs_performance()
        
        # Convergence analysis
        self._plot_convergence_comparison()
        
        # Efficiency analysis
        self._plot_efficiency_analysis()
        
        # Detailed statistical comparison
        self._plot_statistical_comparison()
    
    def _plot_performance_comparison(self):
        """Plot main performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Experiment Performance Comparison', fontsize=16, fontweight='bold')
        
        exp_names = self.comparison_data['experiment_names']
        colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
        
        # Best fitness comparison
        best_fitness = [self.comparison_data['performance_metrics'][name]['best_fitness'] 
                       for name in exp_names]
        
        bars1 = ax1.bar(range(len(exp_names)), best_fitness, color=colors, alpha=0.8)
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Best Fitness Achieved')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, best_fitness):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Best height comparison
        best_heights = [self.comparison_data['performance_metrics'][name]['best_height'] 
                       for name in exp_names]
        
        bars2 = ax2.bar(range(len(exp_names)), best_heights, color=colors, alpha=0.8)
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Best Height (units)')
        ax2.set_title('Best Climbing Height')
        ax2.set_xticks(range(len(exp_names)))
        ax2.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, best_heights):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Convergence speed comparison
        convergence_gens = [self.comparison_data['convergence_data'][name]['convergence_generation'] 
                           for name in exp_names]
        
        bars3 = ax3.bar(range(len(exp_names)), convergence_gens, color=colors, alpha=0.8)
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('Convergence Generation')
        ax3.set_title('Convergence Speed (Lower = Faster)')
        ax3.set_xticks(range(len(exp_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars3, convergence_gens):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Efficiency comparison (fitness per minute)
        total_times = [self.comparison_data['performance_metrics'][name]['total_time'] / 60 
                      for name in exp_names]
        efficiency = [fit / max(time, 1) for fit, time in zip(best_fitness, total_times)]
        
        bars4 = ax4.bar(range(len(exp_names)), efficiency, color=colors, alpha=0.8)
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Fitness per Minute')
        ax4.set_title('Training Efficiency')
        ax4.set_xticks(range(len(exp_names)))
        ax4.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars4, efficiency):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        filename = 'performance_comparison.png'
        filepath = os.path.join(self.directories['graphs'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Generated performance comparison plot")
    
    def _plot_evolution_trajectories(self):
        """Plot evolution trajectories for all experiments"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Evolution Trajectories Comparison', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.comparison_data['experiment_names'])))
        
        for i, exp_name in enumerate(self.comparison_data['experiment_names']):
            if exp_name not in self.comparison_data['evolution_data']:
                continue
                
            evo_data = self.comparison_data['evolution_data'][exp_name]
            color = colors[i]
            label = exp_name.replace('_', ' ').title()
            
            # Max fitness evolution
            ax1.plot(evo_data['generations'], evo_data['max_fitness'], 
                    color=color, linewidth=2, label=label, marker='o', markersize=3)
            
            # Mean fitness evolution
            ax2.plot(evo_data['generations'], evo_data['mean_fitness'], 
                    color=color, linewidth=2, label=label, marker='s', markersize=3)
            
            # Height performance
            ax3.plot(evo_data['generations'], evo_data['max_heights'], 
                    color=color, linewidth=2, label=label, marker='^', markersize=3)
            
            # Population diversity (std fitness)
            ax4.plot(evo_data['generations'], evo_data['std_fitness'], 
                    color=color, linewidth=2, label=label, marker='d', markersize=3)
        
        # Configure subplots
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Maximum Fitness')
        ax1.set_title('Best Fitness Evolution')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Mean Fitness')
        ax2.set_title('Population Mean Fitness')
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Maximum Height (units)')
        ax3.set_title('Climbing Performance')
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Fitness Standard Deviation')
        ax4.set_title('Population Diversity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = 'evolution_trajectories.png'
        filepath = os.path.join(self.directories['graphs'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Generated evolution trajectories plot")
    
    def _plot_config_vs_performance(self):
        """Plot configuration parameters vs performance"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Configuration vs Performance Analysis', fontsize=16, fontweight='bold')
        
        exp_names = self.comparison_data['experiment_names']
        
        # Extract data
        pop_sizes = [self.comparison_data['configs'][name]['population_size'] for name in exp_names]
        mutation_rates = [self.comparison_data['configs'][name]['mutation_rate'] for name in exp_names]
        generations = [self.comparison_data['configs'][name]['generations'] for name in exp_names]
        best_fitness = [self.comparison_data['performance_metrics'][name]['best_fitness'] for name in exp_names]
        
        # Population size vs performance
        ax1.scatter(pop_sizes, best_fitness, s=100, alpha=0.7, c=range(len(exp_names)), cmap='viridis')
        ax1.set_xlabel('Population Size')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Population Size vs Performance')
        ax1.grid(True, alpha=0.3)
        
        # Add experiment labels
        for i, name in enumerate(exp_names):
            ax1.annotate(name.replace('_', '\n'), (pop_sizes[i], best_fitness[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Mutation rate vs performance
        ax2.scatter(mutation_rates, best_fitness, s=100, alpha=0.7, c=range(len(exp_names)), cmap='viridis')
        ax2.set_xlabel('Mutation Rate')
        ax2.set_ylabel('Best Fitness')
        ax2.set_title('Mutation Rate vs Performance')
        ax2.grid(True, alpha=0.3)
        
        for i, name in enumerate(exp_names):
            ax2.annotate(name.replace('_', '\n'), (mutation_rates[i], best_fitness[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Generations vs performance
        ax3.scatter(generations, best_fitness, s=100, alpha=0.7, c=range(len(exp_names)), cmap='viridis')
        ax3.set_xlabel('Number of Generations')
        ax3.set_ylabel('Best Fitness')
        ax3.set_title('Generations vs Performance')
        ax3.grid(True, alpha=0.3)
        
        for i, name in enumerate(exp_names):
            ax3.annotate(name.replace('_', '\n'), (generations[i], best_fitness[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Multi-dimensional analysis (bubble chart)
        convergence_gens = [self.comparison_data['convergence_data'][name]['convergence_generation'] 
                           for name in exp_names]
        
        # Bubble size based on convergence speed (smaller = faster convergence)
        bubble_sizes = [1000 / max(cg, 1) for cg in convergence_gens]
        
        scatter = ax4.scatter(pop_sizes, mutation_rates, s=bubble_sizes, alpha=0.6, 
                             c=best_fitness, cmap='RdYlGn', edgecolors='black')
        ax4.set_xlabel('Population Size')
        ax4.set_ylabel('Mutation Rate')
        ax4.set_title('Multi-Parameter Analysis\n(Color=Fitness, Size∝1/Convergence)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Best Fitness')
        
        for i, name in enumerate(exp_names):
            ax4.annotate(name.replace('_', '\n'), (pop_sizes[i], mutation_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        filename = 'config_vs_performance.png'
        filepath = os.path.join(self.directories['graphs'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Generated configuration vs performance plot")
    
    def _plot_convergence_comparison(self):
        """Plot convergence analysis comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Convergence Analysis Comparison', fontsize=16, fontweight='bold')
        
        exp_names = self.comparison_data['experiment_names']
        colors = plt.cm.Set2(np.linspace(0, 1, len(exp_names)))
        
        # Convergence generation comparison
        convergence_gens = [self.comparison_data['convergence_data'][name]['convergence_generation'] 
                           for name in exp_names]
        
        bars1 = ax1.bar(range(len(exp_names)), convergence_gens, color=colors, alpha=0.8)
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Convergence Generation')
        ax1.set_title('Convergence Speed Comparison')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Normalized convergence curves
        for i, exp_name in enumerate(exp_names):
            if exp_name not in self.comparison_data['evolution_data']:
                continue
                
            evo_data = self.comparison_data['evolution_data'][exp_name]
            max_fitness = evo_data['max_fitness']
            
            # Normalize to [0, 1] scale
            if max_fitness and max(max_fitness) > min(max_fitness):
                normalized = [(f - min(max_fitness)) / (max(max_fitness) - min(max_fitness)) 
                             for f in max_fitness]
                normalized_gens = [g / max(evo_data['generations']) for g in evo_data['generations']]
                
                ax2.plot(normalized_gens, normalized, color=colors[i], linewidth=2, 
                        label=exp_name.replace('_', ' ').title())
        
        ax2.set_xlabel('Normalized Generation (0-1)')
        ax2.set_ylabel('Normalized Fitness (0-1)')
        ax2.set_title('Normalized Convergence Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Final diversity comparison
        final_diversity = [self.comparison_data['final_stats'][name]['final_diversity'] 
                          for name in exp_names]
        
        bars3 = ax3.bar(range(len(exp_names)), final_diversity, color=colors, alpha=0.8)
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('Final Diversity (CV)')
        ax3.set_title('Final Population Diversity')
        ax3.set_xticks(range(len(exp_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Improvement rate comparison
        improvement_rates = [self.comparison_data['convergence_data'][name]['improvement_rate'] 
                            for name in exp_names]
        
        bars4 = ax4.bar(range(len(exp_names)), improvement_rates, color=colors, alpha=0.8)
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Final Improvement Rate')
        ax4.set_title('Learning Rate at Convergence')
        ax4.set_xticks(range(len(exp_names)))
        ax4.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = 'convergence_comparison.png'
        filepath = os.path.join(self.directories['graphs'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Generated convergence comparison plot")
    
    def _plot_efficiency_analysis(self):
        """Plot efficiency analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Efficiency Analysis Comparison', fontsize=16, fontweight='bold')
        
        exp_names = self.comparison_data['experiment_names']
        
        # Calculate efficiency metrics
        best_fitness = [self.comparison_data['performance_metrics'][name]['best_fitness'] 
                       for name in exp_names]
        total_times = [self.comparison_data['performance_metrics'][name]['total_time'] / 60 
                      for name in exp_names]  # Convert to minutes
        generations_completed = [self.comparison_data['performance_metrics'][name]['generations_completed'] 
                               for name in exp_names]
        pop_sizes = [self.comparison_data['configs'][name]['population_size'] for name in exp_names]
        
        # Total evaluations (proxy for computational cost)
        total_evaluations = [gens * pop for gens, pop in zip(generations_completed, pop_sizes)]
        
        # Fitness per minute
        fitness_per_minute = [fit / max(time, 1) for fit, time in zip(best_fitness, total_times)]
        
        # Fitness per evaluation
        fitness_per_eval = [fit / max(evals, 1) for fit, evals in zip(best_fitness, total_evaluations)]
        
        # Time efficiency
        ax1.bar(range(len(exp_names)), fitness_per_minute, alpha=0.8, color='skyblue')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Fitness per Minute')
        ax1.set_title('Time Efficiency')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Computational efficiency
        ax2.bar(range(len(exp_names)), fitness_per_eval, alpha=0.8, color='lightgreen')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Fitness per Evaluation')
        ax2.set_title('Computational Efficiency')
        ax2.set_xticks(range(len(exp_names)))
        ax2.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=0)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Resource utilization scatter
        ax3.scatter(total_evaluations, best_fitness, s=100, alpha=0.7, c=range(len(exp_names)), cmap='viridis')
        ax3.set_xlabel('Total Evaluations')
        ax3.set_ylabel('Best Fitness')
        ax3.set_title('Resource Utilization vs Performance')
        ax3.grid(True, alpha=0.3)
        
        for i, name in enumerate(exp_names):
            ax3.annotate(name.replace('_', '\n'), (total_evaluations[i], best_fitness[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Efficiency ranking radar chart
        # Normalize all metrics to 0-1 scale
        metrics = {
            'Fitness': best_fitness,
            'Time Eff.': fitness_per_minute,
            'Comp. Eff.': fitness_per_eval,
            'Speed': [1/(cg+1) for cg in [self.comparison_data['convergence_data'][name]['convergence_generation'] for name in exp_names]]
        }
        
        # Normalize each metric
        for metric, values in metrics.items():
            max_val = max(values)
            min_val = min(values)
            if max_val > min_val:
                metrics[metric] = [(v - min_val) / (max_val - min_val) for v in values]
            else:
                metrics[metric] = [1.0] * len(values)
        
        # Create radar chart for top 3 experiments
        top_3_idx = sorted(range(len(best_fitness)), key=lambda i: best_fitness[i], reverse=True)[:3]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
        
        colors_radar = ['red', 'blue', 'green']
        for i, exp_idx in enumerate(top_3_idx):
            values = [metrics[metric][exp_idx] for metric in metrics.keys()]
            values += [values[0]]  # Complete the circle
            
            ax4.plot(angles, values, 'o-', linewidth=2, 
                    label=exp_names[exp_idx].replace('_', ' ').title(), 
                    color=colors_radar[i])
            ax4.fill(angles, values, alpha=0.25, color=colors_radar[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics.keys())
        ax4.set_ylim(0, 1)
        ax4.set_title('Top 3 Experiments - Efficiency Radar')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax4.grid(True)
        
        plt.tight_layout()
        
        filename = 'efficiency_analysis.png'
        filepath = os.path.join(self.directories['graphs'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Generated efficiency analysis plot")
    
    def _plot_statistical_comparison(self):
        """Plot detailed statistical comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Comparison Analysis', fontsize=16, fontweight='bold')
        
        exp_names = self.comparison_data['experiment_names']
        
        # Create performance metrics matrix for heatmap
        metrics = ['Best Fitness', 'Best Height', 'Convergence Speed', 'Efficiency']
        
        # Collect and normalize data
        best_fitness = [self.comparison_data['performance_metrics'][name]['best_fitness'] for name in exp_names]
        best_heights = [self.comparison_data['performance_metrics'][name]['best_height'] for name in exp_names]
        convergence_speed = [1/(self.comparison_data['convergence_data'][name]['convergence_generation']+1) for name in exp_names]
        efficiency = [fit / max(self.comparison_data['performance_metrics'][name]['total_time']/60, 1) 
                     for name, fit in zip(exp_names, best_fitness)]
        
        data_matrix = np.array([best_fitness, best_heights, convergence_speed, efficiency])
        
        # Normalize each metric to 0-1 scale
        for i in range(data_matrix.shape[0]):
            row = data_matrix[i]
            if np.max(row) > np.min(row):
                data_matrix[i] = (row - np.min(row)) / (np.max(row) - np.min(row))
            else:
                data_matrix[i] = np.ones_like(row)
        
        # Performance heatmap
        im = ax1.imshow(data_matrix, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=45)
        ax1.set_yticks(range(len(metrics)))
        ax1.set_yticklabels(metrics)
        ax1.set_title('Normalized Performance Heatmap')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(exp_names)):
                text = ax1.text(j, i, f'{data_matrix[i, j]:.2f}', 
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax1)
        
        # Box plot comparison of final fitness distributions
        # Note: This would require access to full population data
        # For now, we'll show final statistics
        final_max = [self.comparison_data['final_stats'][name]['final_max_fitness'] for name in exp_names]
        final_mean = [self.comparison_data['final_stats'][name]['final_mean_fitness'] for name in exp_names]
        final_std = [self.comparison_data['final_stats'][name]['final_std_fitness'] for name in exp_names]
        
        x_pos = range(len(exp_names))
        ax2.bar(x_pos, final_max, alpha=0.7, label='Max Fitness', color='blue')
        ax2.bar(x_pos, final_mean, alpha=0.7, label='Mean Fitness', color='orange')
        ax2.errorbar(x_pos, final_mean, yerr=final_std, fmt='none', color='black', capsize=5)
        
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Final Generation Statistics')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Ranking analysis
        rankings = {}
        for metric, values in zip(metrics, data_matrix):
            rankings[metric] = sorted(range(len(exp_names)), key=lambda i: values[i], reverse=True)
        
        # Calculate average ranking
        avg_rankings = []
        for i in range(len(exp_names)):
            ranks = [rankings[metric].index(i) + 1 for metric in metrics]  # +1 for 1-based ranking
            avg_rankings.append(np.mean(ranks))
        
        bars = ax3.bar(range(len(exp_names)), avg_rankings, alpha=0.8, color='lightcoral')
        ax3.set_xlabel('Experiment')
        ax3.set_ylabel('Average Ranking (Lower = Better)')
        ax3.set_title('Overall Performance Ranking')
        ax3.set_xticks(range(len(exp_names)))
        ax3.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add ranking values on bars
        for bar, rank in zip(bars, avg_rankings):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{rank:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Statistical significance analysis (simplified)
        # Compare all pairs of experiments
        comparison_matrix = np.zeros((len(exp_names), len(exp_names)))
        
        for i in range(len(exp_names)):
            for j in range(len(exp_names)):
                if i != j:
                    # Simple comparison based on normalized performance difference
                    perf_i = np.mean(data_matrix[:, i])
                    perf_j = np.mean(data_matrix[:, j])
                    comparison_matrix[i, j] = perf_i - perf_j
        
        im2 = ax4.imshow(comparison_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(exp_names)))
        ax4.set_xticklabels([name.replace('_', '\n') for name in exp_names], rotation=45)
        ax4.set_yticks(range(len(exp_names)))
        ax4.set_yticklabels([name.replace('_', '\n') for name in exp_names])
        ax4.set_title('Pairwise Performance Comparison\n(Row - Column)')
        
        plt.colorbar(im2, ax=ax4)
        
        plt.tight_layout()
        
        filename = 'statistical_comparison.png'
        filepath = os.path.join(self.directories['graphs'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Generated statistical comparison plot")
    
    def _generate_statistical_analysis(self):
        """Generate detailed statistical analysis"""
        exp_names = self.comparison_data['experiment_names']
        
        # Performance statistics
        best_fitness = [self.comparison_data['performance_metrics'][name]['best_fitness'] for name in exp_names]
        
        stats = {
            'experiment_count': len(exp_names),
            'performance_statistics': {
                'mean_best_fitness': np.mean(best_fitness),
                'std_best_fitness': np.std(best_fitness),
                'min_best_fitness': np.min(best_fitness),
                'max_best_fitness': np.max(best_fitness),
                'range_best_fitness': np.max(best_fitness) - np.min(best_fitness)
            },
            'rankings': self._calculate_rankings(),
            'correlations': self._calculate_correlations(),
            'significance_tests': self._perform_significance_tests()
        }
        
        # Save statistical analysis
        stats_file = os.path.join(self.directories['data'], 'statistical_analysis.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        self.logger.info("Generated statistical analysis")
        
        return stats
    
    def _calculate_rankings(self) -> Dict:
        """Calculate rankings for different metrics"""
        exp_names = self.comparison_data['experiment_names']
        
        rankings = {}
        
        # Fitness ranking
        fitness_values = [self.comparison_data['performance_metrics'][name]['best_fitness'] for name in exp_names]
        fitness_ranking = sorted(range(len(exp_names)), key=lambda i: fitness_values[i], reverse=True)
        rankings['fitness'] = [exp_names[i] for i in fitness_ranking]
        
        # Height ranking
        height_values = [self.comparison_data['performance_metrics'][name]['best_height'] for name in exp_names]
        height_ranking = sorted(range(len(exp_names)), key=lambda i: height_values[i], reverse=True)
        rankings['height'] = [exp_names[i] for i in height_ranking]
        
        # Convergence speed ranking (lower generation = better)
        convergence_values = [self.comparison_data['convergence_data'][name]['convergence_generation'] for name in exp_names]
        convergence_ranking = sorted(range(len(exp_names)), key=lambda i: convergence_values[i])
        rankings['convergence_speed'] = [exp_names[i] for i in convergence_ranking]
        
        # Efficiency ranking
        efficiency_values = []
        for name in exp_names:
            fitness = self.comparison_data['performance_metrics'][name]['best_fitness']
            time = max(self.comparison_data['performance_metrics'][name]['total_time'] / 60, 1)
            efficiency_values.append(fitness / time)
        
        efficiency_ranking = sorted(range(len(exp_names)), key=lambda i: efficiency_values[i], reverse=True)
        rankings['efficiency'] = [exp_names[i] for i in efficiency_ranking]
        
        return rankings
    
    def _calculate_correlations(self) -> Dict:
        """Calculate correlations between configuration parameters and performance"""
        exp_names = self.comparison_data['experiment_names']
        
        # Extract data
        pop_sizes = [self.comparison_data['configs'][name]['population_size'] for name in exp_names]
        mutation_rates = [self.comparison_data['configs'][name]['mutation_rate'] for name in exp_names]
        generations = [self.comparison_data['configs'][name]['generations'] for name in exp_names]
        best_fitness = [self.comparison_data['performance_metrics'][name]['best_fitness'] for name in exp_names]
        
        correlations = {}
        
        if len(set(pop_sizes)) > 1:  # Only calculate if there's variation
            correlations['population_size_vs_fitness'] = np.corrcoef(pop_sizes, best_fitness)[0, 1]
        
        if len(set(mutation_rates)) > 1:
            correlations['mutation_rate_vs_fitness'] = np.corrcoef(mutation_rates, best_fitness)[0, 1]
        
        if len(set(generations)) > 1:
            correlations['generations_vs_fitness'] = np.corrcoef(generations, best_fitness)[0, 1]
        
        return correlations
    
    def _perform_significance_tests(self) -> Dict:
        """Perform statistical significance tests"""
        exp_names = self.comparison_data['experiment_names']
        
        if len(exp_names) < 2:
            return {'note': 'Insufficient experiments for significance testing'}
        
        # Get fitness values
        fitness_values = [self.comparison_data['performance_metrics'][name]['best_fitness'] for name in exp_names]
        
        # Simple statistical tests
        significance_tests = {
            'fitness_variance_test': {
                'variance': np.var(fitness_values),
                'coefficient_of_variation': np.std(fitness_values) / max(np.mean(fitness_values), 0.001),
                'significant_difference': np.var(fitness_values) > (np.mean(fitness_values) * 0.1) ** 2
            },
            'best_vs_worst': {
                'best_experiment': exp_names[np.argmax(fitness_values)],
                'worst_experiment': exp_names[np.argmin(fitness_values)],
                'performance_gap': np.max(fitness_values) - np.min(fitness_values),
                'relative_improvement': (np.max(fitness_values) - np.min(fitness_values)) / max(np.min(fitness_values), 0.001)
            }
        }
        
        return significance_tests
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        exp_names = self.comparison_data['experiment_names']
        
        # Get rankings and statistics
        rankings = self._calculate_rankings()
        stats = self._generate_statistical_analysis()
        
        report_content = f"""
CM3020 AI COURSEWORK PART B - COMPARATIVE ANALYSIS REPORT
{'='*80}

GENERATED: {timestamp}
EXPERIMENTS COMPARED: {len(exp_names)}

EXECUTIVE SUMMARY
{'='*40}
This report presents a comprehensive comparative analysis of {len(exp_names)} 
mountain climbing genetic algorithm experiments. Each experiment used different
configurations to explore the optimization landscape.

EXPERIMENTS ANALYZED:
{chr(10).join([f"  • {name.replace('_', ' ').title()}" for name in exp_names])}

PERFORMANCE RANKINGS
{'='*40}

FITNESS RANKING (Best to Worst):
"""
        
        for i, exp_name in enumerate(rankings['fitness']):
            fitness = self.comparison_data['performance_metrics'][exp_name]['best_fitness']
            report_content += f"{i+1:2d}. {exp_name.replace('_', ' ').title():25s} - {fitness:.4f}\n"
        
        report_content += f"""
CLIMBING HEIGHT RANKING (Best to Worst):
"""
        
        for i, exp_name in enumerate(rankings['height']):
            height = self.comparison_data['performance_metrics'][exp_name]['best_height']
            report_content += f"{i+1:2d}. {exp_name.replace('_', ' ').title():25s} - {height:.4f} units\n"
        
        report_content += f"""
CONVERGENCE SPEED RANKING (Fastest to Slowest):
"""
        
        for i, exp_name in enumerate(rankings['convergence_speed']):
            conv_gen = self.comparison_data['convergence_data'][exp_name]['convergence_generation']
            report_content += f"{i+1:2d}. {exp_name.replace('_', ' ').title():25s} - Generation {conv_gen}\n"
        
        report_content += f"""
EFFICIENCY RANKING (Most to Least Efficient):
"""
        
        for i, exp_name in enumerate(rankings['efficiency']):
            fitness = self.comparison_data['performance_metrics'][exp_name]['best_fitness']
            time = max(self.comparison_data['performance_metrics'][exp_name]['total_time'] / 60, 1)
            efficiency = fitness / time
            report_content += f"{i+1:2d}. {exp_name.replace('_', ' ').title():25s} - {efficiency:.4f} fitness/min\n"
        
        # Configuration analysis
        report_content += f"""

CONFIGURATION ANALYSIS
{'='*40}

PARAMETER RANGES:
Population Size: {min([self.comparison_data['configs'][name]['population_size'] for name in exp_names])} - {max([self.comparison_data['configs'][name]['population_size'] for name in exp_names])}
Mutation Rate: {min([self.comparison_data['configs'][name]['mutation_rate'] for name in exp_names]):.3f} - {max([self.comparison_data['configs'][name]['mutation_rate'] for name in exp_names]):.3f}
Generations: {min([self.comparison_data['configs'][name]['generations'] for name in exp_names])} - {max([self.comparison_data['configs'][name]['generations'] for name in exp_names])}

DETAILED EXPERIMENT CONFIGURATIONS:
"""
        
        for exp_name in exp_names:
            config = self.comparison_data['configs'][exp_name]
            perf = self.comparison_data['performance_metrics'][exp_name]
            report_content += f"""
{exp_name.replace('_', ' ').title()}:
  Population Size: {config['population_size']}
  Generations: {config['generations']}
  Mutation Rate: {config['mutation_rate']:.3f}
  Best Fitness: {perf['best_fitness']:.4f}
  Best Height: {perf['best_height']:.4f} units
  Execution Time: {perf['total_time']/60:.2f} minutes
"""
        
        # Statistical analysis
        perf_stats = stats['performance_statistics']
        report_content += f"""

STATISTICAL ANALYSIS
{'='*40}

PERFORMANCE STATISTICS:
Mean Best Fitness: {perf_stats['mean_best_fitness']:.4f}
Standard Deviation: {perf_stats['std_best_fitness']:.4f}
Minimum: {perf_stats['min_best_fitness']:.4f}
Maximum: {perf_stats['max_best_fitness']:.4f}
Range: {perf_stats['range_best_fitness']:.4f}
Coefficient of Variation: {(perf_stats['std_best_fitness']/perf_stats['mean_best_fitness']*100):.2f}%

CORRELATIONS:
"""
        
        correlations = stats.get('correlations', {})
        for param, corr in correlations.items():
            param_name = param.replace('_', ' ').title()
            report_content += f"{param_name}: {corr:.4f}\n"
        
        # Key insights
        best_exp = rankings['fitness'][0]
        worst_exp = rankings['fitness'][-1]
        best_fitness = self.comparison_data['performance_metrics'][best_exp]['best_fitness']
        worst_fitness = self.comparison_data['performance_metrics'][worst_exp]['best_fitness']
        
        report_content += f"""

KEY INSIGHTS AND FINDINGS
{'='*40}

1. BEST OVERALL PERFORMER:
   Experiment: {best_exp.replace('_', ' ').title()}
   Best Fitness: {best_fitness:.4f}
   Configuration: Pop={self.comparison_data['configs'][best_exp]['population_size']}, 
                 Mut={self.comparison_data['configs'][best_exp]['mutation_rate']:.3f}, 
                 Gen={self.comparison_data['configs'][best_exp]['generations']}

2. PERFORMANCE VARIATION:
   Best vs Worst Gap: {best_fitness - worst_fitness:.4f} ({((best_fitness - worst_fitness)/worst_fitness*100):.1f}% improvement)
   
3. FASTEST CONVERGENCE:
   Experiment: {rankings['convergence_speed'][0].replace('_', ' ').title()}
   Convergence: Generation {self.comparison_data['convergence_data'][rankings['convergence_speed'][0]]['convergence_generation']}

4. MOST EFFICIENT:
   Experiment: {rankings['efficiency'][0].replace('_', ' ').title()}
   Efficiency: {self.comparison_data['performance_metrics'][rankings['efficiency'][0]]['best_fitness'] / max(self.comparison_data['performance_metrics'][rankings['efficiency'][0]]['total_time']/60, 1):.4f} fitness/minute

CONFIGURATION RECOMMENDATIONS:
{'='*40}

Based on the comparative analysis:

FOR MAXIMUM PERFORMANCE:
Use configuration from '{best_exp.replace('_', ' ').title()}' experiment:
- Population Size: {self.comparison_data['configs'][best_exp]['population_size']}
- Mutation Rate: {self.comparison_data['configs'][best_exp]['mutation_rate']:.3f}
- Generations: {self.comparison_data['configs'][best_exp]['generations']}

FOR FASTEST RESULTS:
Use configuration from '{rankings['convergence_speed'][0].replace('_', ' ').title()}' experiment for rapid prototyping.

FOR EFFICIENCY:
Use configuration from '{rankings['efficiency'][0].replace('_', ' ').title()}' experiment for resource-constrained environments.

PARAMETER INSIGHTS:
"""
        
        # Parameter insights based on correlations
        if 'population_size_vs_fitness' in correlations:
            if correlations['population_size_vs_fitness'] > 0.5:
                report_content += "• Larger population sizes tend to improve performance\n"
            elif correlations['population_size_vs_fitness'] < -0.5:
                report_content += "• Smaller population sizes may be more effective\n"
            else:
                report_content += "• Population size shows weak correlation with performance\n"
        
        if 'mutation_rate_vs_fitness' in correlations:
            if correlations['mutation_rate_vs_fitness'] > 0.5:
                report_content += "• Higher mutation rates tend to improve performance\n"
            elif correlations['mutation_rate_vs_fitness'] < -0.5:
                report_content += "• Lower mutation rates may be more effective\n"
            else:
                report_content += "• Mutation rate shows weak correlation with performance\n"
        
        report_content += f"""

FILES GENERATED:
{'='*40}
Visualization Files:
  • performance_comparison.png - Main performance metrics comparison
  • evolution_trajectories.png - Evolution curves for all experiments
  • config_vs_performance.png - Configuration parameter analysis
  • convergence_comparison.png - Convergence analysis
  • efficiency_analysis.png - Efficiency and resource utilization
  • statistical_comparison.png - Statistical analysis and rankings

Data Files:
  • statistical_analysis.json - Detailed statistical analysis
  • comparison_data.json - Complete comparison dataset
  • experiment_summary.csv - Summary table for further analysis

CONCLUSION:
{'='*40}
The comparative analysis reveals significant differences in performance across
different GA configurations for the mountain climbing task. The results provide
clear guidance for parameter selection based on specific objectives (performance,
speed, or efficiency).

{'='*80}
END OF COMPARATIVE ANALYSIS REPORT
{'='*80}
"""
        
        # Save report
        report_file = os.path.join(self.directories['reports'], 'comparative_analysis_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Save as CSV summary for easy analysis
        self._save_summary_csv()
        
        # Save complete comparison data
        comparison_file = os.path.join(self.directories['data'], 'comparison_data.json')
        with open(comparison_file, 'w') as f:
            json.dump(self.comparison_data, f, indent=2, default=str)
        
        self.logger.info("Generated comprehensive comparison report")
        
        return report_file
    
    def _save_summary_csv(self):
        """Save summary data as CSV for easy analysis"""
        exp_names = self.comparison_data['experiment_names']
        
        summary_data = []
        for exp_name in exp_names:
            config = self.comparison_data['configs'][exp_name]
            perf = self.comparison_data['performance_metrics'][exp_name]
            conv = self.comparison_data['convergence_data'][exp_name]
            final = self.comparison_data['final_stats'][exp_name]
            
            row = {
                'Experiment': exp_name,
                'Population_Size': config['population_size'],
                'Generations': config['generations'],
                'Mutation_Rate': config['mutation_rate'],
                'Best_Fitness': perf['best_fitness'],
                'Best_Height': perf['best_height'],
                'Total_Time_Minutes': perf['total_time'] / 60,
                'Convergence_Generation': conv['convergence_generation'],
                'Final_Max_Fitness': final['final_max_fitness'],
                'Final_Mean_Fitness': final['final_mean_fitness'],
                'Final_Diversity': final['final_diversity'],
                'Efficiency_Fitness_Per_Minute': perf['best_fitness'] / max(perf['total_time']/60, 1)
            }
            summary_data.append(row)
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        csv_file = os.path.join(self.directories['data'], 'experiment_summary.csv')
        df.to_csv(csv_file, index=False)
        
        self.logger.info("Saved summary CSV file")


# Test function
def test_comparative_analysis():
    """Test the comparative analysis functionality"""
    print("📊 Testing Comparative Analysis...")
    
    # Create mock experiment results
    mock_experiments = {
        'small_population': {
            'config': {'population_size': 20, 'generations': 50, 'mutation_rate': 0.1},
            'best_fitness': 25.5,
            'best_metrics': {'max_height': 4.2},
            'total_time': 300,
            'generations_completed': 50,
            'generation_data': [
                {'generation': i, 'max_fitness': 5 + i*0.4, 'mean_fitness': 3 + i*0.3, 
                 'std_fitness': 2 - i*0.02, 'max_height': 1 + i*0.08, 'mean_height': 0.5 + i*0.05}
                for i in range(50)
            ],
            'convergence_analysis': {'convergence_generation': 35, 'final_improvement_rate': 0.01}
        },
        'large_population': {
            'config': {'population_size': 60, 'generations': 100, 'mutation_rate': 0.2},
            'best_fitness': 35.8,
            'best_metrics': {'max_height': 6.1},
            'total_time': 800,
            'generations_completed': 100,
            'generation_data': [
                {'generation': i, 'max_fitness': 8 + i*0.28, 'mean_fitness': 5 + i*0.2, 
                 'std_fitness': 3 - i*0.015, 'max_height': 2 + i*0.04, 'mean_height': 1 + i*0.03}
                for i in range(100)
            ],
            'convergence_analysis': {'convergence_generation': 70, 'final_improvement_rate': 0.005}
        }
    }
    
    # Add final stats to mock data
    for exp_name, exp_data in mock_experiments.items():
        final_gen = exp_data['generation_data'][-1]
        exp_data['final_stats'] = {
            'final_max_fitness': final_gen['max_fitness'],
            'final_mean_fitness': final_gen['mean_fitness'],
            'final_std_fitness': final_gen['std_fitness'],
            'final_diversity': final_gen['std_fitness'] / max(final_gen['mean_fitness'], 1),
            'final_complexity': 5.0
        }
    
    # Create analyzer
    analyzer = ComparativeAnalysis('test_comparative_analysis')
    
    # Run comparison
    analyzer.compare_experiments(mock_experiments)
    
    print("✅ Generated comparative analysis")
    print(f"   Output directory: {analyzer.output_dir}")
    print(f"   Experiments compared: {len(mock_experiments)}")
    
    # Clean up test directory
    import shutil
    try:
        shutil.rmtree(analyzer.output_dir)
        print("✅ Cleaned up test files")
    except:
        print("⚠️ Could not clean up test files")
    
    print("✅ All comparative analysis tests completed!")


if __name__ == "__main__":
    test_comparative_analysis()