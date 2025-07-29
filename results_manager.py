#!/usr/bin/env python3
"""
CM3020 AI Coursework Part B - Results Management System (FIXED)
Comprehensive results saving, analysis, and visualization
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import logging


class ResultsManager:
    """
    Comprehensive results management system for saving and analyzing
    all experiment outputs, data, and visualizations
    """
    
    def __init__(self, experiment_name: str, config: Dict):
        self.experiment_name = experiment_name
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Setup logging FIRST before other operations
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.setup_directories()
        
        # Results storage
        self.generation_data = []
        self.creature_data = []
        self.fitness_data = []
        
    def setup_directories(self):
        """Create all necessary output directories"""
        self.base_dir = f"results_{self.experiment_name}_{self.timestamp}"
        
        self.directories = {
            'base': self.base_dir,
            'data': os.path.join(self.base_dir, 'data'),
            'graphs': os.path.join(self.base_dir, 'graphs'),
            'creatures': os.path.join(self.base_dir, 'best_creatures'),
            'logs': os.path.join(self.base_dir, 'logs'),
            'analysis': os.path.join(self.base_dir, 'analysis'),
            'reports': os.path.join(self.base_dir, 'reports'),
            'raw': os.path.join(self.base_dir, 'raw_data')
        }
        
        for dir_path in self.directories.values():
            os.makedirs(dir_path, exist_ok=True)
        
        self.logger.info(f"Created results directories in {self.base_dir}")
    
    def save_generation_data(self, generation_data: List[Dict]):
        """Save generation-by-generation data"""
        self.generation_data = generation_data
        
        try:
            # Save as JSON
            json_file = os.path.join(self.directories['data'], 'generation_data.json')
            with open(json_file, 'w') as f:
                json.dump(generation_data, f, indent=2, default=str)
            
            # Save as CSV for easy analysis
            if generation_data:
                df = pd.DataFrame(generation_data)
                csv_file = os.path.join(self.directories['data'], 'generation_data.csv')
                df.to_csv(csv_file, index=False)
            
            # Save as pickle for complete preservation
            pickle_file = os.path.join(self.directories['raw'], 'generation_data.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(generation_data, f)
            
            self.logger.info(f"Saved generation data: {len(generation_data)} generations")
            
        except Exception as e:
            self.logger.error(f"Error saving generation data: {e}")
    
    def save_creature_data(self, creatures: List, generation: int):
        """Save detailed creature data for a generation"""
        try:
            creature_records = []
            
            for i, creature in enumerate(creatures):
                record = {
                    'generation': generation,
                    'creature_id': i,
                    'fitness': getattr(creature, 'fitness', 0),
                    'gene_count': len(creature.dna),
                    'climbing_metrics': getattr(creature, 'climbing_metrics', {}),
                    'generation_born': getattr(creature, 'generation_born', generation)
                }
                
                # Add fitness breakdown if available
                if hasattr(creature, 'fitness_breakdown'):
                    record['fitness_breakdown'] = creature.fitness_breakdown
                
                creature_records.append(record)
            
            # Save creature data for this generation
            filename = os.path.join(
                self.directories['data'], 
                f'creatures_gen_{generation:03d}.json'
            )
            with open(filename, 'w') as f:
                json.dump(creature_records, f, indent=2, default=str)
            
            # Update accumulated creature data
            self.creature_data.extend(creature_records)
            
            self.logger.info(f"Saved creature data for generation {generation}: {len(creatures)} creatures")
            
        except Exception as e:
            self.logger.error(f"Error saving creature data: {e}")
    
    def save_best_creature(self, creature, generation: int, filename_suffix: str = ""):
        """Save best creature genome and data"""
        try:
            import genome
            
            # Save genome as CSV
            genome_filename = f"best_creature_gen_{generation:03d}{filename_suffix}.csv"
            genome_path = os.path.join(self.directories['creatures'], genome_filename)
            genome.Genome.to_csv(creature.dna, genome_path)
            
            # Save detailed creature data
            creature_data = {
                'generation': generation,
                'fitness': getattr(creature, 'fitness', 0),
                'gene_count': len(creature.dna),
                'climbing_metrics': getattr(creature, 'climbing_metrics', {}),
                'generation_born': getattr(creature, 'generation_born', generation),
                'genome_file': genome_filename,
                'timestamp': datetime.now().isoformat()
            }
            
            if hasattr(creature, 'fitness_breakdown'):
                creature_data['fitness_breakdown'] = creature.fitness_breakdown
            
            data_filename = f"best_creature_gen_{generation:03d}{filename_suffix}.json"
            data_path = os.path.join(self.directories['creatures'], data_filename)
            
            with open(data_path, 'w') as f:
                json.dump(creature_data, f, indent=2, default=str)
            
            self.logger.info(f"Saved best creature for generation {generation}: fitness {creature.fitness:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error saving best creature: {e}")
    
    def save_final_results(self, results: Dict):
        """Save comprehensive final results"""
        try:
            # Save complete results
            results_file = os.path.join(self.directories['data'], 'final_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save experiment configuration
            config_file = os.path.join(self.directories['data'], 'experiment_config.json')
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            
            # Save summary statistics
            self.save_summary_statistics(results)
            
            self.logger.info("Saved final results and configuration")
            
        except Exception as e:
            self.logger.error(f"Error saving final results: {e}")
    
    def save_summary_statistics(self, results: Dict):
        """Save summary statistics"""
        try:
            summary = {
                'experiment_name': self.experiment_name,
                'timestamp': self.timestamp,
                'config_summary': {
                    'population_size': self.config.get('population_size', 0),
                    'generations': self.config.get('generations', 0),
                    'mutation_rate': self.config.get('mutation_rate', 0),
                    'gene_count_range': self.config.get('gene_count_range', (0, 0))
                },
                'performance_summary': {
                    'best_fitness': results.get('best_fitness', 0),
                    'final_fitness': results.get('generation_data', [{}])[-1].get('max_fitness', 0) if results.get('generation_data') else 0,
                    'best_height': results.get('best_metrics', {}).get('max_height', 0),
                    'generations_completed': results.get('generations_completed', 0),
                    'total_time_minutes': results.get('total_time', 0) / 60
                },
                'convergence_info': results.get('convergence_analysis', {}),
                'files_generated': self.get_generated_files_list()
            }
            
            summary_file = os.path.join(self.directories['data'], 'experiment_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving summary statistics: {e}")
    
    def plot_progress(self, generation_data: List[Dict], current_generation: int):
        """Generate progress plots during evolution"""
        if not generation_data:
            return
        
        try:
            # Extract data
            generations = [d['generation'] for d in generation_data]
            max_fitness = [d['max_fitness'] for d in generation_data]
            mean_fitness = [d['mean_fitness'] for d in generation_data]
            std_fitness = [d['std_fitness'] for d in generation_data]
            max_heights = [d.get('max_height', 0) for d in generation_data]
            
            # Create progress plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{self.experiment_name} - Generation {current_generation} Progress', fontsize=16)
            
            # Fitness evolution
            ax1.plot(generations, max_fitness, 'b-', label='Max Fitness', linewidth=2)
            ax1.plot(generations, mean_fitness, 'r-', label='Mean Fitness', linewidth=2)
            ax1.fill_between(generations, 
                            np.array(mean_fitness) - np.array(std_fitness),
                            np.array(mean_fitness) + np.array(std_fitness),
                            alpha=0.3, color='red')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Height progression
            ax2.plot(generations, max_heights, 'g-', marker='o', markersize=3)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Maximum Height Reached')
            ax2.set_title('Climbing Performance')
            ax2.grid(True, alpha=0.3)
            
            # Population diversity
            ax3.plot(generations, std_fitness, 'purple', marker='s', markersize=3)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Fitness Standard Deviation')
            ax3.set_title('Population Diversity')
            ax3.grid(True, alpha=0.3)
            
            # Improvement rate
            if len(max_fitness) > 5:
                improvement_rates = []
                window = 5
                for i in range(window, len(max_fitness)):
                    recent = max_fitness[i-window:i]
                    rate = (max_fitness[i] - recent[0]) / window if recent[0] > 0 else 0
                    improvement_rates.append(rate)
                
                ax4.plot(generations[window:], improvement_rates, 'orange', marker='^', markersize=3)
                ax4.set_xlabel('Generation')
                ax4.set_ylabel('Fitness Improvement Rate')
                ax4.set_title('Learning Rate')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'progress_gen_{current_generation:03d}.png'
            plot_path = os.path.join(self.directories['graphs'], plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Saved progress plot for generation {current_generation}")
            
        except Exception as e:
            self.logger.error(f"Error generating progress plot: {e}")
    
    def generate_final_plots(self, generation_data: List[Dict]):
        """Generate comprehensive final plots"""
        if not generation_data:
            return
        
        try:
            # Main evolution plot
            self._plot_evolution_summary(generation_data)
            
            # Detailed analysis plots
            self._plot_detailed_analysis(generation_data)
            
            self.logger.info("Generated final plots")
            
        except Exception as e:
            self.logger.error(f"Error generating final plots: {e}")
    
    def _plot_evolution_summary(self, generation_data: List[Dict]):
        """Plot main evolution summary"""
        try:
            generations = [d['generation'] for d in generation_data]
            max_fitness = [d['max_fitness'] for d in generation_data]
            mean_fitness = [d['mean_fitness'] for d in generation_data]
            std_fitness = [d['std_fitness'] for d in generation_data]
            max_heights = [d.get('max_height', 0) for d in generation_data]
            mean_heights = [d.get('mean_height', 0) for d in generation_data]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.experiment_name} - Evolution Summary', fontsize=16, fontweight='bold')
            
            # Fitness evolution with confidence bands
            ax1.plot(generations, max_fitness, 'b-', linewidth=3, label='Max Fitness')
            ax1.plot(generations, mean_fitness, 'r-', linewidth=2, label='Mean Fitness')
            ax1.fill_between(generations, 
                            np.array(mean_fitness) - np.array(std_fitness),
                            np.array(mean_fitness) + np.array(std_fitness),
                            alpha=0.3, color='red', label='±1 Std Dev')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Height performance
            ax2.plot(generations, max_heights, 'g-', linewidth=2, label='Max Height', marker='o', markersize=4)
            ax2.plot(generations, mean_heights, 'orange', linewidth=2, label='Mean Height', marker='s', markersize=3)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Height Reached')
            ax2.set_title('Climbing Performance Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Population diversity
            ax3.plot(generations, std_fitness, 'purple', linewidth=2, marker='d', markersize=3)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Fitness Standard Deviation')
            ax3.set_title('Population Diversity')
            ax3.grid(True, alpha=0.3)
            
            # Fitness improvement histogram
            fitness_improvements = []
            for i in range(1, len(max_fitness)):
                improvement = max_fitness[i] - max_fitness[i-1]
                fitness_improvements.append(improvement)
            
            ax4.hist(fitness_improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_xlabel('Fitness Improvement per Generation')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Fitness Improvements')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = 'evolution_summary.png'
            filepath = os.path.join(self.directories['graphs'], filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Generated evolution summary plot")
            
        except Exception as e:
            self.logger.error(f"Error generating evolution summary plot: {e}")
    
    def _plot_detailed_analysis(self, generation_data: List[Dict]):
        """Plot detailed analysis"""
        try:
            if len(generation_data) < 10:
                return
            
            generations = [d['generation'] for d in generation_data]
            max_fitness = [d['max_fitness'] for d in generation_data]
            mean_links = [d.get('mean_links', 0) for d in generation_data]
            max_links = [d.get('max_links', 0) for d in generation_data]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{self.experiment_name} - Detailed Analysis', fontsize=16, fontweight='bold')
            
            # Fitness vs Complexity
            ax1.scatter(mean_links, max_fitness, alpha=0.6, c=generations, cmap='viridis')
            ax1.set_xlabel('Mean Creature Complexity (Links)')
            ax1.set_ylabel('Max Fitness')
            ax1.set_title('Performance vs Complexity')
            ax1.grid(True, alpha=0.3)
            
            # Complexity evolution
            ax2.plot(generations, mean_links, 'brown', linewidth=2, label='Mean Links', marker='o', markersize=3)
            ax2.plot(generations, max_links, 'red', linewidth=2, label='Max Links', marker='^', markersize=3)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Number of Links')
            ax2.set_title('Creature Complexity Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Fitness trend analysis
            if len(max_fitness) >= 20:
                # Calculate moving average
                window = 10
                moving_avg = []
                for i in range(window, len(max_fitness)):
                    avg = np.mean(max_fitness[i-window:i])
                    moving_avg.append(avg)
                
                ax3.plot(generations, max_fitness, 'lightblue', alpha=0.7, label='Raw Fitness')
                ax3.plot(generations[window:], moving_avg, 'darkblue', linewidth=3, 
                        label=f'{window}-Gen Moving Average')
                ax3.set_xlabel('Generation')
                ax3.set_ylabel('Max Fitness')
                ax3.set_title('Fitness Trend Analysis')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Performance improvement rate
            improvement_rates = []
            for i in range(1, len(max_fitness)):
                rate = max_fitness[i] - max_fitness[i-1]
                improvement_rates.append(rate)
            
            ax4.plot(generations[1:], improvement_rates, 'green', linewidth=2, alpha=0.7)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Fitness Improvement')
            ax4.set_title('Generation-to-Generation Improvement')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            filename = 'detailed_analysis.png'
            filepath = os.path.join(self.directories['graphs'], filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Generated detailed analysis plot")
            
        except Exception as e:
            self.logger.error(f"Error generating detailed analysis plot: {e}")
    
    def generate_summary_report(self, results: Dict):
        """Generate comprehensive text summary report"""
        try:
            report_content = f"""
CM3020 AI COURSEWORK PART B - EXPERIMENT REPORT
{'='*80}

EXPERIMENT: {self.experiment_name}
TIMESTAMP: {self.timestamp}
GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXPERIMENT CONFIGURATION
{'='*40}
Population Size: {self.config.get('population_size', 'N/A')}
Generations: {self.config.get('generations', 'N/A')}
Gene Count Range: {self.config.get('gene_count_range', 'N/A')}
Mutation Rate: {self.config.get('mutation_rate', 'N/A')}
Crossover Rate: {self.config.get('crossover_rate', 'N/A')}
Elite Size: {self.config.get('elite_size', 'N/A')}
Arena Size: {self.config.get('arena_size', 'N/A')} units
Mountain Height: {self.config.get('mountain_height', 'N/A')} units
Simulation Time: {self.config.get('simulation_time', 'N/A')} seconds

PERFORMANCE RESULTS
{'='*40}
Best Fitness Achieved: {results.get('best_fitness', 0):.4f}
Best Climbing Height: {results.get('best_metrics', {}).get('max_height', 0):.4f} units
Generations Completed: {results.get('generations_completed', 0)}
Total Execution Time: {results.get('total_time', 0)/60:.2f} minutes

EVOLUTION ANALYSIS
{'='*40}"""
            
            # Add convergence analysis
            convergence = results.get('convergence_analysis', {})
            if convergence:
                report_content += f"""
Convergence Generation: {convergence.get('convergence_generation', 'N/A')}
Final Improvement Rate: {convergence.get('final_improvement_rate', 0):.6f}
Fitness Variance (final): {convergence.get('fitness_variance', 0):.4f}"""
            
            # Add performance summary
            performance = results.get('performance_summary', {})
            if performance:
                report_content += f"""

PERFORMANCE SUMMARY
{'='*40}
Initial Fitness: {performance.get('initial_fitness', 0):.4f}
Final Fitness: {performance.get('final_fitness', 0):.4f}
Total Improvement: {performance.get('fitness_improvement', 0):.4f}
Maximum Fitness: {performance.get('max_fitness_achieved', 0):.4f}
Maximum Height: {performance.get('max_height_achieved', 0):.4f} units
Average Final Performance: {performance.get('average_final_fitness', 0):.4f}"""
            
            # Add generation statistics
            if self.generation_data:
                final_gen = self.generation_data[-1]
                report_content += f"""

FINAL GENERATION STATISTICS
{'='*40}
Population Size: {final_gen.get('population_size', 'N/A')}
Max Fitness: {final_gen.get('max_fitness', 0):.4f}
Mean Fitness: {final_gen.get('mean_fitness', 0):.4f}
Fitness Std Dev: {final_gen.get('std_fitness', 0):.4f}
Mean Gene Count: {final_gen.get('mean_links', 0):.2f}
Max Gene Count: {final_gen.get('max_links', 0)}
Best Height Reached: {final_gen.get('max_height', 0):.4f} units
Mean Height Reached: {final_gen.get('mean_height', 0):.4f} units"""
            
            # Add file listing
            files_list = self.get_generated_files_list()
            report_content += f"""

FILES GENERATED
{'='*40}
Data Files:
{chr(10).join([f"  • {f}" for f in files_list.get('data', [])])}

Visualization Files:
{chr(10).join([f"  • {f}" for f in files_list.get('graphs', [])])}

Creature Files:
{chr(10).join([f"  • {f}" for f in files_list.get('creatures', [])])}

EXPERIMENT COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
TOTAL FILES GENERATED: {sum(len(files) for files in files_list.values())}
RESULTS DIRECTORY: {self.base_dir}

{'='*80}
END OF REPORT
"""
            
            # Save report
            report_file = os.path.join(self.directories['reports'], 'experiment_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info("Generated comprehensive summary report")
            
            return report_file
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return None
    
    def get_generated_files_list(self) -> Dict[str, List[str]]:
        """Get list of all generated files organized by category"""
        files_dict = {}
        
        for category, directory in self.directories.items():
            if category == 'base':
                continue
                
            files_dict[category] = []
            if os.path.exists(directory):
                try:
                    for filename in os.listdir(directory):
                        files_dict[category].append(filename)
                except Exception as e:
                    self.logger.warning(f"Error listing files in {directory}: {e}")
        
        return files_dict


# Test function
def test_results_manager():
    """Test the results manager functionality"""
    print("Testing Results Manager...")
    
    # Create test config and manager
    test_config = {
        'experiment_name': 'test_experiment',
        'population_size': 20,
        'generations': 30,
        'mutation_rate': 0.1
    }
    
    try:
        manager = ResultsManager('test_experiment', test_config)
        print(f"Created directories in: {manager.base_dir}")
        
        # Test data saving
        test_generation_data = [
            {'generation': 0, 'max_fitness': 10, 'mean_fitness': 5, 'std_fitness': 2, 'max_height': 2},
            {'generation': 1, 'max_fitness': 15, 'mean_fitness': 8, 'std_fitness': 3, 'max_height': 3},
            {'generation': 2, 'max_fitness': 20, 'mean_fitness': 12, 'std_fitness': 4, 'max_height': 4}
        ]
        
        manager.save_generation_data(test_generation_data)
        print("Saved generation data")
        
        # Test plotting
        manager.plot_progress(test_generation_data, 2)
        manager.generate_final_plots(test_generation_data)
        print("Generated plots")
        
        # Test report generation
        test_results = {
            'best_fitness': 20,
            'best_metrics': {'max_height': 4},
            'generations_completed': 3,
            'total_time': 180,
            'convergence_analysis': {'convergence_generation': 2},
            'performance_summary': {'fitness_improvement': 10}
        }
        
        manager.save_final_results(test_results)
        report_file = manager.generate_summary_report(test_results)
        if report_file:
            print(f"Generated report: {os.path.basename(report_file)}")
        
        print("All results manager tests completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_results_manager()