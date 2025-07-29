#!/usr/bin/env python3
"""
CM3020 AI Coursework Part B - Mountain Population Management
Based on climbing_population.py with enhanced genetic operations
"""

import random
import numpy as np
import copy
import creature
import genome
from typing import List, Dict, Tuple
import logging


class MountainPopulation:
    """
    Population manager for mountain climbing evolution
    Based on climbing_population.py structure with enhancements
    """
    
    def __init__(self, population_size: int = 50, gene_count_range: Tuple[int, int] = (3, 8)):
        self.population_size = population_size
        self.gene_count_range = gene_count_range
        self.creatures = []
        self.generation = 0
        self.fitness_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def initialize_random_population(self):
        """Initialize random population using existing genome system"""
        self.logger.info(f"Initializing population of {self.population_size} creatures...")
        
        self.creatures = []
        for i in range(self.population_size):
            # Random gene count within range
            gene_count = random.randint(self.gene_count_range[0], self.gene_count_range[1])
            
            # Create creature using existing system
            cr = creature.Creature(gene_count=gene_count)
            
            # Add climbing-specific attributes
            cr.fitness = 0
            cr.climbing_metrics = {}
            cr.creature_id = i
            cr.generation_born = 0
            
            self.creatures.append(cr)
        
        self.logger.info(f"✅ Created {len(self.creatures)} creatures with gene counts {self.gene_count_range}")
    
    def tournament_selection(self, tournament_size: int = 3) -> List[creature.Creature]:
        """Tournament selection for parent selection"""
        selected = []
        
        for _ in range(self.population_size):
            # Select random creatures for tournament
            tournament_candidates = random.sample(
                self.creatures, 
                min(tournament_size, len(self.creatures))
            )
            
            # Select winner (highest fitness)
            winner = max(tournament_candidates, key=lambda x: getattr(x, 'fitness', 0))
            selected.append(winner)
        
        return selected
    
    def roulette_wheel_selection(self) -> List[creature.Creature]:
        """Roulette wheel selection based on fitness"""
        selected = []
        
        # Get fitness values and handle negative fitness
        fitness_values = [max(0, getattr(cr, 'fitness', 0)) for cr in self.creatures]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            # If all fitness is zero, use random selection
            return [random.choice(self.creatures) for _ in range(self.population_size)]
        
        # Calculate selection probabilities
        probabilities = [f / total_fitness for f in fitness_values]
        cumulative_probs = np.cumsum(probabilities)
        
        for _ in range(self.population_size):
            r = random.random()
            selected_idx = np.searchsorted(cumulative_probs, r)
            selected_idx = min(selected_idx, len(self.creatures) - 1)
            selected.append(self.creatures[selected_idx])
        
        return selected
    
    def crossover_creatures(self, parent1: creature.Creature, parent2: creature.Creature) -> creature.Creature:
        """Crossover using existing genome system"""
        try:
            # Use existing crossover function
            new_genome = genome.Genome.crossover(parent1.dna, parent2.dna)
            
            # Create new creature
            child = creature.Creature(gene_count=1)  # Dummy initialization
            child.update_dna(new_genome)
            
            # Reset attributes
            child.fitness = 0
            child.climbing_metrics = {}
            child.generation_born = self.generation + 1
            
            return child
            
        except Exception as e:
            self.logger.warning(f"Crossover failed: {e}. Cloning parent instead.")
            return self.clone_creature(parent1)
    
    def mutate_creature(self, cr: creature.Creature, mutation_rate: float = 0.1, 
                       mutation_amount: float = 0.1) -> creature.Creature:
        """Mutate creature using existing genome system"""
        try:
            # Create copy
            mutated = creature.Creature(gene_count=1)  # Dummy initialization
            mutated_genome = copy.deepcopy(cr.dna)
            
            # Apply point mutation
            mutated_genome = genome.Genome.point_mutate(
                mutated_genome, 
                mutation_rate, 
                mutation_amount
            )
            
            # Update creature with mutated genome
            mutated.update_dna(mutated_genome)
            mutated.fitness = 0
            mutated.climbing_metrics = {}
            mutated.generation_born = self.generation + 1
            
            return mutated
            
        except Exception as e:
            self.logger.warning(f"Mutation failed: {e}. Returning original creature.")
            return self.clone_creature(cr)
    
    def shrink_mutate_creature(self, cr: creature.Creature, shrink_rate: float = 0.3) -> creature.Creature:
        """Apply shrink mutation (remove genes)"""
        try:
            # Create copy
            mutated = creature.Creature(gene_count=1)
            mutated_genome = copy.deepcopy(cr.dna)
            
            # Apply shrink mutation
            mutated_genome = genome.Genome.shrink_mutate(mutated_genome, shrink_rate)
            
            # Update creature
            mutated.update_dna(mutated_genome)
            mutated.fitness = 0
            mutated.climbing_metrics = {}
            mutated.generation_born = self.generation + 1
            
            return mutated
            
        except Exception as e:
            self.logger.warning(f"Shrink mutation failed: {e}. Returning original creature.")
            return self.clone_creature(cr)
    
    def grow_mutate_creature(self, cr: creature.Creature, grow_rate: float = 0.3) -> creature.Creature:
        """Apply grow mutation (add genes)"""
        try:
            # Create copy
            mutated = creature.Creature(gene_count=1)
            mutated_genome = copy.deepcopy(cr.dna)
            
            # Apply grow mutation
            mutated_genome = genome.Genome.grow_mutate(mutated_genome, grow_rate)
            
            # Update creature
            mutated.update_dna(mutated_genome)
            mutated.fitness = 0
            mutated.climbing_metrics = {}
            mutated.generation_born = self.generation + 1
            
            return mutated
            
        except Exception as e:
            self.logger.warning(f"Grow mutation failed: {e}. Returning original creature.")
            return self.clone_creature(cr)
    
    def clone_creature(self, cr: creature.Creature) -> creature.Creature:
        """Create an exact copy of a creature"""
        try:
            clone = creature.Creature(gene_count=1)  # Dummy initialization
            clone.update_dna(copy.deepcopy(cr.dna))
            
            # Reset fitness and metrics
            clone.fitness = 0
            clone.climbing_metrics = {}
            clone.generation_born = self.generation + 1
            
            return clone
            
        except Exception as e:
            self.logger.error(f"Failed to clone creature: {e}")
            # Return a new random creature as fallback
            return creature.Creature(gene_count=random.randint(*self.gene_count_range))
    
    def get_population_statistics(self) -> Dict:
        """Get comprehensive population statistics"""
        if not self.creatures:
            return {}
        
        fitness_values = [getattr(cr, 'fitness', 0) for cr in self.creatures]
        gene_counts = [len(cr.dna) for cr in self.creatures]
        
        # Heights from climbing metrics
        heights = []
        distances = []
        stabilities = []
        
        for cr in self.creatures:
            metrics = getattr(cr, 'climbing_metrics', {})
            heights.append(metrics.get('max_height', 0))
            distances.append(metrics.get('distance_travelled', 0))
            stabilities.append(metrics.get('stability', 0))
        
        stats = {
            'generation': self.generation,
            'population_size': len(self.creatures),
            
            # Fitness statistics
            'fitness_max': np.max(fitness_values) if fitness_values else 0,
            'fitness_mean': np.mean(fitness_values) if fitness_values else 0,
            'fitness_min': np.min(fitness_values) if fitness_values else 0,
            'fitness_std': np.std(fitness_values) if fitness_values else 0,
            'fitness_median': np.median(fitness_values) if fitness_values else 0,
            
            # Gene count statistics
            'gene_count_mean': np.mean(gene_counts) if gene_counts else 0,
            'gene_count_min': np.min(gene_counts) if gene_counts else 0,
            'gene_count_max': np.max(gene_counts) if gene_counts else 0,
            'gene_count_std': np.std(gene_counts) if gene_counts else 0,
            
            # Climbing performance statistics
            'height_max': np.max(heights) if heights else 0,
            'height_mean': np.mean(heights) if heights else 0,
            'distance_max': np.max(distances) if distances else 0,
            'distance_mean': np.mean(distances) if distances else 0,
            'stability_mean': np.mean(stabilities) if stabilities else 0,
            
            # Diversity metrics
            'fitness_diversity': len(set(fitness_values)) / len(fitness_values) if fitness_values else 0,
            'gene_diversity': len(set(gene_counts)) / len(gene_counts) if gene_counts else 0
        }
        
        return stats
    
    def get_best_creature(self) -> creature.Creature:
        """Get the best creature from current population"""
        if not self.creatures:
            return None
        
        return max(self.creatures, key=lambda x: getattr(x, 'fitness', 0))
    
    def get_worst_creature(self) -> creature.Creature:
        """Get the worst creature from current population"""
        if not self.creatures:
            return None
        
        return min(self.creatures, key=lambda x: getattr(x, 'fitness', 0))
    
    def get_top_n_creatures(self, n: int) -> List[creature.Creature]:
        """Get top N creatures by fitness"""
        if not self.creatures:
            return []
        
        sorted_creatures = sorted(
            self.creatures, 
            key=lambda x: getattr(x, 'fitness', 0), 
            reverse=True
        )
        
        return sorted_creatures[:n]
    
    def save_best_creature(self, filename: str) -> bool:
        """Save the best creature's genome to CSV"""
        try:
            best = self.get_best_creature()
            if best:
                genome.Genome.to_csv(best.dna, filename)
                self.logger.info(f"💾 Saved best creature (fitness: {best.fitness:.3f}) to {filename}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to save best creature: {e}")
            return False
    
    def load_creature_from_file(self, filename: str) -> creature.Creature:
        """Load a creature from CSV file"""
        try:
            loaded_genome = genome.Genome.from_csv(filename)
            
            cr = creature.Creature(gene_count=1)  # Dummy initialization
            cr.update_dna(loaded_genome)
            cr.fitness = 0
            cr.climbing_metrics = {}
            cr.generation_born = self.generation
            
            self.logger.info(f"📁 Loaded creature from {filename}")
            return cr
            
        except Exception as e:
            self.logger.error(f"Failed to load creature from {filename}: {e}")
            return None
    
    def add_creature_to_population(self, cr: creature.Creature):
        """Add a creature to the population"""
        if len(self.creatures) < self.population_size:
            self.creatures.append(cr)
        else:
            # Replace worst creature
            worst = self.get_worst_creature()
            worst_idx = self.creatures.index(worst)
            self.creatures[worst_idx] = cr
    
    def record_generation_stats(self, additional_stats: Dict = None):
        """Record statistics for the current generation"""
        stats = self.get_population_statistics()
        
        if additional_stats:
            stats.update(additional_stats)
        
        self.fitness_history.append(stats)
    
    def analyze_population_diversity(self) -> Dict:
        """Analyze genetic and phenotypic diversity in the population"""
        if not self.creatures:
            return {}
        
        # Genetic diversity analysis
        gene_counts = [len(cr.dna) for cr in self.creatures]
        unique_gene_counts = len(set(gene_counts))
        
        # Analyze gene values for diversity
        all_genes = []
        for cr in self.creatures:
            for gene in cr.dna:
                all_genes.extend(gene)
        
        gene_variance = np.var(all_genes) if all_genes else 0
        
        # Fitness diversity
        fitness_values = [getattr(cr, 'fitness', 0) for cr in self.creatures]
        fitness_variance = np.var(fitness_values) if fitness_values else 0
        
        # Performance diversity
        heights = [cr.climbing_metrics.get('max_height', 0) for cr in self.creatures]
        height_variance = np.var(heights) if heights else 0
        
        diversity_metrics = {
            'genetic_diversity': {
                'unique_gene_counts': unique_gene_counts,
                'gene_count_range': (min(gene_counts), max(gene_counts)) if gene_counts else (0, 0),
                'gene_value_variance': gene_variance,
                'total_genes_sampled': len(all_genes)
            },
            'phenotypic_diversity': {
                'fitness_variance': fitness_variance,
                'height_variance': height_variance,
                'unique_fitness_values': len(set(fitness_values)),
                'fitness_range': (min(fitness_values), max(fitness_values)) if fitness_values else (0, 0)
            },
            'population_health': {
                'diversity_score': (unique_gene_counts / self.population_size) * 
                                 (len(set(fitness_values)) / self.population_size),
                'convergence_risk': 1 - (fitness_variance / max(np.mean(fitness_values), 1)),
                'exploration_potential': gene_variance / 10  # Normalized exploration metric
            }
        }
        
        return diversity_metrics
    
    def apply_selection_pressure(self, selection_method: str = 'tournament', 
                                selection_params: Dict = None) -> List[creature.Creature]:
        """Apply selection pressure using specified method"""
        if selection_params is None:
            selection_params = {}
        
        if selection_method == 'tournament':
            tournament_size = selection_params.get('tournament_size', 3)
            return self.tournament_selection(tournament_size)
        
        elif selection_method == 'roulette':
            return self.roulette_wheel_selection()
        
        elif selection_method == 'rank':
            return self.rank_selection(selection_params.get('selection_pressure', 1.5))
        
        elif selection_method == 'elitist':
            elite_ratio = selection_params.get('elite_ratio', 0.2)
            return self.elitist_selection(elite_ratio)
        
        else:
            self.logger.warning(f"Unknown selection method: {selection_method}. Using tournament.")
            return self.tournament_selection()
    
    def rank_selection(self, selection_pressure: float = 1.5) -> List[creature.Creature]:
        """Rank-based selection with adjustable selection pressure"""
        # Sort creatures by fitness
        sorted_creatures = sorted(
            self.creatures, 
            key=lambda x: getattr(x, 'fitness', 0), 
            reverse=True
        )
        
        # Calculate selection probabilities based on rank
        n = len(sorted_creatures)
        probabilities = []
        
        for i in range(n):
            rank = i + 1
            prob = (2 - selection_pressure) / n + 2 * rank * (selection_pressure - 1) / (n * (n - 1))
            probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Select based on probabilities
        selected = []
        for _ in range(self.population_size):
            r = random.random()
            cumulative_prob = 0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(sorted_creatures[i])
                    break
        
        return selected
    
    def elitist_selection(self, elite_ratio: float = 0.2) -> List[creature.Creature]:
        """Elitist selection keeping top performers"""
        elite_count = int(self.population_size * elite_ratio)
        
        # Get elite creatures
        elite_creatures = self.get_top_n_creatures(elite_count)
        
        # Fill remaining with tournament selection from entire population
        selected = elite_creatures[:]
        
        while len(selected) < self.population_size:
            tournament_candidates = random.sample(self.creatures, min(3, len(self.creatures)))
            winner = max(tournament_candidates, key=lambda x: getattr(x, 'fitness', 0))
            selected.append(winner)
        
        return selected
    
    def adaptive_mutation_rate(self, base_rate: float = 0.1) -> float:
        """Calculate adaptive mutation rate based on population diversity"""
        diversity_analysis = self.analyze_population_diversity()
        
        if not diversity_analysis:
            return base_rate
        
        # Get convergence risk
        convergence_risk = diversity_analysis['population_health'].get('convergence_risk', 0)
        
        # Increase mutation rate if population is converging
        if convergence_risk > 0.8:  # High convergence risk
            adaptive_rate = base_rate * 2.0
        elif convergence_risk > 0.6:  # Medium convergence risk
            adaptive_rate = base_rate * 1.5
        else:  # Low convergence risk
            adaptive_rate = base_rate
        
        # Clamp to reasonable range
        return max(0.01, min(0.5, adaptive_rate))
    
    def introduce_random_immigrants(self, immigrant_count: int = 5):
        """Introduce random creatures to maintain diversity"""
        if immigrant_count <= 0:
            return
        
        # Replace worst performers with random immigrants
        sorted_creatures = sorted(
            self.creatures, 
            key=lambda x: getattr(x, 'fitness', 0)
        )
        
        replacement_count = min(immigrant_count, len(sorted_creatures))
        
        for i in range(replacement_count):
            # Create random immigrant
            gene_count = random.randint(*self.gene_count_range)
            immigrant = creature.Creature(gene_count=gene_count)
            immigrant.fitness = 0
            immigrant.climbing_metrics = {}
            immigrant.generation_born = self.generation
            
            # Replace worst creature
            worst_idx = self.creatures.index(sorted_creatures[i])
            self.creatures[worst_idx] = immigrant
        
        self.logger.info(f"🔄 Introduced {replacement_count} random immigrants")
    
    def export_population_stats(self, filename: str):
        """Export population statistics to file"""
        try:
            stats = self.get_population_statistics()
            diversity = self.analyze_population_diversity()
            
            export_data = {
                'generation': self.generation,
                'population_stats': stats,
                'diversity_analysis': diversity,
                'fitness_history': self.fitness_history,
                'creature_details': []
            }
            
            # Add individual creature details
            for i, cr in enumerate(self.creatures):
                creature_data = {
                    'index': i,
                    'fitness': getattr(cr, 'fitness', 0),
                    'gene_count': len(cr.dna),
                    'generation_born': getattr(cr, 'generation_born', 0),
                    'climbing_metrics': getattr(cr, 'climbing_metrics', {})
                }
                export_data['creature_details'].append(creature_data)
            
            import json
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"📊 Exported population stats to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to export population stats: {e}")
    
    def get_lineage_info(self) -> Dict:
        """Get information about creature lineages"""
        generation_counts = {}
        
        for cr in self.creatures:
            gen_born = getattr(cr, 'generation_born', 0)
            generation_counts[gen_born] = generation_counts.get(gen_born, 0) + 1
        
        return {
            'current_generation': self.generation,
            'generation_distribution': generation_counts,
            'oldest_creature_generation': min(generation_counts.keys()) if generation_counts else self.generation,
            'newest_creature_generation': max(generation_counts.keys()) if generation_counts else self.generation,
            'lineage_diversity': len(generation_counts)
        }
    
    def validate_population(self) -> bool:
        """Validate population integrity"""
        try:
            if not self.creatures:
                self.logger.error("Population is empty")
                return False
            
            if len(self.creatures) != self.population_size:
                self.logger.warning(f"Population size mismatch: expected {self.population_size}, got {len(self.creatures)}")
            
            # Check each creature
            for i, cr in enumerate(self.creatures):
                if not hasattr(cr, 'dna') or not cr.dna:
                    self.logger.error(f"Creature {i} has no DNA")
                    return False
                
                if not hasattr(cr, 'fitness'):
                    cr.fitness = 0
                    self.logger.warning(f"Creature {i} missing fitness, set to 0")
                
                if not hasattr(cr, 'climbing_metrics'):
                    cr.climbing_metrics = {}
                    self.logger.warning(f"Creature {i} missing climbing metrics")
            
            self.logger.info("✅ Population validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Population validation failed: {e}")
            return False

