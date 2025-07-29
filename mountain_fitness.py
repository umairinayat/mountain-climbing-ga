#!/usr/bin/env python3
"""
CM3020 AI Coursework Part B - Mountain Fitness Evaluation
Comprehensive fitness evaluation system for mountain climbing creatures
"""

import numpy as np
import math
from typing import Dict, List, Tuple
import logging


class MountainFitnessEvaluator:
    """
    Comprehensive fitness evaluation system for mountain climbing creatures
    Designed to reward climbing behavior while preventing cheating strategies
    """
    
    def __init__(self, simulation_steps: int = 2400, arena_size: float = 20, 
                 mountain_height: float = 5):
        self.simulation_steps = simulation_steps
        self.arena_size = arena_size
        self.mountain_height = mountain_height
        
        # Fitness component weights
        self.weights = {
            'height_climbed': 20.0,      # Primary objective: climb high
            'final_height': 10.0,        # End up high
            'height_efficiency': 8.0,    # Efficient climbing
            'stability': 5.0,            # Stable movement
            'center_proximity': 3.0,     # Stay near mountain
            'upward_progress': 6.0,      # Consistent upward movement
            'movement_quality': 4.0,     # Quality of movement
            'exploration_bonus': 2.0,    # Reward exploration
            'time_bonus': 1.0           # Complete simulation
        }
        
        # Penalty weights
        self.penalties = {
            'falling_penalty': -5.0,     # Penalty for falling
            'stagnation_penalty': -2.0,  # Penalty for not moving
            'boundary_penalty': -10.0,   # Penalty for leaving arena
            'instability_penalty': -3.0  # Penalty for erratic movement
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_fitness(self, metrics: Dict, creature=None) -> float:
        """Calculate comprehensive fitness score"""
        try:
            if metrics.get('simulation_error', False):
                return 0.0
            
            # Extract basic metrics
            initial_pos = metrics.get('initial_position', (0, 0, 0))
            final_pos = metrics.get('final_position', (0, 0, 0))
            max_height = metrics.get('max_height', 0)
            
            # Calculate fitness components
            fitness_components = {}
            
            # 1. Height climbing component (primary objective)
            fitness_components['height_climbed'] = self._calculate_height_fitness(
                initial_pos, max_height, metrics
            )
            
            # 2. Final position component
            fitness_components['final_height'] = self._calculate_final_height_fitness(
                final_pos, initial_pos
            )
            
            # 3. Height efficiency component
            fitness_components['height_efficiency'] = self._calculate_height_efficiency(
                metrics
            )
            
            # 4. Stability component
            fitness_components['stability'] = self._calculate_stability_fitness(
                metrics
            )
            
            # 5. Center proximity component
            fitness_components['center_proximity'] = self._calculate_center_proximity_fitness(
                final_pos
            )
            
            # 6. Upward progress component
            fitness_components['upward_progress'] = self._calculate_upward_progress_fitness(
                metrics
            )
            
            # 7. Movement quality component
            fitness_components['movement_quality'] = self._calculate_movement_quality_fitness(
                metrics
            )
            
            # 8. Exploration bonus
            fitness_components['exploration_bonus'] = self._calculate_exploration_bonus(
                metrics
            )
            
            # 9. Time completion bonus
            fitness_components['time_bonus'] = self._calculate_time_bonus(
                metrics
            )
            
            # Calculate penalties
            penalties = {}
            
            # Falling penalty
            penalties['falling'] = self._calculate_falling_penalty(metrics)
            
            # Stagnation penalty
            penalties['stagnation'] = self._calculate_stagnation_penalty(metrics)
            
            # Boundary penalty
            penalties['boundary'] = self._calculate_boundary_penalty(final_pos)
            
            # Instability penalty
            penalties['instability'] = self._calculate_instability_penalty(metrics)
            
            # Combine fitness components
            total_fitness = 0.0
            
            for component, value in fitness_components.items():
                weighted_value = value * self.weights.get(component, 1.0)
                total_fitness += weighted_value
            
            # Apply penalties
            for penalty, value in penalties.items():
                penalty_weight = self.penalties.get(f"{penalty}_penalty", -1.0)
                total_fitness += value * penalty_weight
            
            # Ensure non-negative fitness
            total_fitness = max(0.0, total_fitness)
            
            # Store detailed breakdown for analysis
            if creature:
                creature.fitness_breakdown = {
                    'total_fitness': total_fitness,
                    'components': fitness_components,
                    'penalties': penalties,
                    'weights_used': self.weights.copy(),
                    'penalty_weights_used': self.penalties.copy()
                }
            
            return total_fitness
            
        except Exception as e:
            self.logger.error(f"Error calculating fitness: {e}")
            return 0.0
    
    def _calculate_height_fitness(self, initial_pos: Tuple, max_height: float, 
                                metrics: Dict) -> float:
        """Calculate fitness based on height climbed"""
        try:
            initial_height = initial_pos[2]
            height_climbed = max(0, max_height - initial_height)
            
            # Exponential reward for climbing higher
            if height_climbed > 0:
                # Normalize by mountain height and apply exponential scaling
                normalized_height = height_climbed / self.mountain_height
                fitness = math.pow(normalized_height, 1.5) * 10
            else:
                fitness = 0
            
            return min(fitness, 50)  # Cap at reasonable value
            
        except Exception as e:
            self.logger.warning(f"Error calculating height fitness: {e}")
            return 0
    
    def _calculate_final_height_fitness(self, final_pos: Tuple, initial_pos: Tuple) -> float:
        """Calculate fitness based on final height position"""
        try:
            final_height = final_pos[2]
            initial_height = initial_pos[2]
            
            # Reward ending up higher than start
            height_gain = final_height - initial_height
            
            if height_gain > 0:
                fitness = height_gain * 2
            else:
                fitness = 0
            
            return min(fitness, 20)  # Cap at reasonable value
            
        except Exception as e:
            self.logger.warning(f"Error calculating final height fitness: {e}")
            return 0
    
    def _calculate_height_efficiency(self, metrics: Dict) -> float:
        """Calculate efficiency of height climbing"""
        try:
            height_climbed = metrics.get('height_climbed', 0)
            total_path_length = metrics.get('total_path_length', 1)
            
            if total_path_length > 0 and height_climbed > 0:
                # Reward efficient climbing (height per unit distance)
                efficiency = height_climbed / total_path_length
                fitness = efficiency * 5
            else:
                fitness = 0
            
            return min(fitness, 15)  # Cap at reasonable value
            
        except Exception as e:
            self.logger.warning(f"Error calculating height efficiency: {e}")
            return 0
    
    def _calculate_stability_fitness(self, metrics: Dict) -> float:
        """Calculate fitness based on movement stability"""
        try:
            stability_score = metrics.get('stability', 0)
            
            # Directly use stability score from environment
            fitness = stability_score * 5
            
            return max(0, min(fitness, 10))  # Ensure positive and capped
            
        except Exception as e:
            self.logger.warning(f"Error calculating stability fitness: {e}")
            return 0
    
    def _calculate_center_proximity_fitness(self, final_pos: Tuple) -> float:
        """Calculate fitness based on staying near mountain center"""
        try:
            center_distance = math.sqrt(final_pos[0]**2 + final_pos[1]**2)
            
            # Reward staying close to center (mountain is at 0,0)
            max_reward_distance = self.arena_size / 4
            
            if center_distance <= max_reward_distance:
                fitness = 5 * (1 - center_distance / max_reward_distance)
            else:
                fitness = 0
            
            return max(0, fitness)
            
        except Exception as e:
            self.logger.warning(f"Error calculating center proximity fitness: {e}")
            return 0
    
    def _calculate_upward_progress_fitness(self, metrics: Dict) -> float:
        """Calculate fitness based on consistent upward progress"""
        try:
            upward_trend_bonus = metrics.get('upward_trend_bonus', 0)
            
            # Reward consistent upward movement
            fitness = upward_trend_bonus * 2
            
            return max(0, min(fitness, 10))
            
        except Exception as e:
            self.logger.warning(f"Error calculating upward progress fitness: {e}")
            return 0
    
    def _calculate_movement_quality_fitness(self, metrics: Dict) -> float:
        """Calculate fitness based on movement quality"""
        try:
            movement_efficiency = metrics.get('movement_efficiency', 0)
            
            # Reward efficient, purposeful movement
            fitness = movement_efficiency * 8
            
            return max(0, min(fitness, 8))
            
        except Exception as e:
            self.logger.warning(f"Error calculating movement quality fitness: {e}")
            return 0
    
    def _calculate_exploration_bonus(self, metrics: Dict) -> float:
        """Calculate bonus for exploration behavior"""
        try:
            num_positions = metrics.get('num_positions_recorded', 0)
            distance_travelled = metrics.get('distance_travelled', 0)
            
            # Small bonus for moving around and exploring
            if num_positions > 100 and distance_travelled > 5:
                fitness = 2
            elif num_positions > 50 and distance_travelled > 2:
                fitness = 1
            else:
                fitness = 0
            
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Error calculating exploration bonus: {e}")
            return 0
    
    def _calculate_time_bonus(self, metrics: Dict) -> float:
        """Calculate bonus for completing full simulation"""
        try:
            simulation_complete = metrics.get('simulation_complete', False)
            simulation_steps = metrics.get('simulation_steps', 0)
            
            if simulation_complete or simulation_steps > self.simulation_steps * 0.8:
                fitness = 1
            else:
                fitness = 0
            
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Error calculating time bonus: {e}")
            return 0
    
    def _calculate_falling_penalty(self, metrics: Dict) -> float:
        """Calculate penalty for falling below start height"""
        try:
            initial_pos = metrics.get('initial_position', (0, 0, 0))
            final_pos = metrics.get('final_position', (0, 0, 0))
            min_height = metrics.get('height_lost', 0)
            
            initial_height = initial_pos[2]
            final_height = final_pos[2]
            
            # Penalty for ending up below start or falling significantly
            if final_height < initial_height - 1:
                penalty = abs(final_height - initial_height)
            elif min_height > 2:  # Fell more than 2 units
                penalty = min_height / 2
            else:
                penalty = 0
            
            return penalty
            
        except Exception as e:
            self.logger.warning(f"Error calculating falling penalty: {e}")
            return 0
    
    def _calculate_stagnation_penalty(self, metrics: Dict) -> float:
        """Calculate penalty for lack of movement"""
        try:
            distance_travelled = metrics.get('distance_travelled', 0)
            num_positions = metrics.get('num_positions_recorded', 1)
            
            # Penalty if creature barely moved
            if num_positions > 50 and distance_travelled < 1:
                penalty = 2
            elif distance_travelled < 0.5:
                penalty = 1
            else:
                penalty = 0
            
            return penalty
            
        except Exception as e:
            self.logger.warning(f"Error calculating stagnation penalty: {e}")
            return 0
    
    def _calculate_boundary_penalty(self, final_pos: Tuple) -> float:
        """Calculate penalty for leaving arena"""
        try:
            x, y, z = final_pos
            
            # Penalty for being too far from arena center
            distance_from_center = math.sqrt(x**2 + y**2)
            arena_radius = self.arena_size / 2
            
            if distance_from_center > arena_radius:
                penalty = (distance_from_center - arena_radius) / arena_radius
            else:
                penalty = 0
            
            return penalty
            
        except Exception as e:
            self.logger.warning(f"Error calculating boundary penalty: {e}")
            return 0
    
    def _calculate_instability_penalty(self, metrics: Dict) -> float:
        """Calculate penalty for unstable movement"""
        try:
            stability_score = metrics.get('stability', 1)
            
            # Penalty for very unstable movement
            if stability_score < 0.3:
                penalty = (0.3 - stability_score) * 5
            else:
                penalty = 0
            
            return penalty
            
        except Exception as e:
            self.logger.warning(f"Error calculating instability penalty: {e}")
            return 0
    
    def get_default_metrics(self) -> Dict:
        """Return default metrics when simulation fails"""
        return {
            'fitness': 0,
            'max_height': 0,
            'final_height': 0,
            'height_climbed': 0,
            'height_lost': 0,
            'distance_travelled': 0,
            'total_path_length': 0,
            'movement_efficiency': 0,
            'stability': 0,
            'center_distance': 0,
            'center_bonus': 0,
            'upward_trend_bonus': 0,
            'final_position': (0, 0, 0),
            'initial_position': (0, 0, 0),
            'num_positions_recorded': 0,
            'simulation_steps': 0,
            'simulation_complete': False,
            'simulation_error': True
        }
    
    def analyze_fitness_breakdown(self, creature) -> Dict:
        """Analyze detailed fitness breakdown for a creature"""
        if not hasattr(creature, 'fitness_breakdown'):
            return {'error': 'No fitness breakdown available'}
        
        breakdown = creature.fitness_breakdown
        
        analysis = {
            'total_fitness': breakdown['total_fitness'],
            'component_contributions': {},
            'penalty_contributions': {},
            'dominant_components': [],
            'improvement_suggestions': []
        }
        
        # Analyze component contributions
        for component, value in breakdown['components'].items():
            weight = breakdown['weights_used'].get(component, 1.0)
            contribution = value * weight
            analysis['component_contributions'][component] = {
                'raw_value': value,
                'weight': weight,
                'contribution': contribution,
                'percentage': (contribution / max(breakdown['total_fitness'], 1)) * 100
            }
        
        # Analyze penalty contributions
        for penalty, value in breakdown['penalties'].items():
            weight = breakdown['penalty_weights_used'].get(f"{penalty}_penalty", -1.0)
            contribution = value * weight
            analysis['penalty_contributions'][penalty] = {
                'raw_value': value,
                'weight': weight,
                'contribution': contribution
            }
        
        # Find dominant components (top 3)
        sorted_components = sorted(
            analysis['component_contributions'].items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )
        analysis['dominant_components'] = [comp[0] for comp in sorted_components[:3]]
        
        # Generate improvement suggestions
        analysis['improvement_suggestions'] = self._generate_improvement_suggestions(
            breakdown, analysis
        )
        
        return analysis
    
    def _generate_improvement_suggestions(self, breakdown: Dict, analysis: Dict) -> List[str]:
        """Generate suggestions for improving fitness"""
        suggestions = []
        
        # Check height climbing
        height_contrib = analysis['component_contributions'].get('height_climbed', {})
        if height_contrib.get('contribution', 0) < 10:
            suggestions.append("Focus on climbing higher - this is the primary objective")
        
        # Check stability
        stability_contrib = analysis['component_contributions'].get('stability', {})
        if stability_contrib.get('contribution', 0) < 2:
            suggestions.append("Improve movement stability - reduce erratic motion")
        
        # Check penalties
        for penalty, data in analysis['penalty_contributions'].items():
            if data['contribution'] < -1:
                if penalty == 'falling':
                    suggestions.append("Reduce falling behavior - creature is losing too much height")
                elif penalty == 'stagnation':
                    suggestions.append("Increase movement - creature is too static")
                elif penalty == 'boundary':
                    suggestions.append("Stay within arena bounds - creature is moving too far from center")
                elif penalty == 'instability':
                    suggestions.append("Reduce unstable movement patterns")
        
        # Check center proximity
        center_contrib = analysis['component_contributions'].get('center_proximity', {})
        if center_contrib.get('contribution', 0) < 1:
            suggestions.append("Stay closer to mountain center for better climbing opportunities")
        
        # Check movement quality
        movement_contrib = analysis['component_contributions'].get('movement_quality', {})
        if movement_contrib.get('contribution', 0) < 1:
            suggestions.append("Improve movement efficiency - move more purposefully toward the mountain")
        
        return suggestions
    
    def compare_fitness_profiles(self, creature1, creature2) -> Dict:
        """Compare fitness profiles of two creatures"""
        if not hasattr(creature1, 'fitness_breakdown') or not hasattr(creature2, 'fitness_breakdown'):
            return {'error': 'Both creatures need fitness breakdown data'}
        
        breakdown1 = creature1.fitness_breakdown
        breakdown2 = creature2.fitness_breakdown
        
        comparison = {
            'fitness_difference': breakdown2['total_fitness'] - breakdown1['total_fitness'],
            'component_differences': {},
            'penalty_differences': {},
            'performance_analysis': {}
        }
        
        # Compare components
        for component in breakdown1['components']:
            val1 = breakdown1['components'][component] * breakdown1['weights_used'].get(component, 1.0)
            val2 = breakdown2['components'].get(component, 0) * breakdown2['weights_used'].get(component, 1.0)
            
            comparison['component_differences'][component] = {
                'creature1': val1,
                'creature2': val2,
                'difference': val2 - val1,
                'improvement': val2 > val1
            }
        
        # Compare penalties
        for penalty in breakdown1['penalties']:
            val1 = breakdown1['penalties'][penalty] * breakdown1['penalty_weights_used'].get(f"{penalty}_penalty", -1.0)
            val2 = breakdown2['penalties'].get(penalty, 0) * breakdown2['penalty_weights_used'].get(f"{penalty}_penalty", -1.0)
            
            comparison['penalty_differences'][penalty] = {
                'creature1': val1,
                'creature2': val2,
                'difference': val2 - val1,
                'improvement': val2 > val1  # Less negative penalty is better
            }
        
        # Performance analysis
        if comparison['fitness_difference'] > 0:
            comparison['performance_analysis']['winner'] = 'creature2'
            comparison['performance_analysis']['improvement_areas'] = [
                comp for comp, data in comparison['component_differences'].items()
                if data['improvement']
            ]
        else:
            comparison['performance_analysis']['winner'] = 'creature1'
            comparison['performance_analysis']['improvement_areas'] = [
                comp for comp, data in comparison['component_differences'].items()
                if not data['improvement']
            ]
        
        return comparison
    
    def get_fitness_statistics(self, creatures: list) -> Dict:
        """Get statistical analysis of fitness across a population"""
        if not creatures:
            return {}
        
        fitness_values = [getattr(cr, 'fitness', 0) for cr in creatures]
        
        # Basic statistics
        stats = {
            'population_size': len(creatures),
            'fitness_mean': np.mean(fitness_values),
            'fitness_std': np.std(fitness_values),
            'fitness_min': np.min(fitness_values),
            'fitness_max': np.max(fitness_values),
            'fitness_median': np.median(fitness_values),
            'fitness_range': np.max(fitness_values) - np.min(fitness_values)
        }
        
        # Component analysis (if breakdown available)
        component_stats = {}
        creatures_with_breakdown = [cr for cr in creatures if hasattr(cr, 'fitness_breakdown')]
        
        if creatures_with_breakdown:
            for component in creatures_with_breakdown[0].fitness_breakdown['components']:
                values = []
                for cr in creatures_with_breakdown:
                    if component in cr.fitness_breakdown['components']:
                        weight = cr.fitness_breakdown['weights_used'].get(component, 1.0)
                        contribution = cr.fitness_breakdown['components'][component] * weight
                        values.append(contribution)
                
                if values:
                    component_stats[component] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        stats['component_statistics'] = component_stats
        
        # Performance distribution
        stats['performance_distribution'] = {
            'excellent': len([f for f in fitness_values if f > stats['fitness_mean'] + stats['fitness_std']]),
            'good': len([f for f in fitness_values if stats['fitness_mean'] < f <= stats['fitness_mean'] + stats['fitness_std']]),
            'average': len([f for f in fitness_values if stats['fitness_mean'] - stats['fitness_std'] < f <= stats['fitness_mean']]),
            'poor': len([f for f in fitness_values if f <= stats['fitness_mean'] - stats['fitness_std']])
        }
        
        return stats
    
    def adaptive_fitness_weights(self, generation: int, population_stats: Dict) -> Dict:
        """Adapt fitness weights based on population performance and generation"""
        adapted_weights = self.weights.copy()
        
        if not population_stats:
            return adapted_weights
        
        # Early generations: emphasize exploration and basic climbing
        if generation < 20:
            adapted_weights['height_climbed'] *= 1.2
            adapted_weights['exploration_bonus'] *= 1.5
            adapted_weights['stability'] *= 0.8
        
        # Mid generations: balance all components
        elif generation < 50:
            # Standard weights
            pass
        
        # Late generations: emphasize refinement and efficiency
        else:
            adapted_weights['height_efficiency'] *= 1.3
            adapted_weights['stability'] *= 1.2
            adapted_weights['movement_quality'] *= 1.1
            adapted_weights['exploration_bonus'] *= 0.7
        
        # Adapt based on population performance
        max_fitness = population_stats.get('fitness_max', 0)
        mean_fitness = population_stats.get('fitness_mean', 0)
        
        # If population is struggling with basic climbing
        if max_fitness < 10:
            adapted_weights['height_climbed'] *= 1.5
            adapted_weights['final_height'] *= 1.3
        
        # If population has good climbers, emphasize quality
        elif max_fitness > 30:
            adapted_weights['height_efficiency'] *= 1.2
            adapted_weights['stability'] *= 1.1
            adapted_weights['movement_quality'] *= 1.1
        
        return adapted_weights
    
    def set_custom_weights(self, custom_weights: Dict):
        """Set custom fitness weights for experimentation"""
        for component, weight in custom_weights.items():
            if component in self.weights:
                self.weights[component] = weight
                self.logger.info(f"Updated weight for {component}: {weight}")
            else:
                self.logger.warning(f"Unknown fitness component: {component}")
    
    def reset_default_weights(self):
        """Reset to default fitness weights"""
        self.weights = {
            'height_climbed': 20.0,
            'final_height': 10.0,
            'height_efficiency': 8.0,
            'stability': 5.0,
            'center_proximity': 3.0,
            'upward_progress': 6.0,
            'movement_quality': 4.0,
            'exploration_bonus': 2.0,
            'time_bonus': 1.0
        }
        
        self.penalties = {
            'falling_penalty': -5.0,
            'stagnation_penalty': -2.0,
            'boundary_penalty': -10.0,
            'instability_penalty': -3.0
        }
        
        self.logger.info("Reset to default fitness weights")
    
    def validate_metrics(self, metrics: Dict) -> bool:
        """Validate that metrics contain required fields"""
        required_fields = [
            'initial_position', 'final_position', 'max_height'
        ]
        
        for field in required_fields:
            if field not in metrics:
                self.logger.warning(f"Missing required metric: {field}")
                return False
        
        # Validate position formats
        for pos_field in ['initial_position', 'final_position']:
            pos = metrics[pos_field]
            if not isinstance(pos, (list, tuple)) or len(pos) != 3:
                self.logger.warning(f"Invalid position format for {pos_field}")
                return False
        
        return True
    
    def get_fitness_config(self) -> Dict:
        """Get current fitness configuration"""
        return {
            'weights': self.weights.copy(),
            'penalties': self.penalties.copy(),
            'simulation_steps': self.simulation_steps,
            'arena_size': self.arena_size,
            'mountain_height': self.mountain_height
        }


