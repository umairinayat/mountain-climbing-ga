#!/usr/bin/env python3
"""
CM3020 AI Coursework Part B - Mountain Environment
Enhanced mountain climbing environment with robust error handling and features
FIXED VERSION - Resolves creature validation and spawning issues
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import math
import os
from typing import List, Tuple, Optional, Dict
import creature
import logging


class MountainEnvironment:
    """
    Enhanced mountain climbing environment with robust error handling
    Comprehensive physics simulation for creature evaluation
    FIXED VERSION - No more validation or spawning errors
    """
    
    def __init__(self, gui=False, arena_size=20, mountain_height=5, time_limit=30):
        self.gui = gui
        self.arena_size = arena_size
        self.mountain_height = mountain_height
        self.time_limit = time_limit
        self.physics_client = None
        self.mountain_id = None
        self.floor_id = None
        self.wall_ids = []
        self.creatures = []
        self.creature_counter = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Simulation parameters
        self.simulation_hz = 240
        self.physics_timestep = 1.0 / self.simulation_hz
        
        # Spawn parameters
        self.spawn_height_offset = 5.0
        self.spawn_radius_min = 7.0
        self.spawn_radius_max = 10.0
        
    def initialize_physics(self):
        """Initialize PyBullet physics engine with enhanced settings"""
        try:
            # Connect to physics server
            if self.gui:
                self.physics_client = p.connect(p.GUI)
                # Enhanced camera setup for better viewing
                p.resetDebugVisualizerCamera(
                    cameraDistance=25,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=[0, 0, 2]
                )
                # Disable some GUI elements for cleaner view
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
            else:
                self.physics_client = p.connect(p.DIRECT)
            
            if self.physics_client < 0:
                self.logger.error("Failed to connect to PyBullet physics server")
                return False
            
            # Set search path for built-in assets
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Enhanced physics parameters
            p.setPhysicsEngineParameter(
                enableFileCaching=0,
                numSolverIterations=10,
                numSubSteps=1,
                contactBreakingThreshold=0.001,
                enableConeFriction=1
            )
            
            # Set gravity and time step
            p.setGravity(0, 0, -10)
            p.setTimeStep(self.physics_timestep)
            p.setRealTimeSimulation(0)  # Step-based simulation for consistency
            
            self.logger.info("Physics engine initialized successfully")
            self.logger.debug(f"Physics client ID: {self.physics_client}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize physics: {e}")
            return False
    
    def create_arena(self):
        """Create arena with walls and floor"""
        try:
            wall_thickness = 0.5
            wall_height = 3
            
            # Create textured floor
            floor_collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[self.arena_size/2, self.arena_size/2, wall_thickness]
            )
            floor_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[self.arena_size/2, self.arena_size/2, wall_thickness], 
                rgbaColor=[0.8, 0.8, 0.6, 1],
                specularColor=[0.1, 0.1, 0.1]
            )
            self.floor_id = p.createMultiBody(
                baseMass=0, 
                baseCollisionShapeIndex=floor_collision, 
                baseVisualShapeIndex=floor_visual, 
                basePosition=[0, 0, -wall_thickness]
            )
            
            if self.floor_id < 0:
                self.logger.error("Failed to create arena floor")
                return False
            
            # Create walls with improved design
            self._create_walls(wall_thickness, wall_height)
            
            self.logger.info("Arena created successfully")
            self.logger.debug(f"Floor ID: {self.floor_id}, Walls: {len(self.wall_ids)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create arena: {e}")
            return False
    
    def _create_walls(self, thickness: float, height: float):
        """Create arena walls with enhanced design"""
        try:
            wall_positions = [
                # North wall
                ([0, self.arena_size/2, height/2], [self.arena_size/2, thickness/2, height/2]),
                # South wall
                ([0, -self.arena_size/2, height/2], [self.arena_size/2, thickness/2, height/2]),
                # East wall
                ([self.arena_size/2, 0, height/2], [thickness/2, self.arena_size/2, height/2]),
                # West wall
                ([-self.arena_size/2, 0, height/2], [thickness/2, self.arena_size/2, height/2])
            ]
            
            self.wall_ids = []
            
            for i, (pos, extents) in enumerate(wall_positions):
                collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=extents)
                visual = p.createVisualShape(
                    shapeType=p.GEOM_BOX, 
                    halfExtents=extents, 
                    rgbaColor=[0.7, 0.7, 0.7, 1],
                    specularColor=[0.2, 0.2, 0.2]
                )
                wall_id = p.createMultiBody(
                    baseMass=0, 
                    baseCollisionShapeIndex=collision, 
                    baseVisualShapeIndex=visual, 
                    basePosition=pos
                )
                
                if wall_id >= 0:
                    self.wall_ids.append(wall_id)
                else:
                    self.logger.warning(f"Failed to create wall {i}")
                    
        except Exception as e:
            self.logger.error(f"Failed to create walls: {e}")
    
    def load_mountain(self, mountain_urdf: str = "shapes/gaussian_pyramid.urdf"):
        """Load mountain terrain with fallback options"""
        try:
            # Try to load specified URDF
            if os.path.exists(mountain_urdf):
                mountain_position = (0, 0, -1)
                mountain_orientation = p.getQuaternionFromEuler((0, 0, 0))
                
                self.mountain_id = p.loadURDF(
                    mountain_urdf, 
                    mountain_position, 
                    mountain_orientation, 
                    useFixedBase=1,
                    flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                )
                
                if self.mountain_id >= 0:
                    self.logger.info(f"Mountain loaded from {mountain_urdf}")
                    return True
                else:
                    self.logger.warning(f"Failed to load URDF {mountain_urdf}, creating simple mountain")
                    return self._create_simple_mountain()
            else:
                self.logger.warning(f"Mountain URDF {mountain_urdf} not found. Creating simple mountain.")
                return self._create_simple_mountain()
                
        except Exception as e:
            self.logger.error(f"Failed to load mountain: {e}")
            return self._create_simple_mountain()
    
    def _create_simple_mountain(self):
        """Create a simple mountain if URDF loading fails"""
        try:
            # Create a multi-level pyramid using multiple geometric shapes
            mountain_parts = []
            
            # Base level - large foundation
            base_size = 8
            base_height = 1
            base_collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[base_size/2, base_size/2, base_height/2]
            )
            base_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[base_size/2, base_size/2, base_height/2], 
                rgbaColor=[0.6, 0.4, 0.2, 1],
                specularColor=[0.1, 0.1, 0.1]
            )
            base_id = p.createMultiBody(
                baseMass=0, 
                baseCollisionShapeIndex=base_collision, 
                baseVisualShapeIndex=base_visual, 
                basePosition=[0, 0, base_height/2]
            )
            
            if base_id >= 0:
                mountain_parts.append(base_id)
            
            # Middle level - medium platform
            mid_size = 5
            mid_height = 1.5
            mid_collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[mid_size/2, mid_size/2, mid_height/2]
            )
            mid_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[mid_size/2, mid_size/2, mid_height/2], 
                rgbaColor=[0.7, 0.5, 0.3, 1],
                specularColor=[0.1, 0.1, 0.1]
            )
            mid_id = p.createMultiBody(
                baseMass=0, 
                baseCollisionShapeIndex=mid_collision, 
                baseVisualShapeIndex=mid_visual, 
                basePosition=[0, 0, base_height + mid_height/2]
            )
            
            if mid_id >= 0:
                mountain_parts.append(mid_id)
            
            # Top level - peak
            top_size = 2
            top_height = 2
            top_collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[top_size/2, top_size/2, top_height/2]
            )
            top_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX, 
                halfExtents=[top_size/2, top_size/2, top_height/2], 
                rgbaColor=[0.8, 0.6, 0.4, 1],
                specularColor=[0.2, 0.2, 0.2]
            )
            top_id = p.createMultiBody(
                baseMass=0, 
                baseCollisionShapeIndex=top_collision, 
                baseVisualShapeIndex=top_visual, 
                basePosition=[0, 0, base_height + mid_height + top_height/2]
            )
            
            if top_id >= 0:
                mountain_parts.append(top_id)
            
            # Add some climbing ramps for more interesting terrain
            self._create_climbing_ramps(base_height, mid_height)
            
            # Store reference to main mountain part
            self.mountain_id = mountain_parts[0] if mountain_parts else -1
            
            if self.mountain_id >= 0:
                self.logger.info("Created simple stepped mountain with climbing features")
                return True
            else:
                self.logger.error("Failed to create any mountain parts")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to create simple mountain: {e}")
            return False
    
    def _create_climbing_ramps(self, base_height: float, mid_height: float):
        """Create climbing ramps to make mountain more interesting"""
        try:
            # Create small ramps/steps for climbing
            ramp_positions = [
                ([3, 3, base_height + 0.3], [1, 0.5, 0.3]),  # NE ramp
                ([-3, 3, base_height + 0.3], [1, 0.5, 0.3]), # NW ramp
                ([3, -3, base_height + 0.3], [1, 0.5, 0.3]), # SE ramp
                ([-3, -3, base_height + 0.3], [1, 0.5, 0.3]) # SW ramp
            ]
            
            for pos, extents in ramp_positions:
                collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=extents)
                visual = p.createVisualShape(
                    shapeType=p.GEOM_BOX, 
                    halfExtents=extents, 
                    rgbaColor=[0.65, 0.45, 0.25, 1]
                )
                p.createMultiBody(
                    baseMass=0, 
                    baseCollisionShapeIndex=collision, 
                    baseVisualShapeIndex=visual, 
                    basePosition=pos
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to create climbing ramps: {e}")
    
    def validate_creature_before_spawn(self, creature_obj) -> bool:
        """FIXED: Validate creature object before attempting to spawn"""
        try:
            # Check basic structure
            if not hasattr(creature_obj, 'dna'):
                self.logger.error("Creature missing DNA attribute")
                return False
            
            # FIXED: Enhanced DNA validation to handle NumPy arrays
            creature_dna = creature_obj.dna
            
            # Check if DNA exists
            if creature_dna is None:
                self.logger.error("Creature has None DNA")
                return False
            
            # FIXED: Safe DNA content checking to avoid NumPy array boolean ambiguity
            try:
                # Handle different DNA types safely
                if hasattr(creature_dna, '__len__'):
                    dna_length = len(creature_dna)
                    has_content = dna_length > 0
                elif hasattr(creature_dna, 'shape'):  # NumPy arrays with shape
                    if len(creature_dna.shape) == 0:  # Scalar
                        has_content = True
                        dna_length = 1
                    else:
                        dna_length = creature_dna.shape[0]
                        has_content = dna_length > 0
                else:
                    # If it doesn't have __len__ or shape, assume it exists if not None
                    has_content = True
                    dna_length = 1
                    
                if not has_content:
                    self.logger.error("Creature has empty DNA")
                    return False
                    
                self.logger.debug(f"Creature DNA validation passed: {dna_length} genes")
                
            except Exception as dna_check_error:
                self.logger.error(f"Error checking DNA content: {dna_check_error}")
                return False
            
            # Check required methods
            required_methods = ['to_xml']  # Only check essential method
            for method in required_methods:
                if not hasattr(creature_obj, method):
                    self.logger.error(f"Creature missing required method: {method}")
                    return False
            
            # Try to generate XML to validate
            try:
                xml_content = creature_obj.to_xml()
                
                if xml_content is None:
                    self.logger.error("Creature generates None XML")
                    return False
                    
                if not isinstance(xml_content, str):
                    self.logger.error(f"Creature XML is not string, got: {type(xml_content)}")
                    return False
                    
                if len(xml_content.strip()) == 0:
                    self.logger.error("Creature generates empty XML")
                    return False
                
                # Basic XML validation
                if '<robot' not in xml_content or '</robot>' not in xml_content:
                    self.logger.error("Creature XML missing robot tags")
                    return False
                
                # Check for reasonable content length
                if len(xml_content) < 50:  # Reduced threshold
                    self.logger.warning(f"Creature XML seems short ({len(xml_content)} chars)")
                    
                self.logger.debug(f"Creature XML validation passed: {len(xml_content)} characters")
                    
            except Exception as xml_error:
                self.logger.error(f"Creature XML generation failed: {xml_error}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Creature validation failed: {e}")
            return False
    
    def spawn_creature(self, creature_obj: creature.Creature, position: Tuple[float, float, float] = None) -> Optional[int]:
        """FIXED: Enhanced creature spawning with comprehensive error handling"""
        try:
            # FIXED: Enhanced creature validation with better error handling
            try:
                if not self.validate_creature_before_spawn(creature_obj):
                    self.logger.warning("Creature failed validation, cannot spawn")
                    return None
            except Exception as validation_error:
                self.logger.error(f"Creature validation error: {validation_error}")
                return None
            
            # Determine spawn position with validation
            if position is None:
                try:
                    # Generate safe spawn position around mountain base
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.uniform(self.spawn_radius_min, self.spawn_radius_max)
                    position = (
                        radius * math.cos(angle), 
                        radius * math.sin(angle), 
                        self.spawn_height_offset
                    )
                except Exception as pos_error:
                    self.logger.error(f"Error generating spawn position: {pos_error}")
                    # Use default safe position
                    position = (8.0, 0.0, 5.0)
            
            # Validate position
            if not self._validate_position(position):
                self.logger.error(f"Invalid spawn position: {position}")
                position = (8.0, 0.0, 5.0)  # Safe default
            
            # Create unique temporary file with enhanced naming
            self.creature_counter += 1
            temp_urdf = f"temp_creature_{self.creature_counter}_{os.getpid()}_{int(time.time() * 1000)}.urdf"
            
            try:
                # Generate creature XML with enhanced error handling
                try:
                    xml_content = creature_obj.to_xml()
                except Exception as xml_gen_error:
                    self.logger.error(f"Failed to generate creature XML: {xml_gen_error}")
                    return None
                
                # Validate XML content
                if not xml_content or not isinstance(xml_content, str) or len(xml_content.strip()) == 0:
                    self.logger.error("Generated invalid or empty creature XML")
                    return None
                
                # Enhanced XML validation
                if '<robot' not in xml_content or '</robot>' not in xml_content:
                    self.logger.error("Invalid creature XML: missing robot tags")
                    return None
                
                # Write to file with comprehensive error handling
                try:
                    with open(temp_urdf, 'w', encoding='utf-8') as f:
                        f.write(xml_content)
                        f.flush()
                        os.fsync(f.fileno())  # Ensure file is written to disk
                        
                except (IOError, OSError) as file_error:
                    self.logger.error(f"Failed to write URDF file {temp_urdf}: {file_error}")
                    return None
                
                # Verify file was created and has content
                if not os.path.exists(temp_urdf):
                    self.logger.error(f"URDF file {temp_urdf} was not created")
                    return None
                
                try:
                    file_size = os.path.getsize(temp_urdf)
                    if file_size == 0:
                        self.logger.error(f"URDF file {temp_urdf} is empty")
                        self._cleanup_temp_file(temp_urdf)
                        return None
                    
                    self.logger.debug(f"Created URDF file {temp_urdf} ({file_size} bytes)")
                except OSError as size_error:
                    self.logger.error(f"Cannot check URDF file size: {size_error}")
                    self._cleanup_temp_file(temp_urdf)
                    return None
                
                # Load creature with multiple attempts and enhanced error handling
                creature_id = None
                max_attempts = 3
                
                for attempt in range(max_attempts):
                    try:
                        # Try loading the URDF with different flags
                        load_flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                        if attempt > 0:
                            load_flags |= p.URDF_USE_INERTIA_FROM_FILE
                        
                        creature_id = p.loadURDF(
                            temp_urdf, 
                            position,
                            useFixedBase=0,
                            flags=load_flags
                        )
                        
                        if creature_id >= 0:
                            self.logger.debug(f"Successfully loaded URDF on attempt {attempt + 1}")
                            break
                        else:
                            self.logger.warning(f"loadURDF returned invalid ID {creature_id} on attempt {attempt + 1}")
                            
                    except Exception as load_error:
                        self.logger.warning(f"loadURDF attempt {attempt + 1} failed: {load_error}")
                        if attempt < max_attempts - 1:
                            time.sleep(0.1)  # Brief pause before retry
                        continue
                
                # Check if loading was successful
                if creature_id is None or creature_id < 0:
                    self.logger.error(f"Failed to load creature from {temp_urdf} after {max_attempts} attempts")
                    self._cleanup_temp_file(temp_urdf)
                    return None
                
                # Verify creature was loaded successfully
                try:
                    pos, orn = p.getBasePositionAndOrientation(creature_id)
                    
                    # Validate position is reasonable
                    if not self._validate_position(pos):
                        self.logger.error(f"Creature {creature_id} has invalid position: {pos}")
                        try:
                            p.removeBody(creature_id)
                        except:
                            pass
                        self._cleanup_temp_file(temp_urdf)
                        return None
                    
                    # Set initial conditions for better simulation
                    try:
                        p.resetBasePositionAndOrientation(creature_id, position, [0, 0, 0, 1])
                        p.resetBaseVelocity(creature_id, [0, 0, 0], [0, 0, 0])
                    except Exception as reset_error:
                        self.logger.warning(f"Could not reset creature initial state: {reset_error}")
                        # Continue anyway, this is not critical
                    
                    # Store creature in tracking list
                    self.creatures.append(creature_id)
                    
                    # Clean up temp file
                    self._cleanup_temp_file(temp_urdf)
                    
                    self.logger.debug(f"Successfully spawned creature {creature_id} at position {position}")
                    return creature_id
                    
                except Exception as verify_error:
                    self.logger.error(f"Failed to verify creature {creature_id}: {verify_error}")
                    try:
                        p.removeBody(creature_id)
                    except:
                        pass
                    self._cleanup_temp_file(temp_urdf)
                    return None
                
            except Exception as processing_error:
                self.logger.error(f"Error processing creature for spawn: {processing_error}")
                self._cleanup_temp_file(temp_urdf)
                return None
                
        except Exception as e:
            self.logger.error(f"Unexpected error spawning creature: {e}")
            if 'temp_urdf' in locals():
                self._cleanup_temp_file(temp_urdf)
            return None
    
    def _validate_position(self, pos: Tuple[float, float, float]) -> bool:
        """Validate that a position is reasonable"""
        try:
            return (isinstance(pos, (list, tuple)) and 
                   len(pos) == 3 and
                   all(isinstance(x, (int, float)) and 
                       not math.isnan(x) and 
                       not math.isinf(x) and
                       abs(x) < 1000 for x in pos))
        except:
            return False
    
    def _cleanup_temp_file(self, filename: str):
        """Safely cleanup temporary file"""
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            self.logger.debug(f"Could not remove temp file {filename}: {e}")
    
    def simulate_creature(self, creature_obj: creature.Creature, creature_id: int, 
                         simulation_steps: int = None) -> Dict:
        """Enhanced creature simulation with comprehensive metrics"""
        if simulation_steps is None:
            simulation_steps = int(self.time_limit * self.simulation_hz)
        
        # Verify creature exists
        if not self._creature_exists(creature_id):
            self.logger.error(f"Creature {creature_id} does not exist")
            return self._get_default_metrics()
        
        try:
            # Get initial position and state
            initial_pos, initial_orn = p.getBasePositionAndOrientation(creature_id)
            creature_obj.update_position(initial_pos)
            
        except Exception as e:
            self.logger.error(f"Error getting initial position for creature {creature_id}: {e}")
            return self._get_default_metrics()
        
        # Initialize tracking variables
        max_height = initial_pos[2]
        min_height = initial_pos[2]
        positions = [initial_pos]
        velocities = []
        step_count = 0
        
        # Get motors with enhanced error handling
        try:
            motors = creature_obj.get_motors()
            if not motors:
                motors = []
                self.logger.debug(f"Creature {creature_id} has no motors")
        except Exception as e:
            self.logger.warning(f"Error getting motors for creature {creature_id}: {e}")
            motors = []
        
        # Enhanced simulation loop
        for step in range(simulation_steps):
            try:
                # Step physics simulation
                p.stepSimulation()
                step_count += 1
                
                # Apply motor control with adaptive frequency
                motor_control_frequency = 24  # Every 24 steps
                if step % motor_control_frequency == 0 and motors:
                    self._apply_motor_control(creature_id, motors, step)
                
                # Record metrics at regular intervals for efficiency
                if step % 10 == 0:
                    if not self._creature_exists(creature_id):
                        self.logger.warning(f"Creature {creature_id} disappeared at step {step}")
                        break
                    
                    try:
                        pos, orn = p.getBasePositionAndOrientation(creature_id)
                        vel, ang_vel = p.getBaseVelocity(creature_id)
                        
                        if not self._validate_position(pos):
                            self.logger.warning(f"Creature {creature_id} has invalid position at step {step}")
                            break
                        
                        positions.append(pos)
                        velocities.append(vel)
                        max_height = max(max_height, pos[2])
                        min_height = min(min_height, pos[2])
                        
                        creature_obj.update_position(pos)
                        
                        # Enhanced early termination conditions
                        if self._should_terminate_simulation(pos, vel, step):
                            break
                        
                    except Exception as pos_error:
                        self.logger.warning(f"Error getting position at step {step}: {pos_error}")
                        break
                
            except Exception as e:
                self.logger.error(f"Error in simulation step {step}: {e}")
                break
        
        # Calculate comprehensive metrics
        try:
            metrics = self._calculate_comprehensive_metrics(
                positions, velocities, max_height, min_height, initial_pos, creature_obj
            )
            metrics['simulation_steps'] = step_count
            metrics['simulation_complete'] = step_count == simulation_steps
            metrics['simulation_efficiency'] = step_count / simulation_steps if simulation_steps > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            metrics = self._get_default_metrics()
        
        return metrics
    
    def _should_terminate_simulation(self, pos: Tuple[float, float, float], 
                                   vel: Tuple[float, float, float], step: int) -> bool:
        """Determine if simulation should terminate early"""
        try:
            # Fell through floor
            if pos[2] < -2:
                return True
            
            # Left arena (with buffer)
            arena_limit = self.arena_size * 1.5
            if abs(pos[0]) > arena_limit or abs(pos[1]) > arena_limit:
                return True
            
            # Completely stuck (but allow initial settling)
            if step > 500:
                velocity_magnitude = math.sqrt(sum(v*v for v in vel))
                if velocity_magnitude < 0.01:
                    return True
            
            # Extreme velocity (unstable simulation)
            velocity_magnitude = math.sqrt(sum(v*v for v in vel))
            if velocity_magnitude > 50:
                return True
            
            return False
            
        except:
            return True
    
    def _creature_exists(self, creature_id: int) -> bool:
        """Enhanced check if creature still exists in simulation"""
        try:
            if creature_id is None or creature_id < 0:
                return False
            
            # Try to get position - this will fail if creature doesn't exist
            pos, orn = p.getBasePositionAndOrientation(creature_id)
            
            # Validate position is reasonable
            return self._validate_position(pos)
            
        except:
            return False
    
    def _apply_motor_control(self, creature_id: int, motors: List, step: int):
        """Enhanced motor control with adaptive parameters"""
        try:
            num_joints = p.getNumJoints(creature_id)
            motor_count = min(len(motors), num_joints)
            
            if motor_count == 0:
                return
            
            # Time-based motor control for more natural movement
            time_factor = step * self.physics_timestep
            
            for jid in range(motor_count):
                try:
                    # Check if motor has get_output method with time parameter
                    if hasattr(motors[jid], 'get_output'):
                        try:
                            motor_output = motors[jid].get_output(time_factor)
                        except TypeError:
                            # Fallback for motors that don't accept time parameter
                            motor_output = motors[jid].get_output()
                    else:
                        # Fallback for motors without get_output method
                        motor_output = 0.0
                    
                    # Enhanced motor output clamping
                    motor_output = max(-15, min(15, motor_output))
                    
                    # Apply motor control with enhanced parameters
                    p.setJointMotorControl2(
                        creature_id, 
                        jid,  
                        controlMode=p.VELOCITY_CONTROL, 
                        targetVelocity=motor_output,
                        force=8,  # Increased force for better control
                        maxVelocity=20
                    )
                    
                except Exception as motor_error:
                    # Skip problematic motors but log occasionally
                    if step % 1000 == 0:
                        self.logger.debug(f"Motor {jid} error: {motor_error}")
                    continue
                    
        except Exception as e:
            # If motor control fails completely, continue without it
            if step % 1000 == 0:  # Log occasionally
                self.logger.debug(f"Motor control failed: {e}")
    
    def _calculate_comprehensive_metrics(self, positions: List, velocities: List, 
                                       max_height: float, min_height: float,
                                       initial_pos: Tuple, creature_obj: creature.Creature) -> Dict:
        """Calculate comprehensive fitness metrics with enhanced analysis"""
        if not positions:
            return self._get_default_metrics()
        
        final_pos = positions[-1]
        
        # Basic climbing metrics
        height_climbed = max(0, max_height - initial_pos[2])
        final_height = final_pos[2]
        height_lost = max(0, initial_pos[2] - min_height)
        
        # Enhanced distance metrics
        try:
            if hasattr(creature_obj, 'get_distance_travelled'):
                distance_travelled = creature_obj.get_distance_travelled()
                if distance_travelled is None or distance_travelled <= 0:
                    distance_travelled = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
            else:
                distance_travelled = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
        except:
            distance_travelled = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
        
        # Movement efficiency metrics
        total_path_length = 0
        movement_efficiency = 0
        
        if len(positions) > 1:
            total_path_length = sum(
                np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
                for i in range(1, len(positions))
            )
            straight_line_distance = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
            movement_efficiency = straight_line_distance / max(total_path_length, 0.001)
        
        # Enhanced stability metrics
        stability_score = 0
        if len(velocities) > 1:
            try:
                velocity_magnitudes = [np.linalg.norm(v) for v in velocities]
                avg_velocity = np.mean(velocity_magnitudes)
                velocity_variance = np.var(velocity_magnitudes)
                
                # Reward controlled movement
                speed_score = max(0, 2 - avg_velocity / 8)  # Adjusted for better scaling
                consistency_score = max(0, 1 - velocity_variance / 15)  # Penalty for erratic movement
                stability_score = (speed_score + consistency_score) / 2
            except:
                stability_score = 0
        
        # Position-based metrics
        center_distance = np.sqrt(final_pos[0]**2 + final_pos[1]**2)
        center_bonus = max(0, 5 - center_distance / 4)  # Bonus for staying near mountain center
        
        # Enhanced height progression analysis
        upward_trend_bonus = 0
        if len(positions) >= 10:
            height_progression = [pos[2] for pos in positions]
            
            # Calculate trend using linear regression
            x_vals = np.arange(len(height_progression))
            try:
                slope, intercept = np.polyfit(x_vals, height_progression, 1)
                upward_trend_bonus = max(0, slope * 15)  # Bonus for consistent upward movement
            except:
                upward_trend_bonus = 0
        
        # Mountain proximity bonus
        mountain_proximity_bonus = 0
        if center_distance < self.arena_size / 4:
            mountain_proximity_bonus = 2 * (1 - center_distance / (self.arena_size / 4))
        
        # Survival bonus
        survival_bonus = len(positions) / max(len(positions), 100) * 2  # Bonus for longer survival
        
        # Composite fitness calculation with enhanced weighting
        primary_fitness = height_climbed * 20          # Primary objective: climb high
        secondary_fitness = final_height * 10          # Secondary: end up high
        efficiency_fitness = movement_efficiency * 6   # Reward efficient movement
        stability_fitness = stability_score * 5        # Reward stable movement
        center_fitness = center_bonus * 3              # Stay near mountain
        trend_fitness = upward_trend_bonus * 4         # Consistent upward progress
        proximity_fitness = mountain_proximity_bonus * 2  # Near mountain bonus
        survival_fitness = survival_bonus             # Survival bonus
        
        total_fitness = (primary_fitness + secondary_fitness + efficiency_fitness + 
                        stability_fitness + center_fitness + trend_fitness + 
                        proximity_fitness + survival_fitness)
        
        # Additional performance metrics
        climbing_efficiency = height_climbed / max(total_path_length, 0.001)
        average_height = np.mean([pos[2] for pos in positions]) if positions else 0
        height_variance = np.var([pos[2] for pos in positions]) if len(positions) > 1 else 0
        
        return {
            'fitness': max(0, total_fitness),
            'max_height': max_height,
            'final_height': final_height,
            'height_climbed': height_climbed,
            'height_lost': height_lost,
            'distance_travelled': distance_travelled,
            'total_path_length': total_path_length,
            'movement_efficiency': movement_efficiency,
            'stability': stability_score,
            'center_distance': center_distance,
            'center_bonus': center_bonus,
            'upward_trend_bonus': upward_trend_bonus,
            'mountain_proximity_bonus': mountain_proximity_bonus,
            'survival_bonus': survival_bonus,
            'climbing_efficiency': climbing_efficiency,
            'average_height': average_height,
            'height_variance': height_variance,
            'final_position': final_pos,
            'initial_position': initial_pos,
            'num_positions_recorded': len(positions),
            'simulation_error': False,
            
            # Detailed fitness breakdown
            'fitness_breakdown': {
                'primary_fitness': primary_fitness,
                'secondary_fitness': secondary_fitness,
                'efficiency_fitness': efficiency_fitness,
                'stability_fitness': stability_fitness,
                'center_fitness': center_fitness,
                'trend_fitness': trend_fitness,
                'proximity_fitness': proximity_fitness,
                'survival_fitness': survival_fitness
            }
        }
    
    def _get_default_metrics(self) -> Dict:
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
            'mountain_proximity_bonus': 0,
            'survival_bonus': 0,
            'climbing_efficiency': 0,
            'average_height': 0,
            'height_variance': 0,
            'final_position': (0, 0, 0),
            'initial_position': (0, 0, 0),
            'num_positions_recorded': 0,
            'simulation_error': True,
            'fitness_breakdown': {
                'primary_fitness': 0,
                'secondary_fitness': 0,
                'efficiency_fitness': 0,
                'stability_fitness': 0,
                'center_fitness': 0,
                'trend_fitness': 0,
                'proximity_fitness': 0,
                'survival_fitness': 0
            }
        }
    
    def reset_environment(self):
        """Enhanced environment reset with comprehensive error handling"""
        try:
            # Remove all creatures with enhanced error handling
            creatures_to_remove = self.creatures.copy()
            self.creatures = []
            
            for creature_id in creatures_to_remove:
                try:
                    if self._creature_exists(creature_id):
                        p.removeBody(creature_id)
                except Exception as remove_error:
                    self.logger.warning(f"Error removing creature {creature_id}: {remove_error}")
            
            # Enhanced cleanup of temporary files
            self._cleanup_temp_files()
            
            # Reset simulation with enhanced error handling
            try:
                p.resetSimulation()
            except Exception as reset_error:
                self.logger.error(f"Failed to reset simulation: {reset_error}")
                # Try to reconnect physics
                return self._reconnect_physics()
            
            # Recreate basic environment
            try:
                p.setGravity(0, 0, -10)
                p.setPhysicsEngineParameter(
                    enableFileCaching=0,
                    numSolverIterations=10,
                    numSubSteps=1
                )
                p.setTimeStep(self.physics_timestep)
                
                if not self.create_arena():
                    raise RuntimeError("Failed to recreate arena")
                
                if not self.load_mountain():
                    raise RuntimeError("Failed to reload mountain")
                    
            except Exception as setup_error:
                self.logger.error(f"Failed to recreate environment: {setup_error}")
                return self._reconnect_physics()
            
            self.logger.debug("Environment reset completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment reset failed: {e}")
            return self._reconnect_physics()
    
    def _cleanup_temp_files(self):
        """Enhanced cleanup of temporary URDF files"""
        try:
            current_time = time.time()
            for filename in os.listdir('.'):
                if (filename.startswith('temp_creature_') and 
                    filename.endswith('.urdf')):
                    try:
                        # Remove files older than 5 minutes or if they're very old
                        file_age = current_time - os.path.getmtime(filename)
                        if file_age > 300:  # 5 minutes
                            os.remove(filename)
                            self.logger.debug(f"Cleaned up old temp file: {filename}")
                    except Exception as file_error:
                        self.logger.debug(f"Could not remove temp file {filename}: {file_error}")
        except Exception as cleanup_error:
            self.logger.debug(f"Error during temp file cleanup: {cleanup_error}")
    
    def _reconnect_physics(self):
        """Attempt to reconnect physics engine"""
        try:
            self.logger.info("Attempting to reconnect physics engine...")
            self.cleanup()
            
            if self.initialize_physics():
                if self.create_arena() and self.load_mountain():
                    self.logger.info("Physics reconnection successful")
                    return True
            
            self.logger.error("Physics reconnection failed")
            return False
            
        except Exception as e:
            self.logger.error(f"Physics reconnection error: {e}")
            return False
    
    def cleanup(self):
        """Enhanced cleanup with comprehensive resource management"""
        try:
            # Clean up creatures
            if self.creatures:
                creatures_to_remove = self.creatures.copy()
                for creature_id in creatures_to_remove:
                    try:
                        if self._creature_exists(creature_id):
                            p.removeBody(creature_id)
                    except:
                        pass
                self.creatures = []
            
            # Clean up temp files
            self._cleanup_temp_files()
            
            # Disconnect physics with error handling
            if self.physics_client is not None:
                try:
                    p.disconnect(self.physics_client)
                except:
                    pass
                self.physics_client = None
                
            # Reset environment state
            self.mountain_id = None
            self.floor_id = None
            self.wall_ids = []
            self.creature_counter = 0
            
            self.logger.debug("Environment cleanup completed")
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_environment_info(self) -> Dict:
        """Get comprehensive information about the current environment"""
        try:
            active_creatures = len([cid for cid in self.creatures if self._creature_exists(cid)])
            
            return {
                'arena_size': self.arena_size,
                'mountain_height': self.mountain_height,
                'time_limit': self.time_limit,
                'simulation_hz': self.simulation_hz,
                'gui_enabled': self.gui,
                'physics_client_active': self.physics_client is not None,
                'physics_client_id': self.physics_client,
                'mountain_loaded': self.mountain_id is not None,
                'mountain_id': self.mountain_id,
                'floor_id': self.floor_id,
                'num_walls': len(self.wall_ids),
                'num_active_creatures': active_creatures,
                'total_creatures_spawned': self.creature_counter,
                'spawn_parameters': {
                    'height_offset': self.spawn_height_offset,
                    'radius_min': self.spawn_radius_min,
                    'radius_max': self.spawn_radius_max
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting environment info: {e}")
            return {'error': str(e)}
    
    def get_creature_info(self, creature_id: int) -> Dict:
        """Get detailed information about a specific creature"""
        try:
            if not self._creature_exists(creature_id):
                return {'error': 'Creature does not exist', 'creature_id': creature_id}
            
            pos, orn = p.getBasePositionAndOrientation(creature_id)
            vel, ang_vel = p.getBaseVelocity(creature_id)
            
            num_joints = p.getNumJoints(creature_id)
            joint_info = []
            
            for i in range(num_joints):
                try:
                    joint_state = p.getJointState(creature_id, i)
                    joint_info.append({
                        'joint_id': i,
                        'position': joint_state[0],
                        'velocity': joint_state[1],
                        'force': joint_state[3]
                    })
                except:
                    pass
            
            return {
                'creature_id': creature_id,
                'position': pos,
                'orientation': orn,
                'linear_velocity': vel,
                'angular_velocity': ang_vel,
                'num_joints': num_joints,
                'joint_states': joint_info,
                'distance_from_center': np.sqrt(pos[0]**2 + pos[1]**2),
                'height_above_ground': pos[2]
            }
            
        except Exception as e:
            return {'error': str(e), 'creature_id': creature_id}
    
    def set_simulation_parameters(self, **kwargs):
        """Set simulation parameters dynamically"""
        try:
            if 'time_limit' in kwargs:
                self.time_limit = max(1, kwargs['time_limit'])
            
            if 'simulation_hz' in kwargs:
                self.simulation_hz = max(60, min(1000, kwargs['simulation_hz']))
                self.physics_timestep = 1.0 / self.simulation_hz
                if self.physics_client is not None:
                    p.setTimeStep(self.physics_timestep)
            
            if 'spawn_height_offset' in kwargs:
                self.spawn_height_offset = max(2.0, kwargs['spawn_height_offset'])
            
            if 'spawn_radius_min' in kwargs:
                self.spawn_radius_min = max(3.0, kwargs['spawn_radius_min'])
            
            if 'spawn_radius_max' in kwargs:
                self.spawn_radius_max = max(self.spawn_radius_min + 1, kwargs['spawn_radius_max'])
            
            self.logger.info("Simulation parameters updated")
            
        except Exception as e:
            self.logger.error(f"Error setting simulation parameters: {e}")
    
    def get_simulation_stats(self) -> Dict:
        """Get comprehensive simulation statistics"""
        try:
            return {
                'total_creatures_spawned': self.creature_counter,
                'active_creatures': len([cid for cid in self.creatures if self._creature_exists(cid)]),
                'physics_client_active': self.physics_client is not None,
                'environment_ready': (self.mountain_id is not None and 
                                    self.floor_id is not None and 
                                    len(self.wall_ids) > 0),
                'simulation_parameters': {
                    'time_limit': self.time_limit,
                    'simulation_hz': self.simulation_hz,
                    'physics_timestep': self.physics_timestep,
                    'arena_size': self.arena_size,
                    'mountain_height': self.mountain_height
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting simulation stats: {e}")
            return {'error': str(e)}


# Enhanced test function with comprehensive validation
def test_mountain_environment():
    """Comprehensive test of the mountain environment functionality"""
    print("🏔️ Testing Enhanced Mountain Environment...")
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Environment creation
    total_tests += 1
    env = MountainEnvironment(gui=False, arena_size=20)
    
    if not env.initialize_physics():
        print("❌ Failed to initialize physics")
        return False
    
    if not env.create_arena():
        print("❌ Failed to create arena")
        return False
    
    if not env.load_mountain():
        print("❌ Failed to load mountain")
        return False
    
    print("✅ Environment setup successful")
    success_count += 1
    
    # Test 2: Environment info
    total_tests += 1
    try:
        env_info = env.get_environment_info()
        if 'arena_size' in env_info and env_info['physics_client_active']:
            print("✅ Environment info retrieval successful")
            success_count += 1
        else:
            print("❌ Environment info incomplete")
    except Exception as e:
        print(f"❌ Environment info failed: {e}")
    
    # Test 3: Creature spawning and simulation
    total_tests += 1
    try:
        import creature
        test_creature = creature.Creature(gene_count=4)
        
        # Validate creature first
        if env.validate_creature_before_spawn(test_creature):
            print("✅ Creature validation successful")
            
            creature_id = env.spawn_creature(test_creature)
            
            if creature_id is not None and creature_id >= 0:
                print(f"✅ Creature spawned successfully (ID: {creature_id})")
                
                # Test creature info
                creature_info = env.get_creature_info(creature_id)
                if 'position' in creature_info:
                    print("✅ Creature info retrieval successful")
                
                # Test simulation
                results = env.simulate_creature(test_creature, creature_id, simulation_steps=1000)
                
                print(f"✅ Simulation completed")
                print(f"   Fitness: {results['fitness']:.3f}")
                print(f"   Max Height: {results['max_height']:.3f}")
                print(f"   Distance Climbed: {results['height_climbed']:.3f}")
                print(f"   Simulation Steps: {results.get('simulation_steps', 0)}")
                
                if results.get('simulation_error'):
                    print("   ⚠️ Simulation had errors")
                else:
                    print("   ✅ Simulation completed successfully")
                    success_count += 1
                
                # Test detailed metrics
                if 'fitness_breakdown' in results:
                    print("✅ Detailed fitness breakdown available")
                    breakdown = results['fitness_breakdown']
                    print(f"   Primary fitness: {breakdown['primary_fitness']:.2f}")
                    print(f"   Stability fitness: {breakdown['stability_fitness']:.2f}")
                
            else:
                print("❌ Failed to spawn creature")
        else:
            print("❌ Creature validation failed")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Test 4: Environment reset
    total_tests += 1
    try:
        if env.reset_environment():
            print("✅ Environment reset successful")
            success_count += 1
        else:
            print("❌ Environment reset failed")
    except Exception as e:
        print(f"❌ Environment reset error: {e}")
    
    # Test 5: Multiple creature spawning
    total_tests += 1
    try:
        creatures_spawned = 0
        for i in range(3):
            test_creature = creature.Creature(gene_count=random.randint(3, 6))
            creature_id = env.spawn_creature(test_creature)
            if creature_id is not None and creature_id >= 0:
                creatures_spawned += 1
        
        if creatures_spawned >= 2:
            print(f"✅ Multiple creature spawning successful ({creatures_spawned}/3)")
            success_count += 1
        else:
            print(f"❌ Multiple creature spawning failed ({creatures_spawned}/3)")
    except Exception as e:
        print(f"❌ Multiple spawning test failed: {e}")
    
    # Cleanup
    try:
        env.cleanup()
        print("✅ Environment cleanup successful")
    except Exception as e:
        print(f"⚠️ Cleanup had issues: {e}")
    
    # Summary
    print(f"\n📊 Test Summary: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Environment is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    test_mountain_environment()