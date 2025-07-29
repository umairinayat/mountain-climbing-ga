#!/usr/bin/env python3
"""
Debug script to identify and fix creature spawning issues
Run this to diagnose the problem with your creature spawning
"""

import os
import sys
import logging
import tempfile
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_creature_creation():
    """Test basic creature creation"""
    try:
        import creature
        logger.info("Testing basic creature creation...")
        
        # Test creature creation with different gene counts
        for gene_count in [3, 5, 8]:
            try:
                test_creature = creature.Creature(gene_count=gene_count)
                logger.info(f"✅ Created creature with {gene_count} genes")
                
                # Test DNA access
                if hasattr(test_creature, 'dna') and test_creature.dna:
                    logger.info(f"✅ Creature has DNA with {len(test_creature.dna)} genes")
                else:
                    logger.error(f"❌ Creature missing DNA or empty DNA")
                    return False
                
                # Test XML generation
                if hasattr(test_creature, 'to_xml'):
                    xml_content = test_creature.to_xml()
                    if xml_content and len(xml_content.strip()) > 0:
                        logger.info(f"✅ Generated XML ({len(xml_content)} characters)")
                        
                        # Save sample XML for inspection
                        with open(f"debug_creature_{gene_count}_genes.xml", 'w') as f:
                            f.write(xml_content)
                        logger.info(f"📁 Saved sample XML to debug_creature_{gene_count}_genes.xml")
                        
                        # Check for required XML elements
                        if '<robot' in xml_content and '</robot>' in xml_content:
                            logger.info("✅ XML contains required robot tags")
                        else:
                            logger.error("❌ XML missing required robot tags")
                            return False
                    else:
                        logger.error(f"❌ Generated empty or invalid XML")
                        return False
                else:
                    logger.error(f"❌ Creature missing to_xml method")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Failed to create creature with {gene_count} genes: {e}")
                return False
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Cannot import creature module: {e}")
        return False

def test_pybullet_basic():
    """Test basic PyBullet functionality"""
    try:
        import pybullet as p
        logger.info("Testing PyBullet basic functionality...")
        
        # Test connection
        physics_client = p.connect(p.DIRECT)
        if physics_client >= 0:
            logger.info("✅ PyBullet connected successfully")
        else:
            logger.error("❌ PyBullet connection failed")
            return False
        
        # Test basic setup
        p.setGravity(0, 0, -10)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        logger.info("✅ Basic physics setup completed")
        
        # Test simple object creation
        box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 1])
        if box_id >= 0:
            logger.info("✅ Created collision shape")
        else:
            logger.error("❌ Failed to create collision shape")
            return False
        
        # Test multibody creation
        body_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=box_id, basePosition=[0, 0, 5])
        if body_id >= 0:
            logger.info("✅ Created test multibody")
            
            # Test position retrieval
            pos, orn = p.getBasePositionAndOrientation(body_id)
            logger.info(f"✅ Retrieved position: {pos}")
        else:
            logger.error("❌ Failed to create multibody")
            return False
        
        # Cleanup
        p.disconnect()
        logger.info("✅ PyBullet test completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Cannot import pybullet: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ PyBullet test failed: {e}")
        return False

def test_urdf_loading():
    """Test URDF file creation and loading"""
    try:
        import pybullet as p
        import creature
        
        logger.info("Testing URDF creation and loading...")
        
        # Connect to PyBullet
        physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)
        
        # Create a simple creature
        test_creature = creature.Creature(gene_count=4)
        
        # Generate XML
        xml_content = test_creature.to_xml()
        logger.info(f"Generated XML content length: {len(xml_content)}")
        
        # Create temporary URDF file
        temp_urdf = "debug_test_creature.urdf"
        
        try:
            with open(temp_urdf, 'w', encoding='utf-8') as f:
                f.write(xml_content)
                f.flush()
                os.fsync(f.fileno())
            
            logger.info(f"✅ Created URDF file: {temp_urdf}")
            
            # Verify file exists and has content
            if os.path.exists(temp_urdf):
                file_size = os.path.getsize(temp_urdf)
                logger.info(f"✅ URDF file exists, size: {file_size} bytes")
                
                if file_size > 0:
                    # Try to load the URDF
                    try:
                        creature_id = p.loadURDF(temp_urdf, [0, 0, 5])
                        
                        if creature_id >= 0:
                            logger.info(f"✅ Successfully loaded URDF, creature ID: {creature_id}")
                            
                            # Test position retrieval
                            try:
                                pos, orn = p.getBasePositionAndOrientation(creature_id)
                                logger.info(f"✅ Retrieved creature position: {pos}")
                                return True
                            except Exception as pos_error:
                                logger.error(f"❌ Failed to get creature position: {pos_error}")
                                return False
                        else:
                            logger.error(f"❌ loadURDF returned invalid ID: {creature_id}")
                            return False
                            
                    except Exception as load_error:
                        logger.error(f"❌ Failed to load URDF: {load_error}")
                        
                        # Print URDF content for debugging
                        logger.info("URDF content for debugging:")
                        with open(temp_urdf, 'r') as f:
                            content = f.read()
                            logger.info(content[:500] + "..." if len(content) > 500 else content)
                        
                        return False
                else:
                    logger.error("❌ URDF file is empty")
                    return False
            else:
                logger.error("❌ URDF file was not created")
                return False
                
        finally:
            # Cleanup
            try:
                if os.path.exists(temp_urdf):
                    os.remove(temp_urdf)
                p.disconnect()
            except:
                pass
        
    except Exception as e:
        logger.error(f"❌ URDF test failed: {e}")
        return False

def test_environment_setup():
    """Test MountainEnvironment setup"""
    try:
        from mountain_environment import MountainEnvironment
        logger.info("Testing MountainEnvironment setup...")
        
        # Create environment
        env = MountainEnvironment(gui=False, arena_size=20)
        
        # Test initialization
        if env.initialize_physics():
            logger.info("✅ Environment physics initialized")
        else:
            logger.error("❌ Environment physics initialization failed")
            return False
        
        # Test arena creation
        if env.create_arena():
            logger.info("✅ Arena created successfully")
        else:
            logger.error("❌ Arena creation failed")
            return False
        
        # Test mountain loading
        if env.load_mountain():
            logger.info("✅ Mountain loaded successfully")
        else:
            logger.error("❌ Mountain loading failed")
            return False
        
        # Test creature spawning
        import creature
        test_creature = creature.Creature(gene_count=4)
        
        creature_id = env.spawn_creature(test_creature)
        if creature_id is not None and creature_id >= 0:
            logger.info(f"✅ Creature spawned successfully, ID: {creature_id}")
            return True
        else:
            logger.error(f"❌ Creature spawning failed, ID: {creature_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        return False
    finally:
        try:
            env.cleanup()
        except:
            pass

def run_comprehensive_debug():
    """Run all debug tests"""
    logger.info("🔍 Starting comprehensive debug analysis...")
    logger.info("=" * 60)
    
    tests = [
        ("Creature Creation", test_creature_creation),
        ("PyBullet Basic", test_pybullet_basic),
        ("URDF Loading", test_urdf_loading),
        ("Environment Setup", test_environment_setup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} test CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 DEBUG SUMMARY:")
    logger.info("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:20s} : {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your system should work correctly.")
    else:
        logger.error("⚠️  Some tests failed. Check the errors above.")
        
        # Provide specific recommendations
        if not results.get("Creature Creation", True):
            logger.error("➤ Fix: Check your creature.py implementation")
        
        if not results.get("PyBullet Basic", True):
            logger.error("➤ Fix: Reinstall PyBullet: pip install pybullet")
        
        if not results.get("URDF Loading", True):
            logger.error("➤ Fix: Check creature XML generation and URDF format")
        
        if not results.get("Environment Setup", True):
            logger.error("➤ Fix: Check mountain_environment.py implementation")

if __name__ == "__main__":
    run_comprehensive_debug()