import numpy as np

def nautical_to_meters(nm_value):
    return nm_value * 1852

# Global compression scale: brings obstacles closer while maintaining relative geometry
# 0.35 scale means a 6 nmi obstacle becomes ~2.1 nmi, 3.4 nmi becomes ~1.2 nmi
# This creates realistic close-range encounters without extending training time
GLOBAL_COMPRESSION_SCALE = 0.35

obstacle_cases = {
    # STRONGLY VALID CASES (collision/close encounter)
    "Case 1": [[[nautical_to_meters(6), nautical_to_meters(0)], 180]],  # Head-on collision - no change
    
    # ENHANCED WEAK CASES - adjusted for stronger collision potential
    # Case 2: was WEAK at 5 nmi heading 90° - move closer and angle for head-on
    "Case 2": [[[nautical_to_meters(3.5), nautical_to_meters(-1.5)], 135]],  # Closer, diagonal threat
    
    # Case 3: was DIVERGING at 3 nmi heading 0° (moving away) - make head-on
    "Case 3": [[[nautical_to_meters(3.5), nautical_to_meters(0)], 180]],  # Head-on collision
    
    # Case 4: was WEAK at 3.44 nmi heading 295° - strengthen heading for crossing
    "Case 4": [[[nautical_to_meters(3.0), nautical_to_meters(1.0)], 225]],  # Closer, head-on-ish
    
    # Case 5: Already has one collision threat - keep as is
    "Case 5": [[[nautical_to_meters(5), nautical_to_meters(-2.0-0.14)], 90], [[nautical_to_meters(7-0.05), nautical_to_meters(0)], 180]],
    
    # Case 6: both WEAK/DIVERGING - strengthen both
    "Case 6": [[[nautical_to_meters(2.5), nautical_to_meters(-0.8)], 90], [[nautical_to_meters(2.8), nautical_to_meters(-0.2)], 135]],  # Both closer, threatening headings
    
    # Case 7: one DIVERGING, one WEAK - strengthen both
    "Case 7": [[[nautical_to_meters(2.8), nautical_to_meters(0.5)], 180], [[nautical_to_meters(2.5), nautical_to_meters(-1.0)], 90]],  # Head-on + crossing
    
    # Case 8: Already has collision threat - keep as is
    "Case 8": [[[nautical_to_meters(5), nautical_to_meters(-2.13)], 90], [[nautical_to_meters(7), nautical_to_meters(0)], 180]],
    
    # Case 9: both WEAK - move closer and adjust headings
    "Case 9": [[[nautical_to_meters(2.8), nautical_to_meters(-0.8)], 90], [[nautical_to_meters(2.5), nautical_to_meters(-1.0)], 135]],  # Both closer, threatening
    
    # Case 10: one DIVERGING, one WEAK - strengthen
    "Case 10": [[[nautical_to_meters(2.5), nautical_to_meters(0.0)], 180], [[nautical_to_meters(2.8), nautical_to_meters(-1.0)], 90]],  # Head-on + crossing
    
    # Case 11: both WEAK - strengthen
    "Case 11": [[[nautical_to_meters(2.5), nautical_to_meters(1.0)], -135], [[nautical_to_meters(2.8), nautical_to_meters(-0.8)], 90]],  # Both threatening
    
    # Case 12: Already has collision threat - keep as is
    "Case 12": [[[nautical_to_meters(7), nautical_to_meters(0)], 180], [[nautical_to_meters(3), nautical_to_meters(0.3+0.05)], -10], [[nautical_to_meters(3.44), nautical_to_meters(-1.55+0.05)], 45]],
    
    # Case 13: Already has collision threat - keep as is
    "Case 13": [[[nautical_to_meters(6), nautical_to_meters(0)], 180], [[nautical_to_meters(3), nautical_to_meters(0.3+0.05)], 350], [[nautical_to_meters(3.4), nautical_to_meters(1.5+0.05)], 295]],
    
    # Case 14: all WEAK - strengthen all three
    "Case 14": [[[nautical_to_meters(2.5), nautical_to_meters(-0.8)], 90], [[nautical_to_meters(2.8), nautical_to_meters(-0.3)], 135], [[nautical_to_meters(2.5), nautical_to_meters(-1.2)], 115]],
    
    # Case 15: all WEAK/DIVERGING - strengthen
    "Case 15": [[[nautical_to_meters(2.8), nautical_to_meters(0.2)], 180], [[nautical_to_meters(2.5), nautical_to_meters(-0.8)], 90], [[nautical_to_meters(2.5), nautical_to_meters(-1.0)], 135]],
    
    # Case 16: all WEAK/DIVERGING - strengthen
    "Case 16": [[[nautical_to_meters(2.8), nautical_to_meters(0.8)], -90], [[nautical_to_meters(2.5), nautical_to_meters(1.0)], -135], [[nautical_to_meters(2.5), nautical_to_meters(-1.0)], 90]],
    
    # Case 17: all WEAK/DIVERGING - strengthen
    "Case 17": [[[nautical_to_meters(2.8), nautical_to_meters(0.0)], 180], [[nautical_to_meters(2.5), nautical_to_meters(0.3)], -135], [[nautical_to_meters(2.5), nautical_to_meters(-0.8)], 90]],
    
    # Case 18: has MODERATE - improve the moderate one and strengthen weak
    "Case 18": [[[nautical_to_meters(2.5), nautical_to_meters(-0.3)], 135], [[nautical_to_meters(2.8), nautical_to_meters(-1.0)], 90], [[nautical_to_meters(2.0), nautical_to_meters(-0.8)], 120]],  # All closer, all stronger
    
    # Case 19: has MODERATE - similar strengthening
    "Case 19": [[[nautical_to_meters(2.5), nautical_to_meters(-0.2)], 135], [[nautical_to_meters(2.5), nautical_to_meters(0.2)], -135], [[nautical_to_meters(2.0), nautical_to_meters(-0.8)], 120]],
    
    # Case 20: all WEAK/DIVERGING - strengthen
    "Case 20": [[[nautical_to_meters(2.8), nautical_to_meters(0.0)], 180], [[nautical_to_meters(2.5), nautical_to_meters(-0.3)], 135], [[nautical_to_meters(2.5), nautical_to_meters(-1.0)], 90]],
    
    # Case 21: all DIVERGING (after Agent 3 fix) - strengthen all
    "Case 21": [[[nautical_to_meters(2.5), nautical_to_meters(-0.2)], 135], [[nautical_to_meters(2.5), nautical_to_meters(0.2)], -135], [[nautical_to_meters(2.5), nautical_to_meters(-0.5)], 90]],  # All closer, all threatening
    
    # Case 22: all WEAK/DIVERGING - strengthen
    "Case 22": [[[nautical_to_meters(2.8), nautical_to_meters(0.0)], 180], [[nautical_to_meters(2.5), nautical_to_meters(-0.8)], 90], [[nautical_to_meters(2.5), nautical_to_meters(-1.0)], 135]],
    
    # Case 23: WEAK - strengthen
    "Case 23": [[[nautical_to_meters(3.0), nautical_to_meters(1.0)], -90]],  # Closer, lateral threat
    
}

# Function to get obstacles for a specific case
def get_obstacles(case_number):
    case_key = f"Case {case_number}"
    return obstacle_cases.get(case_key, [])


def get_obstacle_data(case_number):
    """
    Convert obstacle case data to simulation format
    Args:
        case_number (int): The case number to use (1-22)
    Returns:
        Xob, Yob (lists): X and Y positions in meters
        Vob (list): Velocities in m/s 
        psiob (numpy array): Angles in radians
    """
    # Get obstacle data for the case
    obstacles = get_obstacles(case_number)
    
    # Initialize empty lists
    Xob = []
    Yob = []
    psiob = []
    
    # Default velocity (you may want to adjust this)
    Vob = [18.52] * len(obstacles)  # Assuming 9.5 m/s for all obstacles
    
    # Extract positions and angles from obstacles
    for obstacle in obstacles:
        position = obstacle[0]  # Get [x,y] position
        angle = obstacle[1]    # Get angle in degrees
        
        # Apply global compression scale to bring obstacles closer (maintains geometry)
        Xob.append(position[0] * GLOBAL_COMPRESSION_SCALE)
        Yob.append(position[1] * GLOBAL_COMPRESSION_SCALE)
        psiob.append(np.radians(angle))  # Convert angle to radians
    
    return Xob, Yob, Vob, np.array(psiob)

