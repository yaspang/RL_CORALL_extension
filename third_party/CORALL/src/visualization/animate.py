import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle 
import numpy as np

from src.visualization.rendering import animate_ship, animate_static_obstacle

# Function to animate ship and obstacles

def animate_step(x, y, psi, LOA_own, BOL_own, CPA_own, Xob, Yob, psiob, LOA_ob, BOL_ob, CPA_ob, Risk, Vob, step, l, ax):
    
    
    if step % 100 == 0:
        animate_ship(x, y, psi, LOA_own * 5, BOL_own * 5, CPA_own, [0.41, 0, 0.41], ax)
        #plt.xlabel(r'$X$ (nmi)', fontsize=20)  # Set x-axis label
        #plt.ylabel(r'$Y$ (nmi)', fontsize=20)  # Set y-axis label
        #plt.xlim(15, 20)
        #plt.ylim(-5, 2)
        plt.draw()  # Update the plot
        plt.pause(0.1)  


    # Uncomment this section if you want to handle obstacles
    
    for j in range(len(Xob)):
        obs_col = [0.0, 0.7, 0.0]
        if Risk[j] > 0.75:
            obs_col = [1.0, 0.0, 0.0]
        elif Risk[j] > 0.6:
            obs_col = [1.0, 0.6, 0.0]
        elif Risk[j] > 0.35:
            obs_col = [1.0, 0.9, 0.0]

        if Vob[j] > 0.5:
                
            # Define colors for ships
            colors = [
                        [0, 0, 1],  # Blue
                        [1, 0.5, 0],  # Orange
                        [0, 1, 0]   # Green
                    ]
                    
            animate_ship(Xob[j], Yob[j], psiob[j], LOA_ob[j] * 3, BOL_ob[j] * 3, CPA_ob[j], colors[j], ax)
                    
                    

                    

            plt.draw()  # Update the plot
            plt.pause(0.1)  
                            
        else:
            animate_static_obstacle(Xob[j], Yob[j], CPA_ob[j], obs_col, ax)


def animate_step_dense(
    x, y, psi, LOA_own, BOL_own, CPA_own,
    Xob, Yob, psiob, LOA_ob, BOL_ob, CPA_ob,
    Risk, Vob,
    step, ax
):
    """
    Dense (every-step) animation renderer 
    
    Handle camera control by setting x and y axis limits (avoid zoom behavior)
    """

    # Always draw onto the provided axes
    if ax is None:
        raise ValueError("animate_step_dense requires an 'ax' (matplotlib axes)")
    
    # normalize inputs into 1D numpy arrays
    Xob = np.asarray(Xob, dtype=float).reshape(-1)
    Yob = np.asarray(Yob, dtype=float).reshape(-1)
    psiob = np.asarray(psiob, dtype=float).reshape(-1)

    LOA_ob = np.asarray(LOA_ob, dtype=float).reshape(-1)
    BOL_ob = np.asarray(BOL_ob, dtype=float).reshape(-1)
    CPA_ob = np.asarray(CPA_ob, dtype=float).reshape(-1)

    Risk = np.asarray(Risk, dtype=float).reshape(-1) if Risk is not None else np.zeros(len(Xob), dtype=float)
    Vob = np.asarray(Vob, dtype=float).reshape(-1) if Vob is not None else np.zeros(len(Xob), dtype=float)
    
    # ---- Ownship ----
    own_col = [0.41, 0.0, 0.41] # ownship color 
    animate_ship(x, y, psi, LOA_own * 5, BOL_own * 5, CPA_own, own_col, ax=ax)


    # ---- other ships / obstacles (always) ----
    moving_col = [
        [0.0, 0.0, 1.0],      # blue
        [1.0, 0.5, 0.0],      # orange
        [0.0, 1.0, 0.0],      # green
        [0.2, 0.8, 0.8],      # cyan
        [1.0, 0.0, 1.0]       # magenta
    ]

    n = len(Xob)

    for j in range(n):
        is_moving = (j < len(Vob)) and (Vob[j] > 0.1) # moving if velocity entry exists and if Vob[j] is large treat as a moving obstacle

        if is_moving: 
            col = moving_col[j % len(moving_col)]
            animate_ship(
                Xob[j], Yob[j], psiob[j], LOA_ob[j] * 3, BOL_ob[j] * 3, CPA_ob[j], 
                col, ax
            )
        
        else: 
            # static obstacles are colored by risk threshold (green -> yellow -> orange -> red)
            r = Risk[j] if j < len(Risk) else 0.0
            obs_col = [0.0, 0.7, 0.0]

            if Risk[j] > 0.75:
                obs_col = [1.0, 0.0, 0.0] # red
            elif Risk[j] > 0.6:
                obs_col = [1.0, 0.6, 0.0] # orange
            elif Risk[j] > 0.35:
                obs_col = [1.0, 0.9, 0.0] # yellow

            animate_static_obstacle(Xob[j], Yob[j], CPA_ob[j], obs_col, ax)


        