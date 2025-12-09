
import numpy as np
import matplotlib.pyplot as plt
from model_1 import Model_1
#from Ballistic.colored_messages import *
#from Ballistic.constantes import *

task = 0

if task == 0:

    model = Model_1(h=20, v_0=50, alpha=40)
        

#__________________________________________________________________________________________________
    impact_values = model.set_impact_values()
    model.set_trajectory(impact_values["t_i"],50)     # reponse a la question 9.2)2 

#__________________________________________________________________________________________________
   # figure1
    model.plot_component()                  # reponse a la question 9.2)3
    model.plot_trajectory()


task=1

if task==1:

    model = Model_1(h=20, v_0=50, alpha=40)
    alphas = np.arange(20, 71, 5)  # Angles de 20Â° Ã  70Â° par pas de 5Â°
#_________________________________________________________________________________________________
        # Trajectoires
    trajectoriess=model.set_trajectories(alphas)
#__________________________________________________________________________________________________
    x_list, z_list = trajectoriess[0], trajectoriess[1]  # reponse a la question 10.1)1

#_________________________________________________________________________________________________
        # TracÃ© des trajectoires
    model.plot_trajectories(x_list,z_list,alphas)  # reponse a la question 10.1)2

#_________________________________________________________________________________________________                      
     # PortÃ©e maximale
    model.plot_maximum_distance(alphas)
#_________________________________________________________________________________________________
        # Angle optimal
    optimal_alpha, optimal_distance = model.find_optimal_angle(alphas)
    print(f"Angle optimal: {optimal_alpha}Â°, Portée maximale: {optimal_distance:.2f} m")

#_________________________________________________________________________________________________
        # Altitude maximale
    model.plot_maximum_height(alphas)

    
   