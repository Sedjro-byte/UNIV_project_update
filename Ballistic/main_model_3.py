import numpy as np                      # module de math
import matplotlib.pyplot as plt         # module graphique
from model_3 import Model_3

task = 0

if task == 0:         # test of AnalyticalModel
    t_end, alpha_ref = 3, 20
    model_2 = Model_3({"v_0": 20, "h": 20, "alpha": 40, "npt": 101, "area": 0.01, "rho":1.3 , "mass": 0.1, "Cl":0.18, "Cd": 0.01})
    model_2.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    print("solution:", model_2.x, model_2.z)
    plt.figure(0)
    #______________________________________________________________________________  ICI C EST LA VALIDATION PAR RAPPORTAL ERREUR 
    model_2.validation(3, 101)
    plt.figure(1)
    #______________________________________________________________________________
    model_2.plot_trajectory(3,40)

elif task == 1:         # test of AnalyticalModel
    t_end = 3
    alpha_ref = 20
    model_2 = Model_3({"v_0": 20, "h": 20, "alpha": 40, "npt": 101})
    model_2.solve_trajectory(alpha=alpha_ref, t_end=t_end)  # ici alpha peut être différent de celui choisi plus haut
    # Décommenter au fur et à mesure des implémentations.
    # model_2.plot_trajectory()
    # model_2.validation()
    # impact = model_2.set_impact_values()
    # model_2.get_impact_values()

plt.show()
print("normal end of execution")
