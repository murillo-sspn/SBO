import numpy as np
from numpy.ma.extras import atleast_2d


class rotor_and_diffuser_blades:

    def __init__(self, IN):

        self.IN = IN

        self.dvs_cfd_baseline = {}
        self.dvs_cfd = {}
        for key, val in self.IN['dvs_cfd_baseline'].items():
            self.dvs_cfd_baseline[key] = val[0]
            self.dvs_cfd[key] = [val[0]]

        self.dvs_doe_baseline = {}
        self.dvs_doe = {}
        self.dvs_doe_lower_bound = {}
        self.dvs_doe_upper_bound = {}
        for key, val in self.IN['dvs_doe_baseline_and_ranges'].items():
            baseval, delta_lower, delta_upper = val[0], val[1], val[2]
            if isinstance(baseval,str):
                self.dvs_doe_baseline[key] = self.dvs_cfd_baseline[baseval]
            else:
                self.dvs_doe_baseline[key] = baseval + 0.0
            self.dvs_doe[key] = [self.dvs_doe_baseline[key]]
            self.dvs_doe_lower_bound[key] = self.dvs_doe_baseline[key] + delta_lower
            self.dvs_doe_upper_bound[key] = self.dvs_doe_baseline[key] + delta_upper

        self.b2 = self.IN['general_info']['b2'][0]
        self.r2 = self.IN['general_info']['r2'][0]


    def add_dvs_doe_to_cfd(self, dvs_doe_i_design):


        for key, val in dvs_doe_i_design.items():
            self.dvs_doe[key].append(val)

        self.dvs_cfd["Rotor.Hub.Theta.3y"].append(dvs_doe_i_design["rotor_hub_midstream_angle"])
        dThetaHub = dvs_doe_i_design["rotor_hub_wrap_perturbation"]
        self.dvs_cfd["Rotor.Hub.Theta.4y"].append(self.dvs_cfd["Rotor.Hub.Theta.4y"][0] + dThetaHub)
        self.dvs_cfd["Rotor.Hub.Theta.5y"].append(self.dvs_cfd["Rotor.Hub.Theta.5y"][0] + dThetaHub)
        self.dvs_cfd["Rotor.Shroud.Theta.3y"].append(dvs_doe_i_design["rotor_shroud_midstream_angle"])
        dThetaRake = float(
            180 / np.pi * self.b2 / self.r2 * np.tan(np.pi / 180 * dvs_doe_i_design["rotor_TE_rake_angle"]))
        dThetaShroud = dThetaHub + dThetaRake
        self.dvs_cfd["Rotor.Shroud.Theta.4y"].append(self.dvs_cfd["Rotor.Shroud.Theta.4y"][0] + dThetaShroud)
        self.dvs_cfd["Rotor.Shroud.Theta.5y"].append(self.dvs_cfd["Rotor.Shroud.Theta.5y"][0] + dThetaShroud)

        self.dvs_cfd["Diffuser.Theta.3y"].append(dvs_doe_i_design["diffuser_midstream_angle"])
        self.dvs_cfd["Diffuser.Theta.4y"].append(
            self.dvs_cfd["Diffuser.Theta.4y"][0] + dvs_doe_i_design["diffuser_wrap_perturbation"])
        self.dvs_cfd["Diffuser.Theta.5y"].append(
            self.dvs_cfd["Diffuser.Theta.5y"][0] + dvs_doe_i_design["diffuser_wrap_perturbation"])

        self.dvs_cfd["Diffuser.Thickness.2y"].append(dvs_doe_i_design["diffuser_tn2"])
        self.dvs_cfd["Diffuser.Thickness.3y"].append(dvs_doe_i_design["diffuser_tn3"])
