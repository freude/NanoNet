"""
The module contains sets of tight-binding parameters.
"""

#  NN - Si-Si
PARAMS_SI_SI = {'ss_sigma': -1.9413,
                '1s1s_sigma': -3.3081,
                's1s_sigma': -1.6933,
                'sp_sigma': 2.7836,
                '1sp_sigma': 2.8428,
                'sd_sigma': -2.7998,
                '1sd_sigma': -0.7003,
                'pp_sigma': 4.1068,
                'pp_pi': -1.5934,
                'pd_sigma': -2.1073,
                'pd_pi': 1.9977,
                'dd_sigma': -1.2327,
                'dd_pi': 2.5145,
                'dd_delta': -2.4734}


#  NN - Si-H
PARAMS_H_SI = {'ss_sigma': -3.9997,
               's1s_sigma': -1.6977,
               'sp_sigma': 4.2518,
               'sd_sigma': -2.1055}

PARAMS_H_H = {'ss_sigma': 1}


# 1NN - Bi-Bi
PARAMS_BI_BI1 = {'ss_sigma': -0.608,
                 'sp_sigma': 1.320,
                 'pp_sigma': 1.854,
                 'pp_pi': -0.600}

# 2NN - Bi-Bi
PARAMS_BI_BI2 = {'ss_sigma': -0.384,
                 'sp_sigma': 0.433,
                 'pp_sigma': 1.396,
                 'pp_pi': -0.344}

# 3NN - Bi-Bi
PARAMS_BI_BI3 = {'ss_sigma': 0,
                 'sp_sigma': 0,
                 'pp_sigma': 0.156,
                 'pp_pi': 0}
