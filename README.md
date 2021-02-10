# kepler_kinematics
Calculating the velocities of Kepler stars.

To calculate velocities:
----------------------
*ssh into cluster and navigate to aviary_tests/aviary.
*Create a new run_.py file and a new pymc3_multi_star.py or edit the get_velocities_general.py.
*Change the upper and lower indices in the run file, depending on how many stars you want to run on.
*Change the directory and data file in code/get_velocities_general.py
*The data file must contain ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error.
*Create a new .sh file and do module load slurm, sbatch <.sh file>.
*To watch progress: tail -f slurm-.out

kepler_kinematics
======

*velocities.py*:
Some basic functions for calculating velocities from gaia data
using astropy. Includes definitions of Solar velocity constants.

*velocity_pm_conversion.py*:
Some unit functions for coordinate conversions

*pymc3_functions_one_star.py*:
Functions needed to do coordinate transformations
in a pymc3-friendly framework.
Calls functions in velocities.

*inference.py*: Contains the prior and likelihood function for inferring
velocities.
Calls functions in pymc3_functions_one_star, velocities and
velocity_pm_conversion.

OTHER CODE
====

*code/data.py*: Code for assembling data catalogs, used to calculate Kepler
velocities and to construct the prior.
This catalog is called by aviary/pymc3_functions_one_star:
mc_san_gaia_lam.csv.

DATA
====

Files used in inference.py to construct a prior:
gaia_mc5_velocities.csv

This file is created in ~/projects/old_stuff/aviary///code/calculate_vxyz.py
from gaia_mc5.csv which is created in...

Files used in data.py:

*kepler_dr2_1arcsec.fits*: Megan's crossmatched catalog

*santos.csv*: Rotation periods from Santos et al.

*KeplerRot-LAMOST.csv*: Jason Curtis' LAMOST file.

*Ruth_McQuillan_Masses_Out.csv*: Masses from Travis Berger.

*Table_1_Periodic.txt*: Rotation periods from McQuillan 2014.

NOTEBOOKS
=========
