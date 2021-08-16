# kepler_kinematics
Calculating the velocities of Kepler stars.

To calculate velocities:
----------------------

For individual stars, run through Demo_Notebook.ipynb.
For many stars, follow instructions for running on the cluster below.

* ssh into cluster and navigate to aviary_tests/aviary.
* The file containing vx, vy, vz and distance that you want to use to construct a prior should be in aviary_tests/aviary/aviary and should be called by pym3_
* Create a new run_.py file and a new pymc3_multi_star.py or edit the
    get_velocities_general.py.
* Create a new run_.py file and run_.sh file.
* Change the upper and lower indices in the run file, depending on how many
    stars you want to run on.
* Change the directory and data file in code/get_velocities_general.py
* The data file must contain ra, ra_error, dec, dec_error, parallax,
    parallax_error, pmra, pmra_error, pmdec, pmdec_error.
* Create a new .sh file and do module load slurm, sbatch <.sh file>.
* To watch progress: tail -f slurm-.out
* assemble_all_results.py combines individual .csv results files into one file.
* assemble_results.py also calculates proper motions from the inferred RVs.


kepler_kinematics
======

**velocities.py**:
Some basic functions for calculating velocities from gaia data
using astropy. Includes definitions of Solar velocity constants.

**velocity_pm_conversion.py**:
Some unit functions for coordinate conversions

**pymc3_functions_one_star.py**:
Functions needed to do coordinate transformations
in a pymc3-friendly framework.
Calls functions in velocities.

**inference.py**: Contains the prior and likelihood function for inferring
velocities -- but I think it's for emcee.
Calls functions in pymc3_functions_one_star, velocities and
velocity_pm_conversion.

Other Code
====

**code/data.py**: Code for assembling data catalogs, used to calculate Kepler
velocities and to construct the prior.
This catalog is called by aviary/pymc3_functions_one_star:
gaia_kepler_lamost.csv.

Data
====

**/kepler_kinematics/gaia_kepler_lamost.csv**: This file is created by *code/data.py*.
It is the Gaia-Kepler crossmatch file, combined with LAMOST RVs, and with
velocities directly calculated from RVs, where available.
It is used to construct a prior.

Files used in data.py:

**/data/kepler_dr2_1arcsec.fits**: Megan's crossmatched catalog

**/data/gaia-kepler-lamost_snr.csv**: The raw crossmatched file from the LAMOST website. Includes magnitude S/N ratios.

Notebooks
=========

**Plot_results**: The results.pdf plot in the paper.

**Plot_rv_histogram**: The rv_histogram.pdf plot in the paper.

**Plot_residuals**: The residuals.pdf plot in the paper.

**Plot_prior_comparison**: The prior_comparison.pdf plot in the paper.

**Plot_prior_distributions_2D**: The prior_distributions_2D.pdf plot in the paper.

**Plot_prior_distributions_2D**: The prior_distributions_2D.pdf plot in the paper.

**Plot_kepler_field**: The kepler_field.pdf plot in the paper.

**Plot_CMD**: The CMD.pdf plot in the paper.

**Paper_Plots**: A Companion notebook to the paper, producing plots.

**Demo_Notebook**: A notebook that demonstrates how to calculate stellar velocities. Also saves a figure of a posterior.

**Examining_the_LAMOST_crossmatch**: Looking in detail at the Gaia-LAMOST crossmatch.

**Xmatch_tests.ipynb**: Looking into the APOGEE, LAMOST, Gaia, and Kepler
crossmatches in more detail.

**Lamost_xmatching.ipynb**: Starting from scratch so that I can try to redo the Lamost crossmatch. Crossmatching Gaia and LAMOST by hand!