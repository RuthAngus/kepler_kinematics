from setuptools import setup

setup(name='kepler_kinematics',
      version='0.1rc0',
      description='Tools for calculating kinematic ages of Kepler stars',
      url='http://github.com/RuthAngus/kepler_kinematics',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['kepler_kinematics'],
      include_package_data=True,
      install_requires=['numpy', 'tqdm', 'astropy', 'matplotlib'],
      zip_safe=False)
