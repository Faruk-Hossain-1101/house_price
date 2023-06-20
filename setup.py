from setuptools import find_packages, setup
from typing import List


HYFEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    print("Something")
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

    if HYFEN_E_DOT in requirements:
        requirements.remove(HYFEN_E_DOT)

    return requirements


setup(
    name='RegressorProject',
    version='0.0.1',
    author='Faruk Hossain',
    author_email='hossainf114@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)