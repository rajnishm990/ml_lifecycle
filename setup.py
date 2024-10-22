from setuptools import setup, find_packages
from typing import List

def get_requirements(file:str) -> List[str]:
    requirements = []
    with open(file, 'r') as file_obj:
        requirements = file_obj.read().splitlines()
        
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements


setup(
    name = 'ml_project',
    version = '0.0.1',
    author='Rajnish Mishra',
    author_email='rajnishm990@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

)