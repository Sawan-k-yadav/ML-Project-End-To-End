# This setup.py will be responsible in creating my machine learning project or application as a package
# And even we can deploy in pypi web site

from setuptools import find_packages, setup
from typing import List



# We are trying to add some logic in a function if we want intall lots of library from requirements.txt
HYPHEN_E_DOT = '-e .'
def get_requirements(find_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(find_path) as file_obj:
        requirements=file_obj.readlines()   # It will read "\n" as change in line as well
        requirements=[req.replace("\n","") for req in requirements] # So are replacing "\n" with blank space

        if HYPHEN_E_DOT in requirements:  # This function logic will remove "-e ." which will come while reading all the library of requirements.txt
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="mlproject",
    version="0.0.1",  # giving version to my package
    author="sawan",
    author_email="sawan.gomia@gmail.com",
    packages=find_packages(), # it will find the package from project folder
    install_requires=get_requirements('requirements.txt')
)