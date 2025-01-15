import dielectric_pyscf.utils as utils
from dielectric_pyscf.dielectric_functions import main

utils.check_requirements() # Check requirements
utils.patch() # Patch required for some versions, see function for details
main() #calculate dielectric function