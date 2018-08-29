import os
import tb

"""If you change 'SiNW2.xyz' in 'path=' to some other file name in the directory 
   the function will look for that file. If it's there it will try and compute the 
   hamiltonian. If the file is not there it will print error"""

path = 'c:\users\sammy\desktop\NanoNet\input_samples\SiNW2.xyz'
if os.path.isfile(path):
    for xyz_file in path:
        hamiltonian = tb.Hamiltonian(xyz=os.path.join(path, xyz_file), nn_distance=2.4)
        hamiltonian.initialize()
        #plt.imshow(np.log(np.abs(hamiltonian.h_matrix)))
        #plt.savefig('Hamiltonian.pdf')
        a_si = 5.50
        PRIMITIVE_CELL = [[0, 0, a_si]]
        hamiltonian.set_periodic_bc(PRIMITIVE_CELL)
       #plt.show() - commented out so hamiltonian matrix is not plotted
else:
    print ('Error - file is not in the directory')

