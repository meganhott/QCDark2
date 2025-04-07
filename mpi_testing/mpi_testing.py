from mpi4py import MPI
import numpy as np
import pyscf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N_nodes = comm.Get_size()

if rank == 0:
    a = np.arange(10)
    print(f'{N_nodes} total nodes')
    print(f'rank 0: {MPI.Get_processor_name()}, a = {a}')
else:
    a = None

a = comm.bcast(a, root=0)
b = a + rank

print(f'rank {rank}: {MPI.Get_processor_name()}, b = {b}')

#gathering b
b_rec = None
if rank == 0:
    b_rec = np.empty([N_nodes, a.shape[0]], dtype='int')
comm.Gather(b, b_rec, root=0)

if rank == 0:
    print(b_rec)

#comm.Get_attr(MPI.UNIVERSE_SIZE)
#print(comm.Get_size())

