import numpy as np

import qcdark2.dielectric_pyscf.input_parameters as parmt
from qcdark2.dielectric_pyscf.dielectric_functions import main_setup, main_eps
from qcdark2.dielectric_pyscf.epsilon_routines import save_eps

def get_q_start_stop(N_q, N_nodes):
    if parmt.q_stop is not None:
        N_q = parmt.q_stop
        q_f = parmt.q_stop
    else:
        q_f = N_q

    if parmt.q_start is not None:
        N_q = N_q - parmt.q_start
        q_i = parmt.q_start
    else:
        q_i = 0
    
    q_start = np.array([a[0] for a in np.array_split(np.arange(q_i, q_f), N_nodes)])
    q_stop = np.append(q_start[1:], q_f)
    return q_start, q_stop

if parmt.mpi:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    N_nodes = comm.Get_size()

    if rank == 0: #pyscf calculations and dark_objects setup done on one node
        from qcdark2.dielectric_pyscf.routines import logger
        logger.info(f'MPI = on. Dielectric function calculations for each q will be split among {N_nodes} nodes after initial pyscf DFT calculations.')

        dark_objects = main_setup()

        N_q = len(dark_objects['unique_q'])
        q_start, q_stop = get_q_start_stop(N_q, N_nodes)

        #testing
        print(f'q_start = {q_start}')
        print(f'q_stop = {q_stop}')

        logger.info('\nMPI begins here vvv\n')
    else:
        dark_objects = None
        q_start = np.empty((N_nodes), dtype='int')
        q_stop = np.empty((N_nodes), dtype='int')

    comm.Barrier()

    dark_objects = comm.bcast(dark_objects, root=0) #broadcast dark_objects to all nodes
    comm.Bcast(q_start, root=0)
    comm.Bcast(q_stop, root=0)

    q_start = q_start[rank]
    q_stop = q_stop[rank]

    bin_eps, bin_weights, bin_centers = main_eps(dark_objects, rank=rank, q_start=q_start, q_stop=q_stop)

    #gather Im(eps) and weights from all nodes
    bin_eps_rec = None
    bin_weights_rec = None
    if rank == 0:
        bin_eps_rec = np.empty([i for j in [(N_nodes,), bin_eps.shape] for i in j], dtype='complex')
        bin_weights_rec = np.empty([i for j in [(N_nodes,), bin_weights.shape] for i in j], dtype='float')

    comm.Gather(bin_eps, bin_eps_rec, root=0)
    comm.Gather(bin_weights, bin_weights_rec, root=0)

    #Add together contributions from all nodes and save
    if rank == 0:
        bin_eps_rec = np.sum(bin_eps_rec, axis=0)
        bin_weights_rec = np.sum(bin_weights_rec, axis=0)

        save_eps(bin_eps_rec, bin_weights_rec, bin_centers)

else:
    from qcdark2.dielectric_pyscf.routines import logger
    logger.info('MPI = off. All calculations will be done on one node.')

    dark_objects = main_setup()
    bin_eps, bin_weights, bin_centers = main_eps(dark_objects, rank=None, q_start=parmt.q_start, q_stop=parmt.q_stop)

    save_eps(bin_eps, bin_weights, bin_centers)