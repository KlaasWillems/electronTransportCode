from mpi4py import MPI
import time
import numpy as np

if __name__ == '__main__':
    nproc = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()

    a = np.ones(shape =(4, ), dtype=float)*rank

            # gather estimator
    if rank == 0:
        recvbuf = np.empty((nproc, 4 ), dtype=float)
    else:
        recvbuf = None  # type: ignore

    time.sleep(nproc - rank)
    MPI.COMM_WORLD.Gather(a, recvbuf, root=0)

    if rank == 0:
        print(recvbuf)
