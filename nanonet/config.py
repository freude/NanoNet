try:
    from mpi4py import MPI

    mpi_available = True
except ImportError:
    mpi_available = False

print("MPI available: ", mpi_available)

if mpi_available:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
else:
    # Fallback for non-MPI run
    comm = None
    rank = 0
    size = 1


def set_mpi(mpi_switch: bool):
    global mpi_available
    mpi_available = mpi_switch
