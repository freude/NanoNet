try:
    from mpi4py import MPI

    mpi_available = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    mpi_available = False
    comm = None
    rank = 0
    size = 1
    MPI = None


print("MPI available: ", mpi_available)
print("The pool size is: ", size)


def set_mpi(mpi_switch: bool):
    global mpi_available
    global comm
    global rank
    global size
    mpi_available = mpi_switch
    print("MPI available: ", mpi_available)
    print("The pool size is: ", size)
    if mpi_available:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        # Fallback for non-MPI run
        comm = None
        rank = 0
        size = 1