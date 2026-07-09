import typing as ty
import numpy as np
import numpy.typing as npt
import scipy.linalg as la
import itertools

from fontTools.subset.svg import href_local_target


def _read_hr(iterator, ignore_orbital_order=False):
    r"""
    read the number of wannier functions and the hopping entries
    from *hr.dat and converts them into the right format
    """
    next(iterator)  # skip first line
    num_wann = int(next(iterator))
    nrpts = int(next(iterator))

    # get degeneracy points
    deg_pts = []
    # order in zip important because else the next data element is consumed
    for _, line in zip(range(int(np.ceil(nrpts / 15))), iterator):
        deg_pts.extend(int(x) for x in line.split())
    assert len(deg_pts) == nrpts

    num_wann_square = num_wann ** 2

    def to_entry(line, i):
        """Turns a line (string) into a hop_list entry"""
        entry = line.split()
        orbital_a = int(entry[3]) - 1
        orbital_b = int(entry[4]) - 1
        # test consistency of orbital numbers
        if not ignore_orbital_order:
            if not (orbital_a == i % num_wann) and (
                    orbital_b == (i % num_wann_square) // num_wann
            ):
                raise ValueError(f"Inconsistent orbital numbers in line '{line}'")
        return [
            (float(entry[5]) + 1j * float(entry[6]))
            / (deg_pts[i // num_wann_square]),
            orbital_a,
            orbital_b,
            [int(x) for x in entry[:3]],
        ]

    # skip random empty lines
    lines_nonempty = (l for l in iterator if l.strip())
    hop_list = (to_entry(line, i) for i, line in enumerate(lines_nonempty))

    return num_wann, hop_list


def from_wannier_files(hr_file: str,
        wsvec_file: ty.Optional[str] = None,
        xyz_file: ty.Optional[str] = None,
        win_file: ty.Optional[str] = None,
        h_cutoff: float = 0.0,
        ignore_orbital_order: bool = False,
        pos_kind: str = "wannier",
        distance_ratio_threshold: float = 3.0,
        **kwargs,
):
    """
    Create a :class:`.Model` instance from Wannier90 output files.

    Parameters
    ----------
    hr_file :
        Path of the ``*_hr.dat`` file. Together with the
        ``*_wsvec.dat`` file, this determines the hopping terms.
    wsvec_file :
        Path of the ``*_wsvec.dat`` file. This file determines the
        remapping of hopping terms when ``use_ws_distance`` is used
        in the Wannier90 calculation.
    xyz_file :
        Path of the ``*_centres.xyz`` file. This file is used to
        determine the positions of the orbitals, from the Wannier
        centers given by Wannier90.
    win_file :
        Path of the ``*.win`` file. This file is used to determine
        the unit cell.
    h_cutoff :
        Cutoff value for the hopping strength. Hoppings with a
        smaller absolute value are ignored.
    ignore_orbital_order :
        Do not throw an error when the order of orbitals does not
        match what is expected from the Wannier90 output.
    pos_kind :
        Determines how positions are assinged to orbitals. Valid
        options are `wannier` (use Wannier centres) or
        `nearest_atom` (map to nearest atomic position).
    distance_ratio_threshold :
        [Applies only for pos_kind='nearest_atom']
        The minimum ratio between the second-nearest and nearest
        atom below which an error will be raised.
    kwargs :
        :class:`.Model` keyword arguments.
    """

    if win_file is not None:
        if "uc" in kwargs:
            raise ValueError(
                "Ambiguous unit cell: It can be given either via 'uc' or the 'win_file' keywords, but not both."
            )
        with open(win_file, encoding="utf-8") as f:
            kwargs["uc"] = cls._read_win(f)["unit_cell_cart"]

    if xyz_file is not None:
        if "pos" in kwargs:
            raise ValueError(
                "Ambiguous orbital positions: The positions can be given either via the 'pos' or the 'xyz_file' keywords, but not both."
            )
        if "uc" not in kwargs:
            raise ValueError(
                "Positions cannot be read from .xyz file without unit cell given: Transformation from cartesian to reduced coordinates not possible. Specify the unit cell using one of the keywords 'uc' or 'win_file'."
            )
        with open(xyz_file, encoding="utf-8") as f:
            wannier_pos_list_cartesian, atom_list_cartesian = cls._read_xyz(f)
            wannier_pos_cartesian = np.array(wannier_pos_list_cartesian)
            atom_pos_cartesian = np.array([a.pos for a in atom_list_cartesian])
            if pos_kind == "wannier":
                pos_cartesian: ty.Union[
                    ty.List[npt.NDArray[np.float_]], npt.NDArray[np.float_]
                ] = wannier_pos_cartesian
            elif pos_kind == "nearest_atom":
                if distance_ratio_threshold < 1:
                    raise ValueError(
                        "Invalid value for 'distance_ratio_threshold': must be >= 1."
                    )
                pos_cartesian = ty.cast(ty.List[npt.NDArray[np.float_]], [])
                for p in wannier_pos_cartesian:
                    p_reduced = la.solve(kwargs["uc"].T, np.array(p).T).T
                    T_base = np.floor(p_reduced)
                    all_atom_pos = np.array(
                        [
                            kwargs["uc"].T @ (T_base + T_shift) + atom_pos
                            for atom_pos in atom_pos_cartesian
                            for T_shift in itertools.product([-1, 0, 1], repeat=3)
                        ]
                    )
                    distances = la.norm(p - all_atom_pos, axis=-1)
                    idx = np.argpartition(distances, 2)[:2]
                    nearest, second_nearest = distances[idx]
                    if second_nearest / nearest < distance_ratio_threshold:
                        raise ValueError("Oops")
                    pos_cartesian.append(all_atom_pos[idx[0]])
            else:
                raise ValueError(
                    "Invalid value '{}' for 'pos_kind', must be 'wannier' or 'nearest_atom'".format(
                        pos_kind
                    )
                )
            kwargs["pos"] = la.solve(kwargs["uc"].T, np.array(pos_cartesian).T).T

    with open(hr_file, encoding="utf-8") as f:
        num_wann, hop_entries = _read_hr(
            f, ignore_orbital_order=ignore_orbital_order
        )
        hop_entries = (hop for hop in hop_entries if abs(hop[0]) > h_cutoff)

        if wsvec_file is not None:
            with open(wsvec_file, encoding="utf-8") as f:
                wsvec_generator = cls._async_parse(
                    cls._read_wsvec(f), chunksize=num_wann
                )

                def remap_hoppings(hop_entries):
                    for t, orbital_1, orbital_2, R in hop_entries:
                        # Step _async_parse to where it accepts
                        # a new key.
                        # The _async_parse does not raise StopIteration
                        next(  # pylint: disable=stop-iteration-return
                            wsvec_generator
                        )
                        T_list = wsvec_generator.send(
                            (orbital_1, orbital_2, tuple(R))
                        )
                        N = len(T_list)
                        for T in T_list:
                            # not using numpy here increases performance
                            yield (
                                t / N,
                                orbital_1,
                                orbital_2,
                                tuple(r + t for r, t in zip(R, T)),
                            )

                hop_entries = remap_hoppings(hop_entries)
                return cls.from_hop_list(
                    size=num_wann, hop_list=hop_entries, **kwargs
                )

        return cls.from_hop_list(size=num_wann, hop_list=hop_entries, **kwargs)


if __name__ == "__main__":

    hr_file = "/Users/mykhailoklymenko/Monash_work/data/graphene_bilayer/wannier/ab_bilayer_scf_hr.dat"

    with open(hr_file, encoding="utf-8") as f:
        num_wann, hop_entries = _read_hr(f, ignore_orbital_order=False)

        # hop_entries = (hop for hop in hop_entries if abs(hop[0]) > 0.1)
        hop_entries = list(hop for hop in hop_entries if abs(hop[0]) > 0.0)


    from pprint import pprint
    pprint(hop_entries)