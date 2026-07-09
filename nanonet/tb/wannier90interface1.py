import re
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

__all__ = [
    "HRBlock",
    "W90Dataset",
    "read_win",
    "parse_unit_cell_cart",
    "read_centres",
    "read_hr",
    "read_kpoint_path",
    "load_w90_dataset",
    "read_bands_w90",
]


class W90ParseError(RuntimeError):
    """Raised when a Wannier90 file is missing or cannot be parsed."""


class W90ConsistencyError(RuntimeError):
    """Raised when Wannier90 data are internally inconsistent."""


@dataclass(frozen=True)
class HRBlock:
    """
    Dataclass representing real-space Hamiltonian block :math:`H(R)` with degeneracy.

    Attributes
    ----------
    h : numpy.ndarray
        Complex matrix ``(num_wan, num_wan)`` containing the tight-binding
        amplitudes for lattice vector ``R``.
    degeneracy : int
        Wigner-Seitz multiplicity associated with the shell of ``R``.
    """

    h: np.ndarray
    degeneracy: int


@dataclass(frozen=True)
class W90Dataset:
    """Dataclass for Wannier90 data.

    Attributes
    ----------
    prefix : str
        Wannier90 run prefix.
    root : pathlib.Path
        Directory containing the output files.
    lat_cart : numpy.ndarray
        Cartesian lattice vectors with shape ``(3, 3)`` in angstroms.
    centres_xyz : numpy.ndarray
        Wannier centres in Cartesian coordinates ``(num_wan, 3)``.
    centres_red : numpy.ndarray
        Wannier centres in reduced coordinates ``(num_wan, 3)``.
    num_wan : int
        Number of Wannier functions in the dataset.
    ham_r : dict[tuple[int, int, int], HRBlock]
        Mapping from lattice vectors ``R`` to their Hamiltonian blocks.
    kpath_nodes_red : numpy.ndarray | None
        Reduced coordinates of the ``kpoint_path`` nodes, if present.
    kpath_labels : list[str] | None
        Labels corresponding to ``kpath_nodes_red``.
    bands_k_red : numpy.ndarray | None
        Reduced k-points from Wannier90 band interpolation.
    bands_ene_ev : numpy.ndarray | None
        Interpolated band energies (eV) matching ``bands_k_red``.
    meta : dict | None
        Additional metadata such as spreads or window definitions.
    win_lines : list[str] | None
        Raw lines from ``prefix.win`` when requested by the loader.
    """

    prefix: str
    root: Path
    lat_cart: np.ndarray  # (3,3) Angstrom
    centres_xyz: np.ndarray  # (num_wan,3) Angstrom
    centres_red: np.ndarray  # (num_wan,3) reduced
    num_wan: int
    ham_r: Dict[Tuple[int, int, int], HRBlock]
    # optional extras
    kpath_nodes_red: Optional[np.ndarray] = None
    kpath_labels: Optional[List[str]] = None
    bands_k_red: Optional[np.ndarray] = None
    bands_ene_ev: Optional[np.ndarray] = None
    meta: Optional[dict] = None  # spreads, windows, etc.
    win_lines: Optional[List[str]] = None


# ---------- low-level readers ----------


def _read_text(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()
    except FileNotFoundError as e:
        raise W90ParseError(f"Missing file: {path}") from e


def _extract_block(lines: List[str], name: str) -> List[str]:
    begin, end = f"begin {name}".lower(), f"end {name}".lower()
    in_block, out = False, []
    for raw in lines:
        s = raw.strip()
        t = s.lower()
        if not in_block and t.startswith(begin):
            in_block = True
            continue
        if in_block:
            if t.startswith(end):
                break
            out.append(s.replace(",", " "))  # tolerate commas
    return out


def read_win(root: Path, prefix: str) -> List[str]:
    """Return the raw lines from ``prefix.win``."""
    return _read_text((root / f"{prefix}.win").expanduser())


def parse_unit_cell_cart(win_lines: List[str]) -> np.ndarray:
    """Parse ``unit_cell_cart`` into a Cartesian (3×3) lattice matrix."""
    block = _extract_block(win_lines, "unit_cell_cart")
    if not block or len(block) < 3:
        raise W90ParseError("unit_cell_cart block missing or too short.")
    scale = 1.0
    if block[0].lower() in {"bohr", "ang", "angstrom"}:
        if block[0].lower() == "bohr":
            scale = BOHRTOANG
        block = block[1:]
    lat = np.zeros((3, 3), float)
    for i in range(3):
        parts = block[i].split()
        if len(parts) < 3:
            raise W90ParseError("unit_cell_cart rows need 3 components.")
        lat[i] = [float(parts[j]) * scale for j in range(3)]
    return lat


def read_centres(root: Path, prefix: str, num_wan: int) -> np.ndarray:
    """Read ``prefix_centres.xyz`` and return Wannier centres in Cartesian coords."""
    lines = _read_text(root / f"{prefix}_centres.xyz")
    start = 2
    coords = []
    for idx in range(num_wan):
        try:
            tag, x, y, z, *_ = lines[start + idx].split()
        except Exception as e:
            raise W90ParseError("Centres file shorter than expected.") from e
        if tag != "X":
            raise W90ParseError("Centres file format error (expected 'X').")
        coords.append([float(x), float(y), float(z)])
    return np.asarray(coords, float)


def _cart_to_red(a1, a2, a3, xyz):
    # Here a1..a3 are direct lattice vectors; reduced = xyz @ inv(lat)
    Lat = np.vstack([a1, a2, a3])
    return np.asarray(xyz) @ np.linalg.inv(Lat)


def read_hr(root: Path, prefix: str) -> Tuple[int, Dict[Tuple[int, int, int], HRBlock]]:
    """Read ``prefix_hr.dat`` returning ``(num_wan, {R: HRBlock})``.

    Parameters
    ----------
    root : Path or str
        Directory containing the Wannier90 hr file.
    prefix : str
        Prefix used in the Wannier90 hr file name: "{prefix}_hr.dat".

    Returns
    -------
    num_wan : int
        Number of Wannier functions.
    ham_r : dict
        Mapping from lattice vector triplet R (tuple of ints) to :class:`HRBlock`.
    """
    p = root / f"{prefix}_hr.dat"
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        _ = fh.readline()
        try:
            num_wan = int(fh.readline())
            num_ws = int(fh.readline())
        except Exception as e:
            raise W90ParseError("Cannot read num_wan/num_ws in _hr.dat") from e
        # degeneracies (can span multiple lines)
        deg = []
        while len(deg) < num_ws:
            line = fh.readline()
            if not line:
                raise W90ParseError("Unexpected EOF while reading degeneracies.")
            deg.extend(int(x) for x in line.split())
        deg = np.asarray(deg[:num_ws], int)
        # remainder numeric table
        data = np.loadtxt(fh)  # shape (N,7)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 7:
        raise W90ParseError("_hr.dat must have 7 columns.")
    R = data[:, :3].astype(int)  # Triplets (R1, R2, R3)
    i = data[:, 3].astype(int) - 1  # Wannier function index i
    j = data[:, 4].astype(int) - 1  # Wannier function index j
    v = data[:, 5] + 1j * data[:, 6]  # Hamiltonian matrix element H_{ij}(R)
    # unique shells in encounter order
    _, first_idx, inv = np.unique(R, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(order.size)
    inv = remap[inv]
    unique_R = R[first_idx[order]]
    if deg.size < unique_R.shape[0]:
        raise W90ConsistencyError("Degeneracy list shorter than number of shells.")
    blocks = np.zeros((unique_R.shape[0], num_wan, num_wan), complex)
    np.add.at(blocks, (inv, i, j), v)
    ham_r = {
        tuple(map(int, unique_R[k])): HRBlock(h=blocks[k], degeneracy=int(deg[k]))
        for k in range(unique_R.shape[0])
    }
    return num_wan, ham_r


_KPOINT_LABEL_PATTERN = re.compile(r"^(?P<base>[^\d]+?)(?P<suffix>\d+)?$", re.UNICODE)


def _format_k_label(label: str) -> str:
    special = {
        "g": r"\Gamma",
        "gamma": r"\Gamma",
        "Γ": r"\Gamma",
        "delta": r"\Delta",
        "Δ": r"\Delta",
        "theta": r"\Theta",
        "Θ": r"\Theta",
        "lambda": r"\Lambda",
        "λ": r"\Lambda",
        "xi": r"\Xi",
        "ξ": r"\Xi",
        "pi": r"\Pi",
        "π": r"\Pi",
        "sigma": r"\Sigma",
        "σ": r"\Sigma",
        "upsilon": r"\Upsilon",
        "υ": r"\Upsilon",
        "phi": r"\Phi",
        "ϕ": r"\Phi",
        "psi": r"\Psi",
        "ψ": r"\Psi",
        "omega": r"\Omega",
        "ω": r"\Omega",
    }
    raw = label.strip()
    if not raw:
        return "$$"
    m = _KPOINT_LABEL_PATTERN.match(raw)
    base, suf = (m.group("base"), m.group("suffix")) if m else (raw, None)
    key = base.lower()
    latex = special.get(key) or (
        base if (len(base) == 1 and base.isalpha()) else rf"\mathrm{{{base}}}"
    )
    return rf"${latex}_{{{suf}}}$" if suf else rf"${latex}$"


def read_kpoint_path(win_lines: List[str], *, latex=True):
    """
    Return the reduced-coordinate nodes declared in the ``kpoint_path`` block.

    Parameters
    ----------
    latex : bool, optional
        When True (default) convert labels into LaTeX-friendly strings,
        e.g. ``"G" -> r"$\\Gamma$"``.

    Returns
    -------
    coords : numpy.ndarray
        Array with shape ``(n_nodes, 3)`` containing the reduced coordinates.
    labels : list[str]
        Labels for each node, optionally formatted for LaTeX rendering.
    """
    block = _extract_block(win_lines, "kpoint_path")
    if not block:
        return None, None
    nodes, labels = [], []
    last = None
    for line in block:
        toks = line.split()
        if not toks:
            continue
        if len(toks) % 4:
            raise W90ParseError("kpoint_path entries must be label + 3 coords.")
        for o in range(0, len(toks), 4):
            lbl = toks[o]
            coord = np.array(
                [float(toks[o + 1]), float(toks[o + 2]), float(toks[o + 3])]
            )
            if last is not None and np.allclose(coord, last[1]) and lbl == last[0]:
                continue
            nodes.append(coord)
            labels.append(_format_k_label(lbl) if latex else lbl)
            last = (lbl, coord)
    return np.vstack(nodes), labels


# convenience: assemble dataset
def load_w90_dataset(
    root: Path | str,
    prefix: str,
    *,
    include_bands: bool = True,
    include_win_lines: bool = False,
) -> W90Dataset:
    """Gather lattice, centre, and Hamiltonian data into a :class:`W90Dataset`.

    Parameters
    ----------
    root : Path or str
        Directory containing the Wannier90 files.
    prefix : str
        Prefix used in the Wannier90 file names.

    Returns
    -------
    dataset : W90Dataset
        Container with all relevant data from the Wannier90 output files.
    """
    root = Path(root).expanduser()
    win = read_win(root, prefix)
    lat = parse_unit_cell_cart(win)
    num_wan, ham_r = read_hr(root, prefix)
    centres_xyz = read_centres(root, prefix, num_wan)
    centres_red = _cart_to_red(lat[0], lat[1], lat[2], centres_xyz)
    k_nodes, k_labels = read_kpoint_path(win, latex=True)
    # bands are optional
    bands_k, bands_ene = None, None
    if include_bands:
        try:
            bands_k, bands_ene = read_bands_w90(root, prefix, num_wan)
        except Exception:
            pass
    win_lines = win if include_win_lines else None
    return W90Dataset(
        prefix=prefix,
        root=root,
        lat_cart=lat,
        centres_xyz=centres_xyz,
        centres_red=centres_red,
        num_wan=num_wan,
        ham_r=ham_r,
        kpath_nodes_red=k_nodes,
        kpath_labels=k_labels,
        bands_k_red=bands_k,
        bands_ene_ev=bands_ene,
        meta={},
        win_lines=win_lines,
    )


def read_bands_w90(
    root: Path | str, prefix: str, num_wan: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Wannier90-interpolated band structure.

    Parameters
    ----------
    root : Path or str
        Directory containing the Wannier90 bands files.
    prefix : str
        Prefix used in the Wannier90 bands file names:
        "{prefix}_band.kpt" and "{prefix}_band.dat".
    num_wan : int
        Number of Wannier functions / bands expected.

    Returns
    -------
    kpts_red : (N_k, 3) reduced k-points
    energies_ev : (N_k, num_wan) energies in eV
    """
    root = Path(root).expanduser()
    kpts_path = root / f"{prefix}_band.kpt"
    ene_path = root / f"{prefix}_band.dat"

    if not kpts_path.exists() or not ene_path.exists():
        raise W90ParseError(f"Missing W90 bands files: {kpts_path} or {ene_path}")

    kpts_red = np.loadtxt(kpts_path, skiprows=1)[:, :3]
    ene_raw = np.loadtxt(ene_path)
    if ene_raw.ndim == 1:
        ene_raw = ene_raw[None, :]
    # column 0 is k-index, column 1 is energy; reshape like W90 writes it
    try:
        energies_ev = ene_raw[:, 1].reshape((num_wan, kpts_red.shape[0])).T
    except ValueError as e:
        raise W90ParseError(
            f"Cannot reshape bands: expected {num_wan} bands; "
            f"got {ene_raw.shape} rows for {kpts_red.shape[0]} k-points"
        ) from e
    return kpts_red, energies_ev


class W90:
    r"""Interface to Wannier90

    `Wannier90 <http://www.wannier.org>`_ is a post-processing tool
    that takes as input Bloch wavefunctions and energies generated
    by first-principles electronic structure codes such as
    Quantum-Espresso (PWscf), ABINIT, SIESTA, FLEUR,
    WIEN2k, or VASP. It produces maximally localized Wannier functions
    together with a tight-binding Hamiltonian in the Wannier basis [1]_.

    This class imports tight-binding model parameters from a
    Wannier90 calculation and makes them available in PythTB.
    Upon construction, it reads the relevant Wannier90 output
    files from the specified directory. Use :meth:`model` to
    convert the imported data into a :class:`TBModel` instance.

    The PythTB interface uses the following Wannier90 output files:

    - ``prefix.win``
    - ``prefix_hr.dat``
    - ``prefix_centres.xyz``
    - ``prefix_band.kpt`` (optional)
    - ``prefix_band.dat`` (optional)

    The ``prefix.win`` file provides general input to Wannier90 and is here
    primarily to obtain the lattice vectors.

    To ensure the required files ``prefix_hr.dat`` and ``prefix_centres.xyz``
    are written, include the following flags in the ``prefix.win`` file::

       write_hr = True
       write_xyz = True
       translate_home_cell = False

    These directives instruct Wannier90 to output (i) the real-space
    tight-binding Hamiltonian in ``prefix_hr.dat`` and (ii) the centers of
    the Wannier functions in ``prefix_centres.xyz`` without translating
    them to the home unit cell.

    The optional files ``prefix_band.kpt`` and ``prefix_band.dat``
    can be used to import the Wannier-interpolated band structures
    computed by Wannier90. Please see documentation of function
    :meth:`bands_w90` for more detail.

    Parameters
    ----------
    path : str
        Relative path to the folder that contains Wannier90
        files. These are ``prefix.win``, ``prefix_hr.dat``,
        ``prefix_centres.xyz`` and optionally ``prefix_band.kpt`` and
        ``prefix_band.dat``.

    prefix : str
        This is the prefix used by `Wannier90` code.
        Typically the input to the `Wannier90` code is name ``prefix.win``.

    See Also
    --------
    :ref:`w90-nb`

    Notes
    -----
    Units used throught this interface with Wannier90 are
    electron-volts (eV) and Angstroms.

    .. warning::
        This class has been tested on Wannier90 v3.1.0. Compatibility
        with other versions is not guaranteed.

    .. warning::
        The user needs to make sure that the Wannier functions
        computed using Wannier90 code are well localized. Otherwise the
        tight-binding model may not accurately interpolate the band
        structure. To ensure that the Wannier functions are well
        localized it is often enough to check that the total spread at
        the beginning of the minimization procedure (first total spread
        printed in .wout file) is not more than 20% larger than the
        total spread at the end of the minimization procedure. If those
        spreads differ by much more than 20% user needs to specify
        better initial projection functions.

    .. warning::
        The interpolation is only exact within the frozen energy window
        of the disentanglement procedure.

    .. warning::
        So far PythTB assumes that the position operator is
        diagonal in the tight-binding basis. This is discussed in the
        :download:`notes on tight-binding formalism
        </_static/formalism/pythtb-formalism.pdf>` in Eq. 2.7.,
        :math:`\langle\phi_{{\bf R} i} \vert {\bf r} \vert \phi_{{\bf
        R}' j} \rangle = ({\bf R} + {\bf t}_j) \delta_{{\bf R} {\bf R}'}
        \delta_{ij}`. However, this relation does not hold for Wannier
        functions! Therefore, if you use tight-binding model derived
        from this class in computing Berry-like objects that involve
        position operator such as Berry phase or Berry flux, you would
        not get the same result as if you computed those objects
        directly from the first-principles code! Nevertheless, this
        approximation does not affect other properties such as band
        structure dispersion.


    Examples
    --------
    Read Wannier90 from folder called *example_a*
    This assumes that that folder contains files "silicon.win" (and so on)

    >>> silicon = w90("example_a", "silicon")

    References
    ----------
    .. [1] "Wannier90 as a community code: new features and applications",
            G. Pizzi et al., J. Phys. Cond. Matt. 32,  165902 (2020).
    """

    def __init__(self, path, prefix):
        self.folder = Path(path).expanduser()
        if not self.folder.exists():
            raise FileNotFoundError(f"Wannier90 folder not found: {self.folder}")
        self.path = str(self.folder)
        self.prefix = prefix

        ds = load_w90_dataset(
            self.path, self.prefix, include_bands=False, include_win_lines=True
        )

        # Raw win lines are needed for optional k-path labels in bands_w90.
        self._win_lines = ds.win_lines if ds.win_lines is not None else []

        # adopt dataset fields (preserve your attribute names)
        self.lat = ds.lat_cart
        self.num_wan = ds.num_wan
        self.ham_r = {
            R: {"h": blk.h, "deg": blk.degeneracy} for R, blk in ds.ham_r.items()
        }
        self.xyz_cen = ds.centres_xyz
        self.red_cen = ds.centres_red
        self.lattice = Lattice(self.lat, self.red_cen, periodic_dirs=...)
        self._kpath_nodes_red = ds.kpath_nodes_red
        self._kpath_labels = ds.kpath_labels

        # check if for every non-zero R there is also -R
        self._validate_hr_symmetry()

        # caches (filled lazily)
        self._vecR_cache = {}
        self._dist_cache = {}

    def _validate_hr_symmetry(self):
        R_set = set(self.ham_r.keys())
        for R in R_set:
            if R != (0, 0, 0) and (-R[0], -R[1], -R[2]) not in R_set:
                raise ValueError(f"Did not find negative R for R = {R}!")

    @staticmethod
    def _wrap01(x: np.ndarray) -> np.ndarray:
        out = np.mod(x, 1.0)
        # snap 1.0 → 0.0 to avoid 2π glitches
        out[np.isclose(out, 1.0, atol=1e-12)] = 0.0
        return out

    def _get_vecR(self, R):
        """Cartesian vector for reduced lattice vector R, cached."""
        if not hasattr(self, "_vecR_cache"):
            self._vecR_cache = {}
        if R in self._vecR_cache:
            return self._vecR_cache[R]
        vecR = _red_to_cart((self.lat[0], self.lat[1], self.lat[2]), [R])[0]
        self._vecR_cache[R] = vecR
        return vecR

    def _get_dist_matrix(self, R):
        """Distance for reduced lattice vector R, cached."""
        if not hasattr(self, "_dist_cache"):
            self._dist_cache = {}
        if R in self._dist_cache:
            return self._dist_cache[R]
        vecR = self._get_vecR(R)
        delta = (-self.xyz_cen[:, None, :] + self.xyz_cen[None, :, :]) + vecR[
            None, None, :
        ]
        dist = np.linalg.norm(delta, axis=2)  # (num_wan, num_wan)
        self._dist_cache[R] = dist
        return dist

    def _precompute_distances(self):
        """Precompute distance matrices for all reduced lattice vectors."""
        if not hasattr(self, "_dist_cache"):
            self._dist_cache = {}
        for R in self.ham_r.keys():
            if R not in self._dist_cache:
                self._get_dist_matrix(R)

    def model(
        self,
        zero_energy=0.0,
        min_hopping_norm=None,
        max_distance=None,
        ignorable_imaginary_part=None,
        *,
        onsite_imag_tol: float = 1e-9,
    ):
        r"""Get TBModel associated with this Wannier90 calculation.

        This function returns :class:`pythtb.TBModel` object that can
        be used to interpolate the band structure at arbitrary
        k-point, analyze the wavefunction character, etc.

        The tight-binding basis orbitals in the returned object are
        maximally localized Wannier functions as computed by
        Wannier90. Locations of the orbitals in the returned
        :class:`pythtb.TBModel` object are the centers of
        the Wannier functions computed by Wannier90.

        Parameters
        ----------

        zero_energy : float
            Sets the zero of the energy in the band structure.
            This value is typically set to the Fermi level
            computed by the density-functional code (or to the top of the valence band).
            Units are electron-volts.

        min_hopping_norm : float
            Hopping terms read from Wannier90 with complex norm less than
            *min_hopping_norm* will not be included in the returned
            tight-binding model. This parameters is specified in
            electron-volts. By default all terms regardless of their
            norm are included.

        max_distance : float
            Hopping terms from site *i* to site *j+R* will be ignored if
            the distance from orbital *i* to *j+R* is larger than
            *max_distance*. This parameter is given in Angstroms.
            By default all terms regardless of the distance are included.

        ignorable_imaginary_part : float
            The hopping term will be assumed to be exactly real if the
            absolute value of the imaginary part as computed by Wannier90
            is less than *ignorable_imaginary_part*. By default imaginary
            terms are not ignored. Units are again eV.

        Returns
        -------
        tb : :class:`pythtb.TBModel`
            The :class:`pythtb.TBModel` that can be used to
            interpolate Wannier90 band structure to an arbitrary k-point as well
            as to analyze the character of the wavefunctions.

        Notes
        -----
        - The character of the maximally localized Wannier functions is
          not exactly the same as that specified by the initial
          projections. The orbital character of the Wannier functions can be
          inferred either from the *projections* block in the *prefix*.win or
          from the *prefix*.nnkp file.

        - One way to ensure that the Wannier functions are as close to
          the initial projections as possible is to first choose a good set
          of initial projections (for these initial and final spread should
          not differ more than 20%) and then perform another Wannier90 run
          setting *num_iter=0* in the *prefix*.win file.

        - The tight-binding model returned by this function is only as good as
          the input from Wannier90. In particular, the choice of initial
          projections can have a significant impact on the quality of the
          resulting Wannier functions. It is recommended to experiment with
          different sets of initial projections and to carefully analyze the
          resulting Wannier functions to ensure that they are physically
          meaningful.

        - The number of spin components is always set to 1, even if the
          underlying DFT calculation includes spin.  Please refer to the
          *projections* block or the *prefix*.nnkp file to see which
          orbitals correspond to which spin.

        Examples
        --------
        Get `TBModel` with all hopping parameters

        >>> my_model = silicon.model()

        Simplified model that contains only hopping terms above 0.01 eV

        >>> my_model_simple = silicon.model(min_hopping_norm=0.01)
        >>> my_model_simple.display()

        """
        tb = TBModel(self.lattice)  # initialize the model object

        # remember that this model was computed from w90
        tb._from_w90 = True
        tb._assume_position_operator_diagonal = False

        # Onsites
        hr0 = self.ham_r[(0, 0, 0)]
        deg0 = float(hr0["deg"])
        # Divide by degeneracy only once and assert onsite is (numerically) real
        diag = np.diag(hr0["h"]) / deg0
        # sanity check: imaginary part should be tiny
        if np.max(np.abs(diag.imag)) > onsite_imag_tol:
            raise ValueError(f"Onsite terms should be real (|Im|>{onsite_imag_tol})")
        tb.set_onsite(diag.real - zero_energy)

        # Hoppings

        # Helper to decide if we should process an R (to avoid double counting)
        def _use_R(R):
            r1, r2, r3 = R
            if r1 != 0:
                return r1 > 0
            if r2 != 0:
                return r2 > 0
            return r3 > 0

        if max_distance is not None and not self._dist_cache:
            self._precompute_distances()

        amps_all, ii_all, jj_all, R_all = [], [], [], []
        num_wan = self.num_wan

        for R, blk in self.ham_r.items():
            # Onsite block already handled; keep only off-diagonal pairs here.
            if R == (0, 0, 0):
                use_this_R = True
            else:
                use_this_R = _use_R(R)
            if not use_this_R:
                continue

            # Start from allowed entries and avoid double counting
            if R == (0, 0, 0):
                keep = np.zeros((num_wan, num_wan), dtype=bool)
                iu = np.triu_indices(num_wan, k=1)
                keep[iu] = True
            else:
                keep = np.ones((num_wan, num_wan), dtype=bool)

            Hr = blk["h"]
            deg = float(blk["deg"])  # scalar

            # Divide by degeneracy once per block
            inv_deg = 1.0 / deg  # multiplying inverse is faster than dividing
            H = Hr * inv_deg  # (num_wan, num_wan)

            # Distance cutoff (use cached distances; compute lazily if needed)
            if max_distance is not None:
                dist = self._get_dist_matrix(R)
                keep &= dist <= max_distance
                if not np.any(keep):
                    continue

            # Apply min_hopping_norm filter
            if min_hopping_norm is not None:
                keep &= np.abs(H) >= min_hopping_norm
                if not np.any(keep):
                    continue

            # Optionally zero-out tiny imaginary parts before insertion
            if ignorable_imaginary_part is not None:
                sel = keep & (np.abs(H.imag) < ignorable_imaginary_part)
                if np.any(sel):
                    H = H.copy()
                    H.imag[sel] = 0.0

            ii, jj = np.nonzero(keep)
            if ii.size:
                amps = H[ii, jj]
                R_arr = np.repeat(np.array(R, dtype=int)[None, :], ii.size, axis=0)
                amps_all.append(amps)
                ii_all.append(ii)
                jj_all.append(jj)
                R_all.append(R_arr)

        if amps_all:
            amps = np.concatenate(amps_all)
            ii = np.concatenate(ii_all)
            jj = np.concatenate(jj_all)
            R_arr = np.concatenate(R_all)
            tb._append_hops(amps, ii, jj, R_arr)

        return tb

    def dist_hop(self):
        r"""Get distances and hopping terms of Hamiltonian in Wannier basis.

        This function returns all hopping terms (from orbital *i* to
        *j+R*) as well as the distances between the *i* and *j+R*
        orbitals. For well localized Wannier functions hopping term
        should decay exponentially with distance.

        Returns
        -------
        dist : np.ndarray
            Distances between Wannier function centers (*i* and *j+R*) in Angstroms.

        ham : np.ndarray
            Corresponding hopping terms in eV.

        Notes
        -----
        This function can be used to help determine the *min_hopping_norm*
        and *max_distance* parameters in the :func:`pythtb.w90.model` function
        call.

        Examples
        --------
        Get distances and hopping terms

        >>> (dist, ham) = silicon.dist_hop()

        Plot logarithm of the hopping term as a function of distance

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.scatter(dist, np.log(np.abs(ham)))
        >>> fig.savefig("localization.pdf")

        """

        ret_ham = []
        ret_dist = []
        num_wan = self.num_wan

        for R, blk in self.ham_r.items():
            Ham = blk["h"] / float(blk["deg"])
            dist = self._get_dist_matrix(R)  # (num_wan, num_wan)
            keep = np.ones((num_wan, num_wan), dtype=bool)

            if R == (0, 0, 0):
                np.fill_diagonal(keep, False)  # avoid diagonal terms

            ret_ham.append(Ham[keep])
            ret_dist.append(dist[keep])

        return (np.concatenate(ret_dist), np.concatenate(ret_ham))

    def shells(self, num_digits=2):
        r"""Get all shells of distances between Wannier function centers.

        This is one of the diagnostic tools that can be used to help
        in determining *max_distance* parameter in
        :func:`pythtb.w90.model` function call.

        Parameters
        ----------
        num_digits : int
            Distances will be rounded up to these many digits. Default value is 2.

        Returns
        -------
        shells : list
            All distances between all Wannier function centers (*i* and *j+R*) in Angstroms.

        Examples
        --------
        Print all shells

        >>> print(silicon.shells())
        """

        shells = []
        for R in self.ham_r.keys():
            dist = self._get_dist_matrix(R)
            shells.extend(np.round(dist.ravel(), num_digits).tolist())

        # remove duplicates and sort
        shells = np.sort(list(set(shells)))

        return shells

    def bands_w90(
        self,
        return_k_cart: bool = False,
        return_k_dist: bool = False,
        return_k_nodes: bool = False,
    ):
        r"""Read interpolated band structure from Wannier90 output files.

        The purpose of this function is to compare the interpolation
        in Wannier90 with that in PythTB. This function reads in the band
        structure as interpolated by Wannier90.

        The code assumes that the following files were generated by
        Wannier90,

        - ``prefix_band.kpt``
        - ``prefix_band.dat``

        These files will be generated only if the *prefix*.win file
        contains the *kpoint_path* block.

        Parameters
        ----------
        return_k_cart : bool, optional
            If True, also return k-points in Cartesian coordinates.

            .. versionadded:: 2.0.0

        return_k_dist : bool, optional
            If True, also return the cumulative k-path distance (in inverse Angstroms).

            .. versionadded:: 2.0.0

        return_k_nodes : bool, optional
            If True, also return the k-point nodes and labels used in the path.
            Format is ``(k_nodes, k_labels)`` where ``k_nodes`` is an array
            of reduced coordinates and ``k_labels`` is a list of strings.

            .. versionadded:: 2.0.0

        Returns
        -------
        kpts : array
            k-points in reduced coordinates used in the
            interpolation in Wannier90 code. The expected format is
            the same as the input to
            :func:`pythtb.TBModel.solve_ham`.
        energy : array
            Energies interpolated by Wannier90 in eV. Format is ``energy[kpt,band]``.
        k_dist : array, optional
            Cumulative distances along the path (1/Angstrom) as reported by Wannier90.
            Returned when ``return_k_dist=True``. Useful for plotting band structures.
        k_cart : array, optional
            k-points in Cartesian coordinates (1/Angstrom).
            Returned when ``return_k_cart=True``.
        k_nodes : tuple[array, list[str]], optional
            Tuple ``(k_nodes, k_labels)`` containing the reduced coordinates
            of the k-point nodes and their labels.
            Returned when ``return_k_nodes=True``.

        Notes
        -----
        - The bands returned here are not the same as the band
          structure calculated by the underlying DFT code. The two will
          agree only on the coarse set of k-points that were used in
          Wannier90 generation.
        - If no matrix elements were ignored in the call to :func:`pythtb.w90.model`
          then the two should be exactly the same (up to numerical precision).
          Otherwise one should expect deviations.
        - If one carefully chooses the cutoff parameters in :func:`pythtb.w90.model`
          it is likely that one could reproduce the full band-structure
          with only few dominant hopping terms.

        Examples
        --------
        Get band structure from `Wannier90`

        >>> w90_kpt, w90_evals, w90_k_dist, w90_k_nodes, w90_k_labels = silicon.bands_w90(
        ... return_k_dist=True, return_k_nodes=True)

        Get simplified model

        >>> my_model_simple = silicon.model(min_hopping_norm=0.01)

        Solve simplified model on the same k-path as in `Wannier90`

        >>> evals = my_model.solve_ham(w90_kpt)

        Now plot the comparison of the two

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> for i in range(evals.shape[1]):
        >>>     ax.plot(range(evals.shape[1]), evals[i], "r-", zorder=-50)
        >>> for i in range(w90_evals.shape[0]):
        >>>     ax.plot(range(w90_evals.shape[1]), w90_evals[i], "k-", zorder=-100)
        >>> fig.savefig("comparison.pdf")
        """
        kpts, ene = read_bands_w90(self.folder, self.prefix, self.num_wan)

        B = self.lattice.recip_lat_vecs
        results = (kpts, ene)

        if return_k_dist:
            k_dist = kpath_distance(kpts, b1=B[0], b2=B[1], b3=B[2])
            results += (k_dist,)
        if return_k_cart:
            k_cart = kpts @ B
            results += (k_cart,)
        if return_k_nodes:
            if self._kpath_nodes_red is None or self._kpath_labels is None:
                k_nodes, k_labels = read_kpoint_path(self._win_lines, latex=True)
            else:
                k_nodes, k_labels = self._kpath_nodes_red, self._kpath_labels
            results += (k_nodes, k_labels)
        return results

    def bands_qe(self, return_k_cart=False, return_meta=False, return_kdist=False):
        """
        Read band structure output from Quantum ESPRESSO `bands.x`.

        Reads in the band structure as computed by Quantum ESPRESSO.
        The code assumes that the following file was generated by
        Quantum ESPRESSO, typically using the `bands.x` utility:

        - ``prefix_bands.dat``

        .. versionadded:: 2.0.0

        Parameters
        ----------
        return_k_cart : bool, optional
            If True, also return k-points in Cartesian coordinates.
        return_meta : bool, optional
            If True, return header metadata ``(nbnd, nks)`` when available.
        return_kdist : bool, optional
            If True, also return cumulative k-path distances (1/Angstrom).

        Returns
        -------
        k_frac : array
            k-points in reduced coordinates used in the band structure.
        energies : array
            Energies computed by Quantum ESPRESSO in eV. Format is ``energies[kpt,band]``.
        k_dist : array, optional
            Cumulative distances along the path (1/Angstrom). Returned when
            ``return_kdist=True``. Useful for plotting band structures.
        k_cart : array, optional
            k-points in Cartesian coordinates (1/Angstrom). Returned when
            ``return_k_cart=True``.
        meta : dict, optional
            Metadata dictionary containing ``nbnd`` and ``nks`` when available.
            Returned when ``return_meta=True``.

        Notes
        -----
        - The purpose of this function is to read band structures
          computed by Quantum ESPRESSO for comparison with PythTB
          calculations based on Wannier90 tight-binding models.
        - The band structure from Quantum ESPRESSO can be compared
          with that from Wannier90 using the :func:`bands_w90` method.
        - Ensure that the k-point paths used in Quantum ESPRESSO
          and Wannier90 are consistent for meaningful comparisons.
        """
        k_markers, energies_rows, meta = read_bands_qe(self.folder, self.prefix)

        # Shape energies into (nks, nbnd)
        nks = meta.get("nks", len(energies_rows))
        nbnd = meta.get(
            "nbnd", max(len(r) for r in energies_rows) if energies_rows else 0
        )

        E = np.full((nks, nbnd), np.nan, dtype=float)
        for i in range(min(nks, len(energies_rows))):
            row = energies_rows[i]
            ncopy = min(nbnd, len(row))
            if ncopy:
                E[i, :ncopy] = row[:ncopy]

        # QE k markers are in units of (2π / alat); scale as your previous method
        k_cart = np.asarray(k_markers, float)
        alat = np.linalg.norm(self.lattice.lat_vecs[0])
        k_cart *= 2 * np.pi / alat

        # Convert to reduced coords using your reciprocal lattice
        B = self.lattice.recip_lat_vecs
        k_frac = k_cart @ np.linalg.inv(B)

        results = [k_frac, E]

        if return_kdist:
            k_dist = kpath_distance(k_frac, b1=B[0], b2=B[1], b3=B[2])
            results.append(k_dist)
        if return_k_cart:
            results.append(k_cart)
        if return_meta:
            results.append(meta)

        return tuple(results)

if __name__ == "__main__":

    a, b = read_hr(Path("/Users/mykhailoklymenko/Monash_work/data/graphene_bilayer/wannier"), "ab_bilayer_scf")

    from pprint import pprint

    pprint(a)
    pprint(b[(0, 0, 0)])