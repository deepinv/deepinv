import torch
import math


# Dictionary for Standard aberrations
# See: https://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials
_NAMES = {
    (0, 0): "Zernike(n = 0, m = 0) -- Piston",
    (1, -1): "Zernike(n = 1, m = -1) -- Vertical Tilt",
    (1, 1): "Zernike(n = 1, m = 1) -- Horizontal Tilt",
    (2, -2): "Zernike(n = 2, m = -2) -- Oblique Astigmatism",
    (2, 0): "Zernike(n = 2, m = 0) -- Defocus",
    (2, 2): "Zernike(n = 2, m = 2) -- Vertical Astigmatism",
    (3, -3): "Zernike(n = 3, m = -3) -- Vertical Trefoil",
    (3, -1): "Zernike(n = 3, m = -1) -- Vertical Coma",
    (3, 1): "Zernike(n = 3, m = 1) -- Horizontal Coma",
    (3, 3): "Zernike(n = 3, m = 3) -- Oblique Trefoil",
    (4, -4): "Zernike(n = 4, m = -4) -- Oblique Quadrafoil",
    (4, -2): "Zernike(n = 4, m = -2) -- Oblique Secondary Astigmatism",
    (4, 0): "Zernike(n = 4, m = 0) -- Primary Spherical",
    (4, 2): "Zernike(n = 4, m = 2) -- Vertical Secondary Astigmatism",
    (4, 4): "Zernike(n = 4, m = 4) -- Vertical Quadrafoil",
    (6, 0): "Zernike(n = 6, m = 0) -- Secondary Spherical",
}


class Zernike:
    r"""
    Static utility class for Zernike polynomials (ANSI/Noll Nomenclature).
    All methods are static; no instantiation required.

    Zernike polynomials are a sequence of polynomials that are orthogonal on the unit disk.
    They are commonly used in optical systems to describe wavefront aberrations, see :footcite:t:`lakshminarayanan2011zernike`
    (or `this link <https://e-l.unifi.it/pluginfile.php/1055875/mod_resource/content/1/Appunti_2020_Lezione%2014_4_Zernikepolynomialsaguidefinal.pdf>`_).

    They are defined by two indices: the radial order :math:`n` and the azimuthal frequency :math:`m`, with the constraints that :math:`n >= 0`, :math:`|m| <= n`, and :math:`n - |m|` is even.

    The Zernike polynomial :math:`Z_n^m` can be expressed in polar coordinates :math:`(\\rho, \\theta)` as:

    .. math::

        Z_n^m(\rho, \theta) = 
        
        \begin{cases} 
            N_n^m R_n^m(\rho)  \cos(m \theta)      & \text{ if } m \geq 0 \\ 
            N_n^m R_n^m(\rho)  \sin(|m| \theta)    & \text{ if } m < 0 
        \end{cases}

    where :math:`R_n^m(\rho)` is the radial polynomial defined as:

    .. math::

        R_n^m(\rho) = \sum_{k=0}^{(n - |m|)/2} (-1)^k \frac{(n - k)!}{k! \left(\frac{n + |m|}{2} - k\right)! \left(\frac{n - |m|}{2} - k\right)!} \rho^{n - 2k}
        
    and :math:`N_n^m` is the normalization constant (root-mean-square, ensuring the polynomials have unit energy) given by:

    .. math::

        N_n^m = \begin{cases} \sqrt{n + 1} & \text{ if } m = 0 \\ \sqrt{2(n + 1)} & \text{ if } m \neq 0 \end{cases}

    """

    @staticmethod
    def get_name(n: int, m: int) -> str:
        """Returns the ANSI standard name given indices :math:`n` and :math:`m`.

        :param int n: the radial order
        :param int m: the azimuthal frequency

        :return: The ANSI standard name or a default string if not found.
        """
        Zernike._validate(n, m)
        return _NAMES.get((n, m), f"Zernike(n={n}, m={m})")

    @staticmethod
    def normalization_constant(n: int, m: int) -> float:
        """Returns the Noll RMS normalization constant.

        :param int n: the radial order
        :param int m: the azimuthal frequency

        :return: The normalization constant.
        """
        # Noll constant: sqrt(n+1) if m=0, else sqrt(2n+2)
        # See: https://e-l.unifi.it/pluginfile.php/1055875/mod_resource/content/1/Appunti_2020_Lezione%2014_4_Zernikepolynomialsaguidefinal.pdf (eq. 4)
        if m == 0:
            return math.sqrt(n + 1)
        else:
            return math.sqrt(2 * (n + 1))

    @staticmethod
    def cartesian_evaluate(
        n: int, m: int, x: torch.Tensor, y: torch.Tensor, use_mask: bool = True
    ) -> torch.Tensor:
        """
        Evaluates the Zernike polynomial :math:`Z_n^m` on given Cartesian coordinates.

        :param int n: the radial order
        :param int m: the azimuthal frequency
        :param torch.Tensor x: x-coordinates in Cartesian system
        :param torch.Tensor y: y-coordinates in Cartesian system
        :param bool use_mask: if True, values outside the unit disk are set to zero. Default to `True`
        :return: Evaluated Zernike polynomial values as a tensor
        """
        Zernike._validate(n, m)

        # Convert Cartesian (x, y) -> Polar (rho, theta)
        # This is the most stable way to evaluate Zernike polynomials in Cartesian coordinates
        rho = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)

        return Zernike.polar_evaluate(n, m, rho, theta, use_mask)

    @staticmethod
    def polar_evaluate(
        n: int, m: int, rho: torch.Tensor, theta: torch.Tensor, use_mask: bool = True
    ) -> torch.Tensor:
        """
        Evaluates the Zernike polynomial :math:`Z_n^m` on given polar coordinates.

        :param int n: the radial order
        :param int m: the azimuthal frequency
        :param torch.Tensor rho: radial coordinates in polar system
        :param torch.Tensor theta: angular coordinates in polar system
        :param bool use_mask: if True, values outside the unit disk are set to zero. Default to `True`
        :return: Evaluated Zernike polynomial values as a tensor
        """
        Zernike._validate(n, m)

        # Radial polynomial R(rho)
        R = Zernike._radial_polynomial(n, m, rho)

        # Angular function
        if m >= 0:
            angular = torch.cos(m * theta)
        else:
            angular = torch.sin(abs(m) * theta)

        # Normalization
        norm = Zernike.normalization_constant(n, m)
        Z = norm * R * angular

        # Mask values outside unit disk
        if use_mask:
            Z[rho > 1.0] = 0.0

        return Z

    @staticmethod
    def _radial_polynomial(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
        """
        Internal helper to calculate :math:`R_n^m(rho)`.
        """
        R = torch.zeros_like(rho)
        m_abs = abs(m)
        k_max = (n - m_abs) // 2

        # See: https://en.wikipedia.org/wiki/Zernike_polynomials#Definitions
        for k in range(k_max + 1):
            # Compute coefficient using integer math (math.factorial) for precision
            num = (-1) ** k * math.factorial(n - k)
            den = (
                math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)
            )
            coeff = num / den

            # Add term
            R += coeff * (rho ** (n - 2 * k))
        return R

    @staticmethod
    def _validate(n: int, m: int):
        """Validates indices."""
        if n < 0:
            raise ValueError(f"n must be >= 0. Got {n}.")
        if abs(m) > n:
            raise ValueError(f"|m| must be <= n. Got n={n}, m={m}.")
        if (n - abs(m)) % 2 != 0:
            raise ValueError(f"n - |m| must be even. Got n={n}, m={m}.")

    @staticmethod
    def index_conversion(index: int, *, convention: str = "ansi") -> tuple[int, int]:
        r"""
        Converts a single index for Zernike polynomials between different conventions.
        For more details on the conventions, see `wikipedia <https://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials>`_.

        :param int index: the single index of the Zernike polynomial.
        :param str convention: the convention to convert to. Currently only 'ansi' (for OSA/ANSI standard indices) and 'noll' (for Noll's sequential indices) are implemented.

        Single index for Zernike polynomials:
        """
        # For ansi, we implement the following conversion:
        # https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices
        if convention.lower() == "ansi":
            n = math.floor((2 * index + 0.25) ** 0.5 - 0.5)
            m = 2 * index - n * (n + 2)
            return n, m

        # For Noll's sequential indices, we refer to the following link:
        # https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices
        # The formula is taken from: https://oeis.org/A375779/a375779.pdf
        elif convention.lower() == "noll":
            if index < 1:
                raise ValueError("Noll index must be >= 1")

            n = math.floor((2 * (index - 1) + 0.25) ** 0.5 - 0.5)
            m = n % 2 + 2 * math.floor((index - n * (n + 1) / 2 - 1 + (n + 1) % 2) / 2)
            # Correct sign of m
            m = m * (-1) ** index

            return n, m
        else:
            raise NotImplementedError(
                "Only 'ANSI' and 'Noll' conventions are implemented."
            )
