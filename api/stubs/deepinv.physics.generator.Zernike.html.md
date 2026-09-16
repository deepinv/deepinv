# Zernike

### *class* deepinv.physics.generator.Zernike

Bases: [`object`](https://docs.python.org/3.9/library/functions.html#object)

Static utility class for Zernike polynomials (ANSI/Noll Nomenclature).
All methods are static; no instantiation required.

Zernike polynomials are a sequence of polynomials that are orthogonal on the unit disk.
They are commonly used in optical systems to describe wavefront aberrations, see Lakshminarayanan and Fleck<sup>[1](#footcite-lakshminarayanan2011zernike)</sup>
(or [this link](https://e-l.unifi.it/pluginfile.php/1055875/mod_resource/content/1/Appunti_2020_Lezione%2014_4_Zernikepolynomialsaguidefinal.pdf)).

They are defined by two indices: the radial order $n$ and the azimuthal frequency $m$, with the constraints that $n >= 0$, $|m| <= n$, and $n - |m|$ is even.

The Zernike polynomial $Z_n^m$ can be expressed in polar coordinates $(\\rho, \\theta)$ as:

$$
Z_n^m(\rho, \theta) =

\begin{cases}
    N_n^m R_n^m(\rho)  \cos(m \theta)      & \text{ if } m \geq 0 \\
    N_n^m R_n^m(\rho)  \sin(|m| \theta)    & \text{ if } m < 0
\end{cases}
$$

where $R_n^m(\rho)$ is the radial polynomial defined as:

$$
R_n^m(\rho) = \sum_{k=0}^{(n - |m|)/2} (-1)^k \frac{(n - k)!}{k! \left(\frac{n + |m|}{2} - k\right)! \left(\frac{n - |m|}{2} - k\right)!} \rho^{n - 2k}
$$

and $N_n^m$ is the normalization constant (root-mean-square, ensuring the polynomials have unit energy) given by:

$$
N_n^m = \begin{cases} \sqrt{n + 1} & \text{ if } m = 0 \\ \sqrt{2(n + 1)} & \text{ if } m \neq 0 \end{cases}
$$

<hr />

* **References:**

* <a id='footcite-lakshminarayanan2011zernike'>**[1]**</a> Vasudevan Lakshminarayanan and Andre Fleck. Zernike polynomials: a guide. *Journal of Modern Optics*, 58(7):545–561, 2011.

#### *static* cartesian_evaluate(n, m, x, y, use_mask=True)

Evaluates the Zernike polynomial $Z_n^m$ on given Cartesian coordinates.

* **Parameters:**
  * **n** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the radial order
  * **m** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the azimuthal frequency
  * **x** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – x-coordinates in Cartesian system
  * **y** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – y-coordinates in Cartesian system
  * **use_mask** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, values outside the unit disk are set to zero. Default to `True`
* **Returns:**
  Evaluated Zernike polynomial values as a tensor
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

#### *static* get_name(n, m)

Returns the ANSI standard name given indices $n$ and $m$.

* **Parameters:**
  * **n** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the radial order
  * **m** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the azimuthal frequency
* **Returns:**
  The ANSI standard name or a default string if not found.
* **Return type:**
  [str](https://docs.python.org/3.9/library/stdtypes.html#str)

#### *static* index_conversion(index, , convention='ansi')

Converts a single index for Zernike polynomials between different conventions.
For more details on the conventions, see [wikipedia](https://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials).

* **Parameters:**
  * **index** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the single index of the Zernike polynomial.
  * **convention** ([*str*](https://docs.python.org/3.9/library/stdtypes.html#str)) – the convention to convert to. Currently only ‘ansi’ (for OSA/ANSI standard indices) and ‘noll’ (for Noll’s sequential indices) are implemented.

Single index for Zernike polynomials:

#### *static* normalization_constant(n, m)

Returns the Noll RMS normalization constant.

* **Parameters:**
  * **n** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the radial order
  * **m** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the azimuthal frequency
* **Returns:**
  The normalization constant.
* **Return type:**
  [float](https://docs.python.org/3.9/library/functions.html#float)

#### *static* polar_evaluate(n, m, rho, theta, use_mask=True)

Evaluates the Zernike polynomial $Z_n^m$ on given polar coordinates.

* **Parameters:**
  * **n** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the radial order
  * **m** ([*int*](https://docs.python.org/3.9/library/functions.html#int)) – the azimuthal frequency
  * **rho** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – radial coordinates in polar system
  * **theta** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) – angular coordinates in polar system
  * **use_mask** ([*bool*](https://docs.python.org/3.9/library/functions.html#bool)) – if True, values outside the unit disk are set to zero. Default to `True`
* **Returns:**
  Evaluated Zernike polynomial values as a tensor
* **Return type:**
  [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)
