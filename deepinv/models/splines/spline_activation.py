import torch
import deepinv.models.splines.spline_uitls as spline_autograd_func



class LinearSpline(torch.nn.Module):
    r"""
    Evaluation of integrals of linear B-splines 

    i.e., a quadratic B-spline, with custom autograd function. 
    The implementation is (mainly) taken from `this paper <https://epubs.siam.org/doi/10.1137/23M1565243>`_ and can be found `here <https://github.com/axgoujon/weakly_convex_ridge_regularizer>`_.
    
    :param int num_activations: number of splines
    :param int num_knots: number of knots of the spline(s)
    :param float x_min: position of left-most knot
    :param float x_max: position of right-most knot
    :param float init: constant initialization with value init of the spline coefficients
    :param float slope_min: minimum slope of the activation. None (default) for no constraint
    :param float slope_max: maximum slope of the activation. None (default) for no constraint
    :param bool antisymmetric: Constrain the spline to be antisymmetric, i.e., its potential is symmetric
    :param bool clamp: if true constant extension of the spline outside [x_min,x_max], if false, linear extension
    """

    def __init__(self, num_activations, num_knots, x_min, x_max, init,
                 slope_max=None, slope_min=None, antisymmetric=False, clamp=True):

        super().__init__()

        self.num_knots = int(num_knots)
        self.num_activations = int(num_activations)
        self.init = init
        self.x_min = torch.tensor([x_min])
        self.x_max = torch.tensor([x_max])
        self.slope_min = slope_min
        self.slope_max = slope_max

        self.step_size = (self.x_max - self.x_min) / (self.num_knots - 1)
        self.antisymmetric = antisymmetric
        self.clamp = clamp
        self.no_constraints = ( slope_max is None and slope_min is None and (not antisymmetric) and not clamp)
        self.integrated_coeff = None
        
        # parameters
        coefficients = self.initialize_coeffs()  # spline coefficients
        self.coefficients = nn.Parameter(coefficients)

        self.projected_coefficients_cached = None

        self.init_zero_knot_indexes()
            

    def init_zero_knot_indexes(self):
        """ Initialize indexes of zero knots of each activation.
        """
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = (activation_arange * self.num_knots)

    def initialize_coeffs(self):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        init = self.init
        grid_tensor = torch.linspace(self.x_min.item(), self.x_max.item(), self.num_knots).expand((self.num_activations, self.num_knots))        
        return torch.ones_like(grid_tensor)*init
    
    @property
    def projected_coefficients(self):
        """ B-spline coefficients projected to meet the constraint. """
        if self.projected_coefficients_cached is not None:
            return self.projected_coefficients_cached
        else:
            return(self.clipped_coefficients())

    def cached_projected_coefficients(self):
        """ B-spline coefficients projected to meet the constraint. """
        if self.projected_coefficients_cached is None:
            self.projected_coefficients_cached = self.clipped_coefficients()
        

    @property
    def slopes(self):
        """ Get the slopes of the activations
        """
        coeff = self.projected_coefficients
        slopes = (coeff[:, 1:] - coeff[:, :-1]) / self.step_size

        return slopes

    @property
    def device(self):
        return self.coefficients.device
    
    def hyper_param_to_device(self):
        device = self.device
        self.x_min, self.x_max, self.step_size, self.zero_knot_indexes = self.x_min.to(device), self.x_max.to(device), self.step_size.to(device), self.zero_knot_indexes.to(device)


    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        self.hyper_param_to_device()

        in_shape = x.shape
        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError('Number of input channels must be divisible by number of activations.')

        x = x.view(x.shape[0], self.num_activations, in_channels // self.num_activations, *x.shape[2:])

        x = spline_autograd_func.LinearSpline_Func.apply(x, self.projected_coefficients, self.x_min, self.x_max, self.num_knots, self.zero_knot_indexes)

        x = x.view(in_shape)

        return x

    def derivative(self, x):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        
        self.hyper_param_to_device()

        in_shape = x.shape
        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError('Number of input channels must be divisible by number of activations.')

        x = x.view(-1, self.num_activations, in_channels // self.num_activations, *x.shape[2:])

        coeff = self.projected_coefficients

        x = spline_autograd_func.LinearSplineDerivative_Func.apply(x, coeff, self.x_min, self.x_max, self.num_knots, self.zero_knot_indexes)

        x = x.view(in_shape)

        return x


    def update_integrated_coeff(self):
        print("**** Updating integrated spline coefficients ****")
        coeff = self.projected_coefficients
        
        # extrapolate assuming zero slopes at both ends, i.e. linear interpolation for the integrated function
        coeff_int = torch.cat((coeff[:, 0:1], coeff, coeff[:, -1:]), dim=1)

        # integrate to obtain
        # the coefficents of the corresponding quadratic BSpline expansion
        self.integrated_coeff = torch.cumsum(coeff_int, dim = 1)*self.step_size.to(coeff.device)
        
        # remove value at 0 and reshape
        # this is arbitray, as integration is up to a constant
        self.integrated_coeff = (self.integrated_coeff - self.integrated_coeff[:, (self.num_knots + 2)//2].view(-1,1)).view(-1)

        # store once for all knots indexes
        # not the same as for the linear-spline as we have 2 more "virtual" knots now
        self.zero_knot_indexes_integrated = (torch.arange(0, self.num_activations) * (self.num_knots + 2)).to(self.device)

    def integrate(self, x):
        in_shape = x.shape

        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError('Number of input channels must be divisible by number of activations.')
        
        if self.integrated_coeff is None:
            self.update_integrated_coeff()

        if x.device != self.integrated_coeff.device:
            self.integrated_coeff = self.integrated_coeff.to(x.device)
            self.zero_knot_indexes_integrated = self.zero_knot_indexes_integrated.to(x.device)

        x = x.view(-1, self.num_activations, in_channels // self.num_activations, *x.shape[2:])

        x = spline_autograd_func.Quadratic_Spline_Func.apply(x - self.step_size, self.integrated_coeff, self.x_min - self.step_size, self.x_max + self.step_size, self.num_knots + 2, self.zero_knot_indexes_integrated)

        x = x.view(in_shape)

        return(x)

    def extra_repr(self):
        """ repr for print(model) """

        s = ('num_activations={num_activations}, '
             'init={init}, num_knots={num_knots}, range=[{x_min[0]:.3f}, {x_max[0]:.3f}], '
             'slope_max={slope_max}, '
             'slope_min={slope_min}.'
             )

        return s.format(**self.__dict__)

    
    def clipped_coefficients(self):        
        """Simple projection of the spline coefficients to enforce the constraints, for e.g. bounded slope"""
        
        device = self.device
        
        if self.no_constraints:
            return(self.coefficients)
            
        cs = self.coefficients
        
        new_slopes = (cs[:, 1:] - cs[:, :-1]) / self.step_size

        if self.slope_min is not None or self.slope_max is not None:
            new_slopes = torch.clamp(new_slopes, self.slope_min, self.slope_max)
        
        
        # clamp extension
        if self.clamp:
            new_slopes[:,0] = 0
            new_slopes[:,-1] = 0
        
        new_cs = torch.zeros(self.coefficients.shape, device=device, dtype=cs.dtype)


        new_cs[:,1:] = torch.cumsum(new_slopes, dim=1) * self.step_size

        # preserve the mean, unless antisymmetric
        if not self.antisymmetric:
            new_cs = new_cs + (cs-new_cs).mean(dim=1).unsqueeze(1)

        
        # antisymmetry
        if self.antisymmetric:
            inv_idx = torch.arange(new_cs.size(1)-1, -1, -1).long().to(new_cs.device)
            # or equivalently torch.range(tensor.num_knots(0)-1, 0, -1).long()
            inv_tensor = new_cs[:,inv_idx]
            new_cs = 0.5*(new_cs - inv_tensor)

        return new_cs
