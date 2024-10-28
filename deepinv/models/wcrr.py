import torch

class SplineActivation(torch.nn.Module):
    def __init__(self,max_noise_level=30,convex=False):# max_noise_level?
        self.alpha_spline=LinearSpline(num_knots=11,x_min=0,x_max=max_noise_level,num_activation=80,init=5,clamp=False)
        self.mu_spline=LinearSpline(num_knots=11,x_min=0,x_max=max_noise_level,num_activation=1,init=4.,clamp=False)
        self.phi_minus=LinearSpline(num_knots=101,slope_min=0,slope_max=1,num_activation=1,init=0.,antisymmetric=True)
        self.phi_plus=LinearSpline(num_knots=101,slope_min=0,slope_max=1,num_activation=1,init=0.,antisymmetric=True)

    def forward(self,x,sigma):
        # sigma is either a float or 
        alpha=torch.exp(self.alpha_spline(sigma))/(sigma+1e-5)
        return 1/alpha**2 * (self.mu_spline(sigma)*self.phi_plus.integrate(x)-self.phi_minux.integrate(x))


class LinearSpline(torch.nn.Module):
    """
    Class for LinearSpline activation functions

    Args:
        num_knots (int): number of knots of the spline
        num_activations (int) : number of activation functions
        init : 
        x_min (float): position of left-most knot
        x_max (float): position of right-most knot
        slope_min (float or None): minimum slope of the activation
        slope_max (float or None): maximum slope of the activation
        antisymmetric (bool): Constrain the potential to be symmetric <=> activation antisymmetric
    """

    def __init__(self, num_activations, num_knots, x_min, x_max, init=0.,
                 slope_max=None, slope_min=None, antisymmetric=False, clamp=True, device='cpu'):

        super().__init__()

        self.device=device
        self.num_knots = num_knots
        self.num_activations = num_activations
        self.init = init
        self.x_min = torch.tensor([x_min],device=device)
        self.x_max = torch.tensor([x_max],device=device)
        self.slope_min = slope_min
        self.slope_max = slope_max

        
        self.antisymmetric = antisymmetric
        self.clamp = clamp
        #self.no_constraints = ( slope_max is None and slope_min is None and (not antisymmetric) and not clamp)
        #self.integrated_coeff = None
        
        # parameters
        self.grid_delta = (self.x_max - self.x_min) / (self.num_knots - 1)
        grid_tensor = torch.linspace(self.x_min.item(), self.x_max.item(), self.num_knots).expand((self.num_activations, self.num_knots)).to(device)
        coefficients = torch.ones_like(grid_tensor,device=device)*self.init
        self.coefficients = torch.nn.Parameter(coefficients).to(device)

        #self.projected_coefficients_cached = None

        #self.init_zero_knot_indexes()

    def clipped_coefficients(self):        
        """Simple projection of the spline coefficients to enforce the constraints, for e.g. bounded slope"""
        
        device = self.device
        
        if self.no_constraints:
            return(self.coefficients)
            
        cs = self.coefficients
        
        new_slopes = (cs[:, 1:] - cs[:, :-1]) / self.grid_delta

        if self.slope_min is not None or self.slope_max is not None:
            new_slopes = torch.clamp(new_slopes, self.slope_min, self.slope_max)
        
        
        # clamp extension
        if self.clamp:
            new_slopes[:,0] = 0
            new_slopes[:,-1] = 0
        
        new_cs = torch.zeros(self.coefficients.shape, device=device, dtype=cs.dtype)


        new_cs[:,1:] = torch.cumsum(new_slopes, dim=1) * self.grid_delta

        # preserve the mean, unless antisymmetric
        if not self.antisymmetric:
            new_cs = new_cs + new_cs.mean(dim=1).unsqueeze(1)

        
        # antisymmetry

        ##### care about that later ######
        if self.antisymmetric:
            raise NotImplementedError()
            inv_idx = torch.arange(new_cs.size(1)-1, -1, -1).long().to(new_cs.device)
            # or equivalently torch.range(tensor.num_knots(0)-1, 0, -1).long()
            inv_tensor = new_cs[:,inv_idx]
            new_cs = 0.5*(new_cs - inv_tensor)

        return new_cs
 
    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                4D, depending on weather the layer is
                convolutional ('conv')

        Returns:
            output (torch.Tensor)
        """
        in_shape = x.shape
        in_channels = in_shape[1]

        #if in_channels % self.num_activations != 0:
        #    raise ValueError('Number of input channels must be divisible by number of activations.')

        x = x.view(x.shape[0], self.num_activations, in_channels // self.num_activations, *x.shape[2:])

        projected_coefficients=self.clipped_coefficients()
        x = LinearSpline_Func.apply(x, projected_coefficients, self.x_min, self.x_max, self.num_knots, self.zero_knot_indexes)

        x = x.view(in_shape)

        return x           


class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):

        # The value of the spline at any x is a combination 
        # of at most two coefficients
        grid_delta = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - grid_delta.item())


        floored_x = torch.floor((x_clamped - x_min) / grid_delta)  #left coefficient

        fracs = (x - x_min) / grid_delta - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()

        coefficients_vect = coefficients.view(-1)

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + \
            coefficients_vect[indexes] * (1 - fracs)

        ctx.save_for_backward(fracs, coefficients, indexes, grid_delta)
        # ctx.results = (fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients, indexes, grid_delta = ctx.saved_tensors

        coefficients_vect = coefficients.view(-1)

        grad_x = (coefficients_vect[indexes + 1] -
                  coefficients_vect[indexes]) / grid_delta * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect, dtype=coefficients_vect.dtype)
        # right coefficients gradients
   

        grad_coefficients_vect.scatter_add_(0,
                                            indexes.view(-1) + 1,
                                            (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indexes.view(-1),
                                            ((1 - fracs) * grad_out).view(-1))

        grad_coefficients = grad_coefficients_vect.view(coefficients.shape)

        return grad_x, grad_coefficients, None, None, None, None
