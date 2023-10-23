# Optimizers for parameters on manifolds

from typing import Optional, List

import os

import torch
from torch import nn

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
from geomstats.geometry.manifold import Manifold
from .parameters import StiefelParameter

# When to use Riemannian Gradient or not
LIST_OF_PARAMETERS_ON_MANIFOLDS = [StiefelParameter]


def _get_lr_with_strategy(lr: float, strategy: str, step_count: int,
                        decay: float) -> float:
    """Get the learning rate with a given strategy.

    Parameters
    ----------
    lr : float
        Initial learning rate.

    strategy : str
        Learning rate strategy to use. Can be None, in which case the
        learning rate is constant.
        Other options are: "inverse", "inverse_sqrt", "exponential" or
        "linear".

    step_count : int
        Number of steps since the beginning of the training.

    decay : float
        Decay rate of the learning rate. Actual decay depends on the
        learning rate strategy.

    Returns
    -------
    float
        Learning rate with the given strategy.
    """
    if strategy is None:
        return lr
    if strategy == "inverse":
        return lr / (1 + decay * step_count)
    if strategy == "inverse_sqrt":
        return lr / (1 + decay * step_count)**0.5
    if strategy == "exponential":
        return lr * decay**step_count
    if strategy == "linear":
        return lr * (1 - decay * step_count)
    raise ValueError(f"Unknown learning rate strategy {strategy}.")
    

class ManifoldGradientDescent():
    """Generic class for a basic Riemannian gradient descent optimizer on a
    geomstats manifold.

    Attributes
    ----------
    parameters : torch.nn.ParameterList
        Parameters to optimize.

    lr : float
        Learning rate of the optimizer.

    manifolds : List of geomstats Manifold
        Manifolds on which the parameters are defined.
    """

    def __init__(self, parameters:List[nn.Parameter],
                 manifolds: Optional[List[Manifold]] = None,
                 lr: float = 1e-2, lr_strategy: Optional[str] = None,
                 lr_decay: Optional[float] = 0) -> None:
        """Constructor of the ManifoldGradientDescent class.

        Parameters
        ----------
        parameters : torch.nn.ParameterList
            Parameters to optimize.

        manifolds : List of geomstats Manifold, optional
            Manifolds on which the parameters are defined. Can have a signle
            elemenet, in which case all parameters are assumed to be on the
            same manifold.
            If None, the parameters must implement a get_manifold() method.

        lr : float, optional
            Learning rate of the optimizer. Default is 1e-2.

        lr_strategy : str, optional
            Learning rate strategy to use. Can be None, in which case the
            learning rate is constant. Default is None.
            Other options are: "inverse", "inverse_sqrt", "exponential" or 
            "linear".

        lr_decay : float, optional
            Decay rate of the learning rate. Default is 0.
            Actual decay depends on the learning rate strategy.
        """
        self.parameters = parameters
        self.lr = lr
        self.lr_strategy = lr_strategy
        self.lr_decay = lr_decay
        self.initial_lr = lr
        self.step_count = 0
        
        if manifolds is None:
            self.manifolds = [p.get_manifold() for p in parameters]

        elif len(manifolds) == 1:
            self.manifolds = [manifolds[0] for _ in parameters]

        else:
            assert len(manifolds) == len(parameters), \
                    f"Manifolds and parameters must have the same length." \
                    f"Got {len(manifolds)} manifolds and" \
                    f" {len(parameters)} parameters."

            self.manifolds = manifolds

    def step(self) -> None:
        """Performs a single optimization step."""
        for parameter, manifold in zip(self.parameters, self.manifolds):
            lr = _get_lr_with_strategy(self.lr, self.lr_strategy,
                                    self.step_count, self.lr_decay)
            direction_tangentspace = manifold.to_tangent(
                    parameter.grad.data, parameter.data).to(parameter.data.device)
            if not hasattr(manifold.metric, "retraction"):
                map_function = manifold.metric.exp
            else:
                map_function = manifold.metric.retraction
            parameter.data = map_function(
                tangent_vec=-lr*direction_tangentspace,
                base_point=parameter.data
            ).to(parameter.data.device)

    def zero_grad(self) -> None:
        """Sets the gradient of all parameters to zero."""
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()

    def __repr__(self) -> str:
        """Representation of the optimizer.

        Returns
        -------
        str
            Representation of the optimizer.
        """
        unique_manifolds_names = set([m.__class__.__name__
                                    for m in self.manifolds])
        count_manifolds = {name: 0 for name in unique_manifolds_names}
        for m in self.manifolds:
            count_manifolds[m.__class__.__name__] += 1
        string = f'{self.__class__.__name__}('
        string += f'lr={self.lr}, lr_strategy={self.lr_strategy}, ' \
                f'lr_decay={self.lr_decay}, '
        for name, count in count_manifolds.items():
                string += f'{name}({count}), '
        string = string[:-2] + ')'             
        return string

    def __str__(self) -> str:
        """String representation of the optimizer.

        Returns
        -------
        str
            String representation of the optimizer.
        """
        return self.__repr__()

class MixRiemannianOptimizer:
    """An optimizer that uses Riemannian gradient descent for parameters on
    manifolds and another standard optimizer for other parameters.

    Attributes
    -----------
    parameters : torch.nn.ParameterList
        Parameters to optimize.

    optimizer : torch.optim.Optimizer
        Standard optimizer to use for parameters that are not on a manifold.

    lr : float, optional
        Learning rate of the optimizer. Default is 1e-2.
    """

    def __init__(self, parameters: nn.ParameterList,
                optimizer: torch.optim.Optimizer = torch.optim.SGD,
                lr: float = 1e-2, args_manifold: Optional[dict] = None,
                *args, **kwargs) -> None:
        """Constructor of the MixRiemannianOptimizer class.

        Parameters
        ----------
        parameters : torch.nn.ParameterList
            Parameters to optimize.

        optimizer : torch.optim.Optimizer
            Standard optimizer to use for parameters that are not on a manifold.

        lr : float, optional
            Learning rate of the optimizer. Default is 1e-2.

        args_manifold : dict, optional
            Arguments to pass to the ManifoldGradientDescent optimizer.
            Besides the parameters and the learning rate.

        *args, **kwargs
            Arguments and keyword arguments to pass to the standard optimizer.
        """
        self.lr = lr
        self.parameters = list(parameters)
        indice_manifolds = [i for i,_ in enumerate(self.parameters)
                            if any([isinstance(_, parameter_possible)
                                for parameter_possible
                                in LIST_OF_PARAMETERS_ON_MANIFOLDS])]
        indice_standard = [i for i,_ in enumerate(self.parameters)
                        if i not in indice_manifolds]
        self.manifold_parameters = [self.parameters[i] 
                                    for i in indice_manifolds]
        self.standard_parameters = [self.parameters[i] 
                                    for i in indice_standard]

        if len(self.standard_parameters) > 0:
            self.standard_optimizer = optimizer(self.standard_parameters,
                                lr=self.lr, *args, **kwargs)
        else:
            self.standard_optimizer = None
        if args_manifold is None:
            self.manifold_optimizer = ManifoldGradientDescent(
                    self.manifold_parameters, lr=self.lr)
        else:
            self.manifold_optimizer = ManifoldGradientDescent(
                    self.manifold_parameters, lr=self.lr, **args_manifold)

    def step(self) -> None:
        """Performs a single optimization step."""
        if self.standard_optimizer is not None:
            self.standard_optimizer.step()
        self.manifold_optimizer.step()

    def zero_grad(self) -> None:
        """Sets the gradient of all parameters to zero."""
        if self.standard_optimizer is not None:
            self.standard_optimizer.zero_grad()
        self.manifold_optimizer.zero_grad()

    def __repr__(self) -> str:
        if self.standard_optimizer is None:
            return f'{self.__class__.__name__}(' \
              f'lr={self.lr}, ' \
            f'manifold_optimizer={self.manifold_optimizer.__class__.__name__})'
        return f'{self.__class__.__name__}(' \
         f'lr={self.lr}, ' \
         f'standard_optimizer={self.standard_optimizer.__class__.__name__}, ' \
         f'manifold_optimizer={self.manifold_optimizer.__class__.__name__})'

    def __str__(self) -> str:
        """String representation of the optimizer.

        Returns
        -------
        str
            String representation of the optimizer.
        """
        return self.__repr__()

