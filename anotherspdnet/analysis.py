# Useful classes and functions for analysis

import os
from dataclasses import dataclass
import torch
from torch import nn
from math import prod
import warnings
from typing import List, Optional, Callable



@dataclass
class EigenvaluesLogger:
    """Class to log eigenvalues evolution of SPDnet type layers.
    One log file per layer tracked.


    if type_layers_tracked is given, we track all the layers of the model that
    are of the type given in the list. 

    If filter_layer is given, we track all
    the layers that verify the condition given by filter_layer. If both are
    given, we track all the layers that verify the condition given by
    filter_layer and that are of the type given in the list.

    if type_layers_tracked is None, the list should be filled with all the
    layers of the model that you want tracked. 

    if mode is "no_spd", only log when input matrix is not SPD. If mode is
    If mode is "all", the logger will log the eigenvalues at each forward pass
    of the layer.
    """
    model: torch.nn.Module
    type_layers_tracked: List[nn.Module]
    filter_layer: Callable[[nn.Module], bool]
    list_layers: List[nn.Module]
    storage_path: str
    step_name: str = "step_0"
    step: int = -1
    log_file_basename: str = "eig"
    separator: str = ";"
    mode: str = "no_spd"
    spd_tolerance: float = 1e-4

    def __post_init__(self) -> None:

        # Verifying that the storage path exists
        if not os.path.exists(self.storage_path):
            raise ValueError(f'The storage path {self.storage_path} does not '
                              'exist')

        if self.type_layers_tracked is None and self.list_layers is None:
            raise ValueError('Either type_layers_tracked or list_layers '
                            'should be given')

        if self.type_layers_tracked is not None and self.list_layers is not None:
            raise ValueError('Either type_layers_tracked or list_layers '
                    'should be given')

        if self.type_layers_tracked is not None:
            self.list_layers = []
            self.number_layers = 0
            for layer in self.model.modules():
                if isinstance(layer, tuple(self.type_layers_tracked)):
                    if self.filter_layer is not None:

                        if self.filter_layer(layer):
                            self.list_layers.append(layer)
                            self.number_layers += 1
                    else:
                        self.list_layers.append(layer)
                        self.number_layers += 1

        self.handles = []

    def hook_layers(self) -> None:
        """hook the layers to track the eigenvalues"""
        if len(self.handles) > 0:
            for handle in self.handles:
                handle.remove()
            self.handles = []

        def _forward_hook_all(layer: nn.Module, input: torch.Tensor,
                      output: torch.Tensor) -> None:
            """Forward hook to log the eigenvalues of the layer"""
            layer_number = layer.layer_number
            # Copy the output to avoid modifying it
            _output = output.clone().detach().requires_grad_(False)
            eig = torch.linalg.eigvalsh(_output)

            # Reshaping the eigenvalues if needed
            if len(eig.shape) > 2:
                eig = eig.reshape((prod(eig.shape[:-1]), eig.shape[-1]))

            if eig.shape[-1] > layer.dim:
                warnings.warn(f'Layer {layer_number} has output of '
                            f'dimension {len(eig)} but should have '
                            f'dimension {layer.dim}')

            # Getting the log file
            log_file = os.path.join(self.storage_path,
                                    self.log_file_basename +
                                    f'_{layer_number}.csv')
            # Writing the eigenvalues
            with open(log_file, 'a') as f:
                for sample in range(eig.shape[0]):
                    f.write(f'{self.step_name}{self.separator}')
                    for i in range(layer.dim):
                        if i == layer.dim - 1:
                            f.write(f'{eig[sample, i]}')
                        else:
                            f.write(f'{eig[sample, i]}{self.separator}')
                    f.write('\n')

            # Writing bias if exists
            if hasattr(layer, 'bias'):
                log_file = os.path.join(self.storage_path,
                            self.log_file_basename +
                            f'_{layer_number}_bias.csv')
                # Check if norm not already logged
                if os.stat(log_file).st_size == 0:
                    write_bias = True
                else:
                    with open(log_file, 'r') as f:
                        last_line = f.readlines()[-1]
                    write_bias = last_line.split(self.separator)[0] != \
                            f'{self.step_name}'

                if write_bias:
                    with open(log_file, 'a') as f:
                        f.write(f'{self.step_name}{self.separator}'
                                f'{torch.norm(layer.bias)}\n')

        def _forward_hook_nospd(layer: nn.Module, input: torch.Tensor,
                                output: torch.Tensor) -> None:
            """Forward hook to log the eigenvalues of the layer"""
            layer_number = layer.layer_number
            # Copy the input to avoid modifying it
            _output = output.clone().detach().requires_grad_(False)
            eig = torch.linalg.eigvalsh(_output)
            if not torch.all(eig >= self.spd_tolerance):
                if len(eig.shape) > 2:
                    eig = eig.reshape((prod(eig.shape[:-1]), eig.shape[-1]))

                if eig.shape[-1] > layer.dim:
                    warnings.warn(f'Layer {layer_number} has output of '
                                f'dimension {len(eig)} but should have '
                                f'dimension {layer.dim}')

                # Getting the log file
                log_file = os.path.join(self.storage_path,
                                        self.log_file_basename +
                                        f'_{layer_number}.csv')
                # Writing the eigenvalues
                with open(log_file, 'a') as f:
                    for sample in range(eig.shape[0]):
                        f.write(f'{self.step_name}{self.separator}')
                        for i in range(len(eig[sample])):
                            if i == len(eig[sample]) - 1:
                                f.write(f'{eig[sample, i]}')
                            else:
                                f.write(f'{eig[sample, i]}{self.separator}')
                        f.write('\n')

                # Writing bias if exists
                if hasattr(layer, 'bias'):
                    log_file = os.path.join(self.storage_path,
                                self.log_file_basename +
                                f'_{layer_number}_bias.csv')
                    # Check if norm not already logged
                    if os.stat(log_file).st_size == 0:
                        write_bias = True
                    else:
                        with open(log_file, 'r') as f:
                            last_line = f.readlines()[-1]
                        write_bias = last_line.split(self.separator)[0] != \
                                f'{self.step_name}'
                    if write_bias:
                        with open(log_file, 'a') as f:
                            f.write(f'{self.step_name}{self.separator}'
                                    f'{torch.norm(layer.bias)}\n')

        for layer_number, layer in enumerate(self.list_layers):

            # To have access in the hook to the layer number
            layer.layer_number = layer_number

            if self.mode == "no_spd":
                handle = layer.register_forward_hook(_forward_hook_nospd)
            else:
                handle = layer.register_forward_hook(_forward_hook_all)
            self.handles.append(handle)


    def create_log_files(self) -> None:
        """Create the log files for each layer tracked"""

        # File to get association between layer number and layer name
        log_file = os.path.join(self.storage_path, 'layers.csv')
        with open(log_file, 'w') as f:
            f.write(f'layer_name{self.separator}layer_number\n')
            for i, layer in enumerate(self.list_layers):
                f.write(f'{str(layer)}{self.separator}{i}\n')

        # Files to log the eigenvalues
        for i, layer in enumerate(self.list_layers):
            log_file = os.path.join(self.storage_path,
                            self.log_file_basename + f'_{i}.csv')
            with open(log_file, 'w') as f:
                f.write(f'step{self.separator}')
                for j in range(layer.dim):
                    if j == layer.dim - 1:
                        f.write(f'eig_{j}')
                    else:
                        f.write(f'eig_{j}{self.separator}')
                f.write('\n')

            if hasattr(layer, 'bias'):
                log_file = os.path.join(self.storage_path,
                                        self.log_file_basename
                                        + f'_{i}_bias.csv')
                with open(log_file, 'w') as f:
                    f.write(f'step{self.separator}norm bias')
                    f.write('\n')

    def new_step(self, step_name: Optional[str] = None) -> None:
        """Create a new step in the logging (e.g. new epoch).

        Parameters
        ----------
        step_name : Optional[str]
            Name of the step. By default, we use the current step number.
        """

        self.step += 1
        if step_name is None:
            step_name = f"step_{self.step}"
        self.step_name = step_name

        # Creating the log files if needed
        if self.step == 0:
                self.create_log_files()

        # Rehooking the layers
        self.hook_layers()
