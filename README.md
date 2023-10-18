# Another SPDnet implementation

![Build status](https://img.shields.io/github/actions/workflow/status/ammarmian/anotherspdnet/build_test.yaml) ![Documentation Status](https://img.shields.io/github/actions/workflow/status/ammarmian/anotherspdnet/documentation.yaml?label=documentation)



This repository contains an alternative implementation of SPDnet from:
* Davoudi
* torchspdnet

Main differences are:
* Extensive relying on `torch.einsum` for increased computational speed
* Grad of BiMap coded with `einsum` rather than using `autograd`
* Works for any number of batch dimensions (rather than 1 for Davoudi and 2 for torchspdnet)
* bias term on the eigenvalues in the reEig layers ?
