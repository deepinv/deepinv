- [x] Updating the doc with working examples
_ _ _ 
- 
	- [x] functional/ (Flo and Hai 02/04)
_ _ _ 
- 
	- [x] generators/base.py (Flo 03/04)
	- [x] generators/blur.py (Flo 03/04)
	- [x] generators/mri.py (Flo 03/04)
	- **Comment (Flo)**: I had to change the output of AccelerationMaskGenerator.step() because it was outputting 2 identical (repeated) channels per default, which was different the output shape announced in the doc, where channels was set to 1 by default. This should be verified with the person who coded this. (I kept the 2 channel return as a commented line)
- - [x] generators/noise.py (Flo 03/04)

- - [x] Double check by Hai that doc is okay

**Comment (Flo)** : Need to refactorize the doc, as right now individual physics generators appear both under the *Introduction/Generators/* section and in each *Forward operators/* section. I think Julian chose the second option, that we did not see with Hai on Tuesday.

_ _ _ 
- [ ] Refactor doc physics
- [ ] A tour of blur operators
- [ ] Check multiGPU class generator

- [ ] Coding unit test using pytest
  - [ ] Check output types for generator (dict, device, dtype)
  - [ ] Check all boundary conditions + colour + non square images for physics (adjoint)
  - [ ] Check that generators return the right stuff
  - [ ] Check GeneratorMixture (ce qu'a déjà fait Flo)
  - [ ] Check SpaceVaryingBlur
  - [ ] Check ProductConvolution
_ _ _ 
- [ ] Discussions
  - [ ] Move rotate, scale, ... to functional? -> would be better
  - [ ] Restructure the doc (in particular self-supervised, ...)
  - [ ] Add functions to Physics such as jvp, ... This would make it possible to play easily with nonlinear operators, tangent maps,...

