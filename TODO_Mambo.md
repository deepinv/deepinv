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

- - [ ] Double check by Hai that doc is okay

**Comment (Flo)** : Need to refactorize the doc, as right now individual physics generators appear both under the *Introduction/Generators/* section and in each *Forward operators/* section. I think Julian chose the second option, that we did not see with Hai on Tuesday.

_ _ _ 

- [ ] Coding unit test using pytest


_ _ _ 

- [ ] Master example on deblurring to illustrate the different added functionalities
