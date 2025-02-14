# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DEEPINV_DATA  ?= ../../datasets/

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	DEEPINV_DATA=$(DEEPINV_DATA) $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# This is deprecated, use html-no-examples instead
html-noplot:
	$(SPHINXBUILD) -W -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-fast:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-pattern:
	$(SPHINXBUILD) -W -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo
	@echo "Build finished. The HTML pages are in _build/html"

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf source/auto_examples/
	rm -rf source/generated/
	rm -rf source/gen_modules/
	rm -rf source/api/stubs/
	rm source/sg_execution_times.rst

clean-win:
	@if exist $(BUILDDIR) rmdir /s /q $(BUILDDIR)
	@if exist source\auto_examples rmdir /s /q source\auto_examples
	@if exist source\generated rmdir /s /q source\generated
	@if exist source\gen_modules rmdir /s /q source\gen_modules
	@if exist source\api\stubs rmdir /s /q source\api\stubs
	@if exist source\sg_execution_times.rst del source\sg_execution_times.rst
