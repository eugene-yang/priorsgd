all: clean build

cl: clean
clean: 
	@rm -rf build
	@rm -f *.c
	@rm -f *.so
	@rm -f *.html
	@echo "Cleaned sgd package"

build:
	@python setup.py build_ext -i
