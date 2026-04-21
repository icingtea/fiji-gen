ifndef LIBTORCH_PATH
	$(error LIBTORCH_PATH is not set. Please export it as an environment variable before running make.)
endif

.PHONY: all clean

all:
	mkdir -p build
	cd build && cmake ..
	$(MAKE) -C build

clean:
	rm -rf build bin
