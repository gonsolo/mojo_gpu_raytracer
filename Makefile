all: view
raytracer: raytracer.mojo
	mojo build raytracer.mojo
cpu.ppm gpu.ppm: raytracer
	./raytracer
gpu.jpeg: gpu.ppm
	magick gpu.ppm gpu.jpeg
view: cpu.ppm gpu.ppm
	gimp cpu.ppm gpu.ppm
e edit:
	vi raytracer.mojo
clean:
	git clean -dfx
.PHONY: all e edit view
