all: view
raytracer: raytracer.mojo
	mojo build raytracer.mojo
cpu.ppm gpu.ppm: raytracer
	./raytracer
gpu.jpeg: gpu.ppm
	magick render.ppm render.jpeg
view: cpu.ppm gpu.ppm
	gimp cpu.ppm gpu.ppm
e edit:
	vi raytracer.mojo
clean:
	rm -f raytracer render.ppm
.PHONY: all e edit view
