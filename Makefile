all: view
raytracer: raytracer.mojo
	uv run mojo build raytracer.mojo
cpu.ppm gpu.ppm cpu_parallel.ppm: raytracer
	./raytracer
gpu.jpeg: gpu.ppm
	magick gpu.ppm gpu.jpeg
view: cpu.ppm gpu.ppm cpu_parallel.ppm
	gimp cpu.ppm gpu.ppm cpu_parallel.ppm
e edit:
	uv run vim raytracer.mojo
clean:
	git clean -dfx
.PHONY: all e edit view
