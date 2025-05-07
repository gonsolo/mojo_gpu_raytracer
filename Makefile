all: view
raytracer:
	mojo build raytracer.mojo
render.ppm: raytracer
	./raytracer
view: render.ppm
	gimp render.ppm
edit:
	vi raytracer.mojo
clean:
	rm -f raytracer render.ppm
.PHONY: all edit view
