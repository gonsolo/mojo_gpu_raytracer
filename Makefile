all: view
raytracer: raytracer.mojo
	mojo build raytracer.mojo
render.ppm: raytracer
	./raytracer
render.jpeg: render.ppm
	magick render.ppm render.jpeg
view: render.ppm
	gimp render.ppm
edit:
	vi raytracer.mojo
clean:
	rm -f raytracer render.ppm
.PHONY: all edit view
