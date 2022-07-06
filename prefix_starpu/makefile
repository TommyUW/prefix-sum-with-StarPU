CFLAGS += $$(pkg-config --cflags starpu-1.2)
CFLAGS += -fopenmp
LDLIBS += $$(pkg-config --libs starpu-1.2)
LDLIBS += -fopenmp
main: main.o
clean:
	rm -f main *.o

