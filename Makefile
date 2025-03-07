CPPFLAGS = -std=c++20 -Wall -Werror -pedantic -ggdb
PROGRAMS = kmean_color_test hw5 mnist_kmeans

all : $(PROGRAMS)

Color.o : Color.cpp Color.h
	mpic++ $(CPPFLAGS) $< -c -o $@

kmean_color_test.o : kmean_color_test.cpp Color.h ColorKMeans.h KMeans.h
	mpic++ $(CPPFLAGS) $< -c -o $@

kmean_color_test : kmean_color_test.o Color.o
	mpic++ $(CPPFLAGS) kmean_color_test.o Color.o -o $@

run_sequential : kmean_color_test
	./kmean_color_test

hw5.o : hw5.cpp Color.h ColorKMeansMPI.h KMeansMPI.h
	mpic++ $(CPPFLAGS) $< -c -o $@

hw5 : hw5.o Color.o
	mpic++ $(CPPFLAGS) hw5.o Color.o -o $@

run_hw5 : hw5
	mpirun -n 2 ./hw5

valgrind : hw5
	mpirun -n 2 valgrind ./hw5

bigger_test : hw5
	mpirun -n 10 ./hw5

# MNIST extra credit targets
mnist_kmeans.o : mnist_kmeans.cpp MnistKMeansMPI.h KMeansMPI.h
	mpic++ $(CPPFLAGS) $< -c -o $@

mnist_kmeans : mnist_kmeans.o
	mpic++ $(CPPFLAGS) mnist_kmeans.o -o $@

run_mnist : mnist_kmeans
	mpirun -n 4 ./mnist_kmeans

clean :
	rm -f $(PROGRAMS) Color.o kmean_color_test.o hw5.o mnist_kmeans.o
	rm -f *.html