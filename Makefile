all:
	c++ BMP_test.cpp BMP.cpp -Xpreprocessor -fopenmp -lomp -mpopcnt -O3 -o ./bmp

clean:
	rm -rf *.gch *.o out
