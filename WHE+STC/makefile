all: Train_transT Test_transT
Train_transT: Train_transT.cpp
	g++ Train_transT.cpp -o Train_transT -O2 -lpthread
Test_transT: Test_transT.cpp
	g++ Test_transT.cpp -o Test_transT -O2 -lpthread
clean:
	rm -rf Train_transT Test_transT