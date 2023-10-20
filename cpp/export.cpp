#include <vector>
#include <iostream>
#include <fstream>

#include "engine.h"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " core_model.onnx engine.plan" << endl;
    }

	ifstream onnxFile;
	onnxFile.open(argv[1], ios::in | ios::binary);
	cout << "Load model from " << argv[1] << endl;

    if (!onnxFile.good()) {
		cerr << "\nERROR: Unable to read specified ONNX model " << argv[1] << endl;
		return -1;
	}

	onnxFile.seekg (0, onnxFile.end);
	size_t size = onnxFile.tellg();
	onnxFile.seekg (0, onnxFile.beg);

	auto *buffer = new char[size];
	onnxFile.read(buffer, size);
	onnxFile.close();

	bool verbose = false;
	size_t workspace_size =(1ULL << 28);
	const vector<int> dynamic_batch_opts{1, 8, 16};
	float score_thresh = 0.1f;

	cout << "Building engine..." << endl;
	auto engine = mobilenet::Engine(buffer, size, dynamic_batch_opts,
		score_thresh, verbose, workspace_size);
	engine.save(string(argv[2]));

	delete [] buffer;
	return 0;
}