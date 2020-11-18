#ifndef NN_H
#define NN_H
class NeuralNetwork
{
public:
	NeuralNetwork();
	float** inputs();
    float sigmoid(float z);
	float sigmoid_prime(float z);
	void train();
	void output();
private:
	float w1,w2, b;
	float** m_dataset = nullptr;
};
#endif // !NN_H

