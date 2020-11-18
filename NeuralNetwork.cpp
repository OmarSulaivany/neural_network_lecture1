#include "NeuralNetwork.h"
#include <iostream>
#include <time.h>
#include <cmath>
#include <omp.h>

using namespace std;

NeuralNetwork::NeuralNetwork()
{
	srand(static_cast <unsigned> (time(0)));

	this->w1 = -2 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 - (-2))));
	this->w2 = -2 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 - (-2))));
	this->b = -2 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 - (-2))));
}

float** NeuralNetwork::inputs()
{
	int rows=10, columns=3; 

	float data[10][3] = {
		{3,1.5,         1},
		{4, 1.5,        1},
		{2, 1,          0},
		{3, 0,          0},
		{2, 2,          1},
		{3, 1,          1},
		{2, 0.5,        0},
		{0, 2,          0},
		{5.5, 1,        1},
		{1, 1,          0} } ;

			
	this->m_dataset = new float* [rows];
	for (int i = 0; i < rows; i++)
	{
		this->m_dataset[i] = new float[columns];
		for (int j = 0; j < columns; j++)
		{
			this->m_dataset[i][j] = data[i][j];
			//cout << this->m_dataset[i][j] << " ";
		}
		//cout << endl;
		
	}
	cout <<"\nWeights and biases before the training \n";
	cout <<"\n\nW1 = " << this->w1<<endl<<"W2 = "<<this->w2 << endl << "b= " << this->b << endl;
	

	return this->m_dataset;
}

 float NeuralNetwork::sigmoid( float z)
{
	return 1 / (1 + exp(-z));

}

float NeuralNetwork::sigmoid_prime(float z)
{
	return sigmoid(z) * (1 - sigmoid(z));
}


void NeuralNetwork::train()
{
	float learning_rate = 0.1;
	float z,pred,target,loss,dloss_dpred, dpred_dz, dz_dw1, dz_dw2, dz_db, dloss_dz, dloss_dw1, dloss_dw2, dloss_db;

	for(unsigned epoches=0; epoches<100; epoches++) // number of iterations
	for (int i = 0; i < 10; i++)
	{
		z = (this->m_dataset[i][0] * this->w1 + this->m_dataset[i][1]*this->w2) + this->b;
		pred = sigmoid(z);                  //network prediction	
		target = this->m_dataset[i][2];      //Target
		loss = pow((pred - target),2);     //loss "cost function"
	
		if (epoches > 98)
		{
			cout << "Input1 : " << this->m_dataset[i][0] << endl <<"Input2 : "<< this->m_dataset[i][1] << endl;
			cout << "Z= " << z << endl;
			
			cout << "Pred= " << pred <<endl;
			
			cout << "Target= " << target<<endl;
			
			cout << "Loss= "<<loss <<endl<<endl;
		}

		dloss_dpred = 2 * (pred - target); //Derivative of loss with respect to pred.

		dpred_dz = sigmoid_prime(z);       //Derivative of pred with respect to Z.

		
			dz_dw1 = this->m_dataset[i][0]; // Derivative of Z with respect to weight1.
			dz_dw2 = this->m_dataset[i][1]; // Derivative of Z with respect to weight2.
			dz_db = 1;                      // Derivative of Z with respect to bias.

		dloss_dz = dloss_dpred * dpred_dz; // Derivative of loss with respect to Z.

		dloss_dw1 = dloss_dz * dz_dw1; // Derivative of loss with respect to weight1.
		dloss_dw2 = dloss_dz * dz_dw2; // Derivative of loss with respect to weight2.
		
		dloss_db = dloss_dz * dz_db; // Derivative of loss with respect to bias.

		this->w1 = this->w1 - learning_rate * dloss_dw1; //update the weight by a small fraction "Alpha" multiply derivative of loss w.r.t w1.
		this->w2 = this->w2 - learning_rate * dloss_dw2; //update the weight by a small fraction "Alpha" multiply derivative of loss w.r.t w2.	
		this->b = this->b - learning_rate * dloss_db;    //update the bias by a small fraction "Alpha" multiply derivative of loss w.r.t b.

	}
	cout << "\nWeights and biases after the training \n";
	cout << "W1 = " << this->w1 <<endl<<"W2 = "<<this->w2 <<endl << "b= " << this->b << endl<<endl;

	
}

void NeuralNetwork::output()
{
	float out, prediction,z;

	float x1 = 3.5;
	float x2 = 1;
	z = (this->w1 * x1 + this->w1 * x2) + this->b;

	out = sigmoid(z);

	//cout << "Real       = " << m_input_array[18][1]<<endl;
	cout << "Output = " << out << endl<<endl;

	prediction = round(out);
	cout << "Predicted = " << prediction << endl;

}