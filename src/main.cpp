#include <iostream>
#include <string>
#include <array>
#include "ntars/models/DenseNetwork.hpp"
#include "mnist/mnist_reader.hpp"

#include "../config.h.in"

int main() 
{
    NTARS::DenseNeuralNetwork network{{784, 128, 64, 10}};

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

        std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
        std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
        std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
        std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;


    return 0;
}
