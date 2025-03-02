#include <iostream>
#include <string>
#include <array>
#include "ntars/models/DenseNetwork.hpp"
#include "ntars/base/data.hpp"
#include "mnist/mnist_reader.hpp"

#include "../config.h"

int main() 
{
    NTARS::DenseNeuralNetwork network{{784, 128, 64, 10}};
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::vector<std::vector<NTARS::DATA::TrainingData<std::vector<double>>>> batches{};

    const size_t batch_size = 100;
    const size_t epochs = 1;
    const double learningRate = 0.05;

    const auto& data = dataset.training_images;
    for (size_t i = 0; i < data.size() / batch_size; ++i)
    {
        std::vector<NTARS::DATA::TrainingData<std::vector<double>>> miniBatch{};
        for (size_t j = 0; j < batch_size && (i + j) < data.size(); ++j)
        {
            NTARS::DATA::TrainingData<std::vector<double>> newData{};
            newData.data = std::vector<double>(data.at(i + j).begin(), data.at(i + j).end());
            newData.label = dataset.training_labels.at(i + j);
    
            miniBatch.emplace_back(std::move(newData));
        }
        batches.emplace_back(std::move(miniBatch));
    }

    double result;

/*     for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < batches.size(); ++i)
        {
            result = network.train(batches[i], learningRate);
        }
    }
 */

    result = network.train(batches.at(0), learningRate);
    std::vector<double> outputs = network.run(batches.at(0).at(0).data);

    std::cout << "Result (Rights / Total): " + std::to_string(result);

    return 0;
}
