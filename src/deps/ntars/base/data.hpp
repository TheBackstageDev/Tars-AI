#ifndef NTARS_DATA_HPP
#define NTARS_DATA_HPP

#include <vector>
#include <string>
#include <variant>
#include "ntars/models/DenseNetwork.hpp"

namespace NTARS
{
    namespace DATA
    {
        template<typename T>
        struct TrainingData
        {
            uint32_t id{++lastId};
            T data;
            std::vector<std::string> labels;
        
        private:
            static inline uint32_t lastId{0};
        };

        class DataManager
        {
        public:
            using DataType = std::variant<TrainingData<double>, TrainingData<std::vector<double>>, TrainingData<std::string>>;

            void addData(const DataType& newData)
            {
                data_.emplace_back(newData);
            }

            inline std::vector<DataType> getData() const { return data_; }
            inline DataType& getDataFromId(uint32_t id) { return data_[id]; }
            inline void removeData(uint32_t id) {
                data_.erase(std::remove_if(data_.begin(), data_.end(), [id](const DataType& data) {
                    return std::visit([id](const auto& d) {
                        return d.id == id;
                    }, data);
                }), data_.end());
            }
            
        private:
            std::vector<DataType> data_;
        };

        void backpropagation(DenseNeuralNetwork& network, DataManager& manager, double learningRate = 0.1f)
        {

        }
    } // namespace DATA
} // namespace NTARS

#endif // NTARS_DATA_HPP