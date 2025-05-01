#ifndef NTARS_DATA_HPP
#define NTARS_DATA_HPP

#include <vector>
#include <string>
#include <variant>
#include <fstream>
#include "json/json.hpp"
#include <iostream>

namespace NTARS
{
    namespace DATA
    {
        template<typename T>
        struct TrainingData
        {
            T data;
            std::vector<float> label;
        };

        template<typename T>
        bool saveDataJSON(TrainingData<T>& inData, std::string name)
        {
            nlohmann::json saved;
            std::filesystem::path outputPath = std::filesystem::current_path() / "data";     
            
            saved["data"] = inData.data;
            saved["label"] = inData.label;
        
            if (!std::filesystem::exists(outputPath))
            {
                std::filesystem::create_directory(outputPath);
            }

            std::ofstream outFile(outputPath / std::string(name + ".json"));

            if (outFile.is_open())
            {
                outFile << saved.dump(); 
                outFile.close();
                std::cout << "TrainingData saved successfully: " << outputPath << std::endl;
                return true;
            }
            else
            {
                std::cerr << "Could not open file for writing: " << outputPath << std::endl;
            }

            return false;
        }

        template<typename T>
        bool saveListDataJSON(std::vector<TrainingData<T>>& inData, std::string name)
        {
            nlohmann::json saved;

            std::filesystem::path outputPath = std::filesystem::current_path() / "data";     

            for (const auto& data : inData)
            {
                nlohmann::json savedData;

                savedData["data"] = data.data;
                savedData["label"] = data.label;
                saved.push_back(savedData);
            }

            if (!std::filesystem::exists(outputPath))
            {
                std::filesystem::create_directory(outputPath);
            }

            std::ofstream outFile(outputPath / std::string(name + ".json"));

            if (outFile.is_open())
            {
                outFile << saved.dump(); 
                outFile.close();
                std::cout << "TrainingData saved successfully: " << outputPath << std::endl;
                return true;
            }
            else
            {
                std::cerr << "Could not open file for writing: " << outputPath << std::endl;
            }

            return false;
        }

        template<typename T>
        std::vector<TrainingData<T>> loadDataListJSON(const std::string path)
        {
            std::filesystem::path inputPath = std::filesystem::current_path() / "data" / (path + ".json");

            std::ifstream inFile(inputPath);
            
            if (!inFile.is_open())
            {
                std::cerr << "Could not open file for reading: " << inputPath << std::endl;
                return {};
            }
        
            nlohmann::json saved;
            inFile >> saved;
            inFile.close();
        
            std::vector<TrainingData<T>> dataList{};
        
            for (const auto& entry : saved)
            {
                TrainingData<T> data{};
        
                if constexpr (std::is_same_v<T, std::vector<float>>)
                {
                    data.data = entry["data"].get<std::vector<float>>(); 
                    data.label = entry["label"].get<std::vector<float>>(); 
                }
                else
                {
                    data.data = entry["data"].get<T>(); 
                    data.label = entry["label"].get<T>(); 
                }
        
                dataList.push_back(std::move(data));
            }
        
            return dataList;
        }

    } // namespace DATA
} // namespace NTARS

#endif // NTARS_DATA_HPP