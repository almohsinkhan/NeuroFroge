#pragma once

#include "module.h"
#include <vector>
#include <memory>
#include <utility>
#include <iostream>
#include <stdexcept>

class Sequential{
    private:
        // own and store each layer in the model
        std::vector<std::unique_ptr<Module>> layers;

        // store the input for each layer
        std::vector<Tensor> inputs;

        //each layer need original input currently support backward() implementation
        // so sequential much store the input for each layer 

    public:
        size_t size() const {
            return layers.size();
        }

        // add a layer to the model
        template <typename T, typename... Args>
        void add(Args&&... args) {
            layers.push_back(
                std::make_unique<T>(std::forward<Args>(args)...)
            );
        }

        Tensor forward(const Tensor& input){
            // remove stored inputs from previous forward pass
            inputs.clear();

            Tensor output = input;

            for (auto& layer : layers){
                inputs.push_back(output);
                output = layer->forward(output);
            }

            return output;
        }

        Tensor backward(const Tensor& grad_output){
            if (inputs.size() != layers.size()){
                throw std::runtime_error(
                    "Sequential backward() called before forward()"
                );
            }
            Tensor grad = grad_output;

            for (size_t i = layers.size(); i-- > 0;){
                grad = layers[i]->backward(inputs[i], grad);
            }   
            return grad;
        }

        void zero_grad(){
            for (auto& layer : layers){
                layer->zero_grad();
            }
        }

        void update(double learning_rate, int batch_size){
            for (auto& layer : layers){
                layer->update(learning_rate, batch_size);
            }
        }

        void print_summary() const {
            std::cout << "Sequential Model\n";
            std::cout << "Layers: " << layers.size() << "\n";

            for (size_t i = 0; i < layers.size(); i++){
                std::cout << i << ": " 
                          << layers[i]->name() 
                          << "\n";    
            }
        }
};