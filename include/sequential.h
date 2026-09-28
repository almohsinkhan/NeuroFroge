#pragma once

#include "module.h"
#include <vector>

class Sequential{
    private:
        // store a pointer to each layer in the model
        std::vector<Module*> layers;

        // store the input for each layer
        std::vector<Tensor> inputs;

        //each layer need original input currently support backward() implementation
        // so sequential much store the input for each layer 

    public:
        // add a layer to the model
        void add(Module* layer){
            layers.push_back(layer);
        }

        Tensor forward(const Tensor& input){
            // remove stored inputs from previous forward pass
            inputs.clear();

            Tensor output = input;

            for (Module* layer : layers){
                inputs.push_back(output);
                output = layer->forward(output);
            }

            return output;
        }

        Tensor backward(const Tensor& grad_output){
            Tensor grad = grad_output;

            for (int i =  layers.size() - 1; i >= 0; i--){
                grad = layers[i]->backward(inputs[i], grad);
            }   
            return grad;
        }

        void zero_grad(){
            for (Module* layer : layers){
                layer->zero_grad();
            }
        }

        void update(double learning_rate, int batch_size){
            for (Module* layer : layers){
                layer->update(learning_rate, batch_size);
            }
        }
};