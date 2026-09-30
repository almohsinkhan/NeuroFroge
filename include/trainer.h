#pragma once

#include "sequential.h"
#include "dataset.h"
#include "loss.h"

class Trainer {
    private:
        Sequential& model;
        Dataset& dataset;
        Loss& loss;

    public:
        Trainer(Sequential& model,
                Dataset& dataset,
                Loss& loss);

        void fit(int epochs,
                 int batch_size,
                 double learning_rate,
                int train_size);
        
};