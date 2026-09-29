#pragma once

#include "sequential.h"
#include "dataset.h"

class Trainer {
    private:
        Sequential& model;
        Dataset& dataset;

    public:
        Trainer(Sequential& model, Dataset& dataset);

        void fit(int epochs,
                 int batch_size,
                 double learning_rate,
                int train_size);

        
};