#pragma once

#include "sequential.h"
#include "dataset.h"
#include "dataloader.h"
#include "loss.h"

class Trainer {
    private:
        Sequential& model;
        Dataset& dataset;
        Loss& loss;
        BatchFormat format;

    public:
        Trainer(Sequential& model,
                Dataset& dataset,
                Loss& loss,
                BatchFormat format = BatchFormat::MLP);

        void fit(int epochs,
                 int batch_size,
                 double learning_rate,
                int train_size);
        
};