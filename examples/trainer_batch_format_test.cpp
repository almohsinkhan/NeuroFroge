#include "cross_entropy.h"
#include "dataset.h"
#include "flatten.h"
#include "linear.h"
#include "sequential.h"
#include "trainer.h"

int main() {
    Dataset dataset = Dataset::fromIDX(
        "data/FashionMNIST/raw/train-images-idx3-ubyte",
        "data/FashionMNIST/raw/train-labels-idx1-ubyte"
    );

    CrossEntropyLoss loss;

    Sequential mlp;
    mlp.add<Linear>(784, 10);
    Trainer mlp_trainer(
        mlp,
        dataset,
        loss,
        BatchFormat::MLP
    );
    mlp_trainer.fit(1, 2, 0.001, 2);

    Sequential cnn_input_path;
    cnn_input_path.add<Flatten>();
    cnn_input_path.add<Linear>(784, 10);
    Trainer cnn_trainer(
        cnn_input_path,
        dataset,
        loss,
        BatchFormat::CNN
    );
    cnn_trainer.fit(1, 2, 0.001, 2);
}
