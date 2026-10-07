#include <iostream>

#include "dataset.h"
#include "dataloader.h"

int main() {

    Dataset dataset(
        "data/images.csv",
        "data/labels.csv"
    );

    DataLoader loader(dataset, 32);

    int batch_number = 0;

    while (loader.hasNext()) {

        Batch batch = loader.next();

        std::cout << "Batch " << batch_number + 1 << "\n";
        std::cout << "Images shape: ";
        batch.images.printShape();

        std::cout << "Labels: "
                  << batch.labels.size()
                  << "\n\n";

        batch_number++;
    }

    return 0;
}