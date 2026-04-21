#include <iostream>
#include <torch/torch.h>

// Define XOR neural network module
struct XORNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    XORNet() {
        // Input: 2 → Hidden: 4
        fc1 = register_module("fc1", torch::nn::Linear(2, 128));

        // Hidden: 4 → Output: 1
        fc2 = register_module("fc2", torch::nn::Linear(128, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::sigmoid(fc2->forward(x));
        return x;
    }
};

int main() {
    // Fix randomness for reproducibility
    torch::manual_seed(0);

    // Select device (CPU fallback if CUDA unavailable)
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);

    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU")
              << "\n";

    XORNet model;
    model.to(device);

    // XOR dataset
    auto inputs =
        torch::tensor({{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}})
            .to(device);

    auto targets = torch::tensor({{0.0}, {1.0}, {1.0}, {0.0}}).to(device);

    // Optimizer
    torch::optim::SGD optimizer(model.parameters(),
                                torch::optim::SGDOptions(0.1));

    // Training loop
    const int epochs = 2000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();

        auto output = model.forward(inputs);

        auto loss = torch::binary_cross_entropy(output, targets);

        loss.backward();
        optimizer.step();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss.item<float>()
                      << "\n";
        }
    }

    // Evaluation
    std::cout << "\nPredictions:\n";
    auto preds = model.forward(inputs).to(torch::kCPU);

    std::cout << preds << "\n";

    // Rounded results (more interpretable)
    std::cout << "\nRounded:\n";
    std::cout << torch::round(preds) << "\n";

    return 0;
}