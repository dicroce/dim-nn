// dim-nn
//
// Q: Why is it dim?
// A: Because it's not that bright.
//
// Q: What is it good for?
// A: Learning about neural networks and backpropagation.
//
// Q: What is it not good for?
// A: Anything else.
//
// Q: Will it ever be good for anything else?
// A: Probably not. If I ever learn enough about them I would like to implement a convolutional layer.
//
// Q: What is implemented?
// A: relu, leaky-relu, softmax, dense networks, adam & regular gradient descent.
//
// Q: Is it fast?
// A: No. But it does use std::inner_product() which is something I always wanted to do since I found out it existed.

#include <vector>
#include <array>
#include <numeric>
#include <iostream>
#include <functional>
#include <random>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include "iris.h"

using namespace std;

template<typename T>
struct activation_function
{
    std::function<void(std::vector<T>&)> activation;
    std::function<void(const std::vector<T>&, std::vector<T>&)> derivative;
};

template<typename T>
struct activation_functions
{
    inline static const activation_function<T> relu = {
        [](std::vector<T>& x) {
            for (auto& val : x)
                val = (val > T(0)) ? val : T(0);
        },
        [](const std::vector<T>& x, std::vector<T>& dx) {
            dx.resize(x.size());
            for (size_t i = 0; i < x.size(); ++i)
                dx[i] = T(x[i] > 0);
        }
    };

    inline static const activation_function<T> leaky_relu = {
        [](std::vector<T>& x) {
            for (auto& val : x)
                val = (val > T(0)) ? val : T(0.01) * val;
        },
        [](const std::vector<T>& x, std::vector<T>& dx) {
            dx.resize(x.size());
            for (size_t i = 0; i < x.size(); ++i)
                dx[i] = (x[i] > T(0)) ? T(1) : T(0.01);
        }
    };

    inline static const activation_function<T> softmax = {
        [](std::vector<T>& x) {
            T max_val = *std::max_element(x.begin(), x.end());
            std::vector<T> exp_values(x.size());
            T sum_exp = 0;
            for (size_t i = 0; i < x.size(); ++i) {
                exp_values[i] = std::exp(x[i] - max_val);
                sum_exp += exp_values[i];
            }
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = exp_values[i] / sum_exp;
            }
        },
        [](const std::vector<T>& y, std::vector<T>& dy) {
            dy.resize(y.size());
            // Note: For cross-entropy loss with softmax, derivative simplifies and is handled in loss function
        }
    };
};

template<typename T>
struct loss_function
{
    std::function<T(const std::vector<T>&, const std::vector<T>&)> compute_loss;
    std::function<std::vector<T>(const std::vector<T>&, const std::vector<T>&)> derivative;
};

template<typename T>
struct loss_functions {
    inline static const loss_function<T> mse =
    {
        // MSE Loss
        [](const std::vector<T>& outputs, const std::vector<T>& targets)
        {
            T sum = 0;
            for (size_t i = 0; i < outputs.size(); ++i)
            {
                T diff = outputs[i] - targets[i];
                sum += diff * diff;
            }
            return sum / outputs.size();
        },
        // Derivative of MSE
        [](const std::vector<T>& outputs, const std::vector<T>& targets)
        {
            std::vector<T> result(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i)
                result[i] = 2 * (outputs[i] - targets[i]) / outputs.size();
            return result;
        }
    };

    inline static const loss_function<T> cross_entropy =
    {
        // Cross-Entropy Loss
        [](const std::vector<T>& outputs, const std::vector<T>& targets)
        {
            T sum = 0;
            for (size_t i = 0; i < outputs.size(); ++i)
            {
                sum -= targets[i] * std::log(outputs[i] + std::numeric_limits<T>::epsilon());
            }
            return sum / outputs.size();
        },
        // Derivative of Cross-Entropy Loss
        [](const std::vector<T>& outputs, const std::vector<T>& targets)
        {
            std::vector<T> result(outputs.size());
            for (size_t i = 0; i < outputs.size(); ++i)
                result[i] = (outputs[i] - targets[i]) / outputs.size();
            return result;
        }
    };
};

template<typename T>
class layer {
public:
    std::vector<T> _neurons;
    std::vector<T> _unactivated_neurons;
    std::vector<T> _errors;
    std::vector<std::vector<T>> _weights;
    activation_function<T> _activation_func;
    std::vector<T> _inputs; // Store inputs for use in backward pass

    layer(size_t num_neurons, size_t inputs_per_neuron, activation_function<T> activation = activation_functions<T>::relu) :
        _activation_func(activation)
    {
        _neurons.resize(num_neurons);
        _unactivated_neurons.resize(num_neurons);
        _errors.resize(num_neurons);
        _weights.resize(num_neurons);

        for (auto& w : _weights)
        {
            w.resize(inputs_per_neuron + 1); // +1 for bias
            initialize_weights(w, inputs_per_neuron, num_neurons);
        }
    }

    // Weight initialization with fixed seed for reproducibility
    void initialize_weights(std::vector<T>& w, size_t input_size, size_t output_size)
    {
        static std::mt19937 gen(42); // Fixed seed
        T range = std::sqrt(6.0 / (input_size + output_size));
        std::uniform_real_distribution<T> dist(-range, range);

        for (size_t i = 0; i < w.size(); ++i)
            w[i] = dist(gen);
    }

    virtual void forward(const std::vector<T>& inputs) = 0;
    virtual void backward(const std::vector<T>& prev_layer_neurons, T learningRate) = 0;
    virtual void backward(const std::vector<T>& prev_layer_neurons, T learningRate,
                          std::vector<std::vector<T>>& m, std::vector<std::vector<T>>& v,
                          size_t t, T beta1, T beta2, T epsilon) = 0;
};

template<typename T>
class optimizer {
public:
    virtual void update_weights(std::vector<std::shared_ptr<layer<T>>>& layers, T learning_rate) = 0;
};

template<typename T>
class sgd_optimizer : public optimizer<T> {
public:
    void update_weights(std::vector<std::shared_ptr<layer<T>>>& layers, T learning_rate) override {
        for (size_t idx = 0; idx < layers.size(); ++idx) {
            auto& layer = layers[idx];
            auto prev_layer = idx > 0 ? layers[idx - 1] : nullptr;
            layer->backward(prev_layer ? prev_layer->_neurons : std::vector<T>(), learning_rate);
        }
    }
};

template<typename T>
class adam_optimizer : public optimizer<T> {
public:
    adam_optimizer(T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : _beta1(beta1), _beta2(beta2), _epsilon(epsilon), _t(0)
    {
    }

    void update_weights(std::vector<std::shared_ptr<layer<T>>>& layers, T learning_rate) override {
        ++_t;

        for (size_t idx = 0; idx < layers.size(); ++idx) {
            auto& layer = layers[idx];
            auto prev_layer = idx > 0 ? layers[idx - 1] : nullptr;

            if (_m.find(layer.get()) == _m.end()) {
                _m[layer.get()].resize(layer->_weights.size());
                _v[layer.get()].resize(layer->_weights.size());
                for (size_t i = 0; i < layer->_weights.size(); ++i) {
                    _m[layer.get()][i].resize(layer->_weights[i].size(), 0);
                    _v[layer.get()][i].resize(layer->_weights[i].size(), 0);
                }
            }

            layer->backward(prev_layer ? prev_layer->_neurons : std::vector<T>(), learning_rate,
                            _m[layer.get()], _v[layer.get()], _t, _beta1, _beta2, _epsilon);
        }
    }

private:
    T _beta1;
    T _beta2;
    T _epsilon;
    size_t _t;
    std::unordered_map<layer<T>*, std::vector<std::vector<T>>> _m;
    std::unordered_map<layer<T>*, std::vector<std::vector<T>>> _v;
};

template<typename T>
class dense_layer : public layer<T> {
public:
    dense_layer(size_t num_neurons, size_t inputs_per_neuron, activation_function<T> activation = activation_functions<T>::relu) :
        layer<T>(num_neurons, inputs_per_neuron, activation)
    {
    }

    void forward(const std::vector<T>& inputs) override
    {
        this->_inputs = inputs; // Store inputs for backward pass
        for (size_t i = 0; i < this->_neurons.size(); ++i) {
            // Compute the dot product of weights and inputs
            this->_unactivated_neurons[i] = std::inner_product(inputs.begin(), inputs.end(), this->_weights[i].begin(), T(0));
            // Add the bias term
            this->_unactivated_neurons[i] += this->_weights[i].back();
        }
        this->_activation_func.activation(this->_unactivated_neurons);
        this->_neurons = this->_unactivated_neurons;
    }

    void backward(const std::vector<T>& prev_layer_neurons, T learningRate) override
    {
        for (size_t i = 0; i < this->_neurons.size(); ++i)
        {
            for (size_t j = 0; j < prev_layer_neurons.size(); ++j)
                this->_weights[i][j] -= learningRate * this->_errors[i] * prev_layer_neurons[j];
            this->_weights[i].back() -= learningRate * this->_errors[i]; // Update bias
        }
    }

    void backward(const std::vector<T>& prev_layer_neurons, T learningRate,
                  std::vector<std::vector<T>>& m, std::vector<std::vector<T>>& v,
                  size_t t, T beta1, T beta2, T epsilon) override
    {
        for (size_t i = 0; i < this->_neurons.size(); ++i)
        {
            for (size_t j = 0; j < prev_layer_neurons.size(); ++j)
            {
                T g = this->_errors[i] * prev_layer_neurons[j];
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g;
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g;

                T m_hat = m[i][j] / (1 - std::pow(beta1, t));
                T v_hat = v[i][j] / (1 - std::pow(beta2, t));

                this->_weights[i][j] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
            }

            // Bias term
            T g = this->_errors[i];
            m[i].back() = beta1 * m[i].back() + (1 - beta1) * g;
            v[i].back() = beta2 * v[i].back() + (1 - beta2) * g * g;

            T m_hat = m[i].back() / (1 - std::pow(beta1, t));
            T v_hat = v[i].back() / (1 - std::pow(beta2, t));

            this->_weights[i].back() -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

    void calculate_errors(const std::vector<T>& target_or_next_layer_errors, const std::vector<std::vector<T>>& next_layer_weights, bool is_output_layer)
    {
        if (is_output_layer)
        {
            // For cross-entropy loss with softmax activation, the error is outputs - targets
            for (size_t i = 0; i < this->_neurons.size(); ++i)
                this->_errors[i] = this->_neurons[i] - target_or_next_layer_errors[i];
        }
        else
        {
            for (size_t i = 0; i < this->_neurons.size(); ++i)
            {
                this->_errors[i] = 0;
                for (size_t j = 0; j < target_or_next_layer_errors.size(); ++j)
                    this->_errors[i] += target_or_next_layer_errors[j] * next_layer_weights[j][i];
                // Apply derivative of activation function
                T derivative = this->_unactivated_neurons[i] > T(0) ? T(1) : T(0.01); // Leaky ReLU derivative
                this->_errors[i] *= derivative;
            }
        }
    }
};

template<typename T = float>
class neural_network {
public:
    neural_network(float learning_rate, loss_function<T> loss_function = loss_functions<T>::mse, std::unique_ptr<optimizer<T>> optimizer = std::make_unique<sgd_optimizer<T>>()) :
        _learning_rate(learning_rate),
        _loss_function(loss_function),
        _optimizer(std::move(optimizer))
    {
    }

    void add_layer(std::shared_ptr<layer<T>> layer) {_layers.push_back(layer);}

    std::vector<T> forward(const std::vector<T>& inputs)
    {
        std::vector<T> layer_input = inputs;
        for (auto& layer : _layers)
        {
            layer->forward(layer_input);
            layer_input = layer->_neurons; // Output of current layer is input to next
        }
        return layer_input; // Output of last layer
    }

    T compute_loss(const std::vector<T>& outputs, const std::vector<T>& targets)
    {
        return _loss_function.compute_loss(outputs, targets);
    }

    void backpropagate(const std::vector<T>& inputs, const std::vector<T>& targets)
    {
        // Calculate errors for output layer
        auto output_layer = std::dynamic_pointer_cast<dense_layer<T>>(_layers.back());
        output_layer->calculate_errors(targets, {}, true);

        // Calculate errors for hidden layers
        for (long i = _layers.size() - 2; i >= 0; --i)
        {
            auto layer = std::dynamic_pointer_cast<dense_layer<T>>(_layers[i]);
            auto next_layer = std::dynamic_pointer_cast<dense_layer<T>>(_layers[i + 1]);

            layer->calculate_errors(next_layer->_errors, next_layer->_weights, false);
        }

        // Update weights using the optimizer
        _optimizer->update_weights(_layers, _learning_rate);
    }

private:
    float _learning_rate;
    std::vector<std::shared_ptr<layer<T>>> _layers;
    loss_function<T> _loss_function;
    std::unique_ptr<optimizer<T>> _optimizer;
};

int main(int argc, char* argv[])
{
    // Normalize the input data
    std::vector<float> sepal_lengths, sepal_widths, petal_lengths, petal_widths;
    for (const auto& iris : irises)
    {
        sepal_lengths.push_back(iris.sepal_length);
        sepal_widths.push_back(iris.sepal_width);
        petal_lengths.push_back(iris.petal_length);
        petal_widths.push_back(iris.petal_width);
    }

    auto min_max = [](const std::vector<float>& data) {
        auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
        return std::make_pair(*min_it, *max_it);
    };

    auto [sl_min, sl_max] = min_max(sepal_lengths);
    auto [sw_min, sw_max] = min_max(sepal_widths);
    auto [pl_min, pl_max] = min_max(petal_lengths);
    auto [pw_min, pw_max] = min_max(petal_widths);

    // Create a neural network
    neural_network<float> nn(0.01, loss_functions<float>::cross_entropy, std::make_unique<adam_optimizer<float>>());

    // Define the network architecture
    auto hiddenLayer1 = std::make_shared<dense_layer<float>>(10, 4, activation_functions<float>::leaky_relu);
    auto hiddenLayer2 = std::make_shared<dense_layer<float>>(10, 10, activation_functions<float>::leaky_relu);
    auto outputLayer = std::make_shared<dense_layer<float>>(3, 10, activation_functions<float>::softmax);

    // Add layers to the network
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(hiddenLayer1));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(hiddenLayer2));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(outputLayer));

    // Training parameters
    int numEpochs = 5000;

    // Fixed random seed for reproducibility
    std::default_random_engine rng(42);

    // Train the network
    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        float totalLoss = 0.0f;

        // Shuffle the dataset each epoch
        std::shuffle(irises.begin(), irises.end(), rng);

        for (const auto& iris : irises)
        {
            // Normalize the input features
            std::vector<float> inputs = {
                (iris.sepal_length - sl_min) / (sl_max - sl_min),
                (iris.sepal_width - sw_min) / (sw_max - sw_min),
                (iris.petal_length - pl_min) / (pl_max - pl_min),
                (iris.petal_width - pw_min) / (pw_max - pw_min)
            };

            // Prepare the expected output (one-hot encoded)
            std::vector<float> expectedOutput(3, 0.0f);
            int speciesIndex = static_cast<int>(iris.species) - 1;
            expectedOutput[speciesIndex] = 1.0f;

            // Forward pass
            auto outputs = nn.forward(inputs);

            // Calculate loss
            auto loss = nn.compute_loss(outputs, expectedOutput);
            totalLoss += loss;

            // Backward pass
            nn.backpropagate(inputs, expectedOutput);
        }

        // Print the average loss for every 100 epochs
        if ((epoch + 1) % 100 == 0)
        {
            float avgLoss = totalLoss / irises.size();
            printf("Epoch: %d, Loss: %.4f\n", epoch + 1, avgLoss);
        }
    }

    // Test the trained network
    for (const auto& iris : irises)
    {
        // Normalize the input features
        std::vector<float> inputs = {
            (iris.sepal_length - sl_min) / (sl_max - sl_min),
            (iris.sepal_width - sw_min) / (sw_max - sw_min),
            (iris.petal_length - pl_min) / (pl_max - pl_min),
            (iris.petal_width - pw_min) / (pw_max - pw_min)
        };

        // Forward pass
        auto outputs = nn.forward(inputs);

        // Find the predicted class
        auto max_it = std::max_element(outputs.begin(), outputs.end());
        int predicted_class = std::distance(outputs.begin(), max_it) + 1;

        // Print the results
        printf("Actual species: %d, Predicted species: %d, Probabilities: ", static_cast<int>(iris.species), predicted_class);
        for (const auto& prob : outputs)
        {
            printf("%.4f ", prob);
        }
        printf("\n");
    }

    return 0;
}
