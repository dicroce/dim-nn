
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
            for (size_t i = 0; i < y.size(); ++i) {
                dy[i] = y[i] * (1 - y[i]);
                for (size_t j = 0; j < y.size(); ++j) {
                    if (i != j) {
                        dy[i] -= y[i] * y[j];
                    }
                }
            }
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
};

template<typename T>
class layer {
public:
    std::vector<T> _neurons;
    std::vector<T> _unactivated_neurons;
    std::vector<T> _errors;
    std::vector<std::vector<T>> _weights;
    activation_function<T> _activation_func;

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

    // Weight initialization
    void initialize_weights(std::vector<T>& w, size_t input_size, size_t output_size)
    {
        T range = std::sqrt(6.0 / (input_size + output_size));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-range, range);

        for (size_t i = 0; i < w.size(); ++i)
            w[i] = dist(gen);
    }

    virtual void forward(const std::vector<T>& inputs) = 0;
    virtual void backward(const std::vector<T>& prev_layer_neurons, T learningRate) = 0;
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
        std::vector<T> prev_layer_output;
        for (auto& layer : layers) {
            layer->backward(prev_layer_output, learning_rate);
            prev_layer_output = layer->_neurons;
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

        std::vector<T> prev_layer_output = layers.front()->_neurons;
        for (auto& layer : layers) {
            if (_m.find(layer) == _m.end()) {
                _m[layer].resize(layer->_weights.size());
                _v[layer].resize(layer->_weights.size());
                for (size_t i = 0; i < layer->_weights.size(); ++i) {
                    _m[layer][i].resize(layer->_weights[i].size(), 0);
                    _v[layer][i].resize(layer->_weights[i].size(), 0);
                }
            }

            for (size_t i = 0; i < layer->_weights.size(); ++i) {
                auto& w = layer->_weights[i];
                auto& m = _m[layer][i];
                auto& v = _v[layer][i];

                // Update weights
                for (size_t j = 0; j < prev_layer_output.size(); ++j) {
                    T g = layer->_errors[i] * prev_layer_output[j];
                    // ... (rest of the weight update code) ...
                }

                // Update bias weight
                T bias_g = layer->_errors[i];
                size_t bias_idx = w.size() - 1;
                m[bias_idx] = _beta1 * m[bias_idx] + (1 - _beta1) * bias_g;
                v[bias_idx] = _beta2 * v[bias_idx] + (1 - _beta2) * bias_g * bias_g;
                T m_hat = m[bias_idx] / (1 - std::pow(_beta1, _t));
                T v_hat = v[bias_idx] / (1 - std::pow(_beta2, _t)); 
                w[bias_idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + _epsilon);
            }

            layer->backward(prev_layer_output, learning_rate);
            prev_layer_output = layer->_neurons;
        }
    }

private:
    T _beta1; 
    T _beta2;
    T _epsilon;
    size_t _t;
    std::unordered_map<std::shared_ptr<layer<T>>, std::vector<std::vector<T>>> _m;
    std::unordered_map<std::shared_ptr<layer<T>>, std::vector<std::vector<T>>> _v;
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
        std::vector<T> derivatives(this->_neurons.size());
        this->_activation_func.derivative(this->_unactivated_neurons, derivatives);

        for (size_t i = 0; i < this->_neurons.size(); ++i)
        {
            for (size_t j = 0; j < prev_layer_neurons.size(); ++j)
                this->_weights[i][j] -= learningRate * this->_errors[i] * derivatives[i] * prev_layer_neurons[j];
            this->_weights[i].back() -= learningRate * this->_errors[i] * derivatives[i]; // Update bias
        }
    }

    void calculate_errors(const std::vector<T>& target_or_next_layer_errors, const std::vector<std::vector<T>>& next_layer_weights, bool is_output_layer)
    {
        if (is_output_layer)
        {
            for (size_t i = 0; i < this->_neurons.size(); ++i)
                this->_errors[i] = this->_neurons[i] - target_or_next_layer_errors[i];
        }
        else
        {
            std::vector<T> next_errors(target_or_next_layer_errors.size());
            for (size_t i = 0; i < this->_neurons.size(); ++i)
            {
                this->_errors[i] = 0;
                for (size_t j = 0; j < target_or_next_layer_errors.size(); ++j)
                    this->_errors[i] += target_or_next_layer_errors[j] * next_layer_weights[j][i];
                this->_errors[i] *= this->_unactivated_neurons[i] > T(0) ? T(1) : T(0.01); // Adjust for leaky ReLU derivative
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
#if 1
    // Create a neural network
    neural_network<float> nn(0.001, loss_functions<float>::mse, std::make_unique<adam_optimizer<float>>());

    // Define the network architecture
    auto hiddenLayer1 = std::make_shared<dense_layer<float>>(10, 4, activation_functions<float>::leaky_relu);
    auto hiddenLayer2 = std::make_shared<dense_layer<float>>(10, 10, activation_functions<float>::leaky_relu);
    auto outputLayer = std::make_shared<dense_layer<float>>(3, 10, activation_functions<float>::softmax);

    // Add layers to the network
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(hiddenLayer1));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(hiddenLayer2));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(outputLayer));

    // Train the network
    int numEpochs = 35000;
    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        float totalLoss = 0.0f;
        for (const auto& iris : irises)
        {
            // Prepare the input features
            std::vector<float> inputs = {iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width};

            // Prepare the expected output (one-hot encoded)
            std::vector<float> expectedOutput(3, 0.0f);
            int speciesIndex = static_cast<int>(iris.species) - 1;
            expectedOutput[speciesIndex] = 1.0f;

            // Forward pass
            auto outputLayer = nn.forward(inputs);

            // Calculate loss
            auto loss = nn.compute_loss(outputLayer, expectedOutput);
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

    shuffle(begin(irises), end(irises), default_random_engine {});

    // Test the trained network
    for (const auto& iris : irises)
    {
        // Prepare the input features
        std::vector<float> inputs = {iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width};

        // Forward pass
        auto outputLayer = nn.forward(inputs);

        // Print the predicted probabilities
        printf("Input: %.1f, %.1f, %.1f, %.1f, %.1f\n", iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width, iris.species);
        printf("Predicted probabilities: ");

        // Remember, the output uses one hot encoding (so 3 element array where the position of the value corresponds to species:
        //   (position 0 == species 1.0, position 1 == species 2.0, position 2 == species 3.0))
        for (const auto& prob : outputLayer)
        {
            printf("%.4f ", prob);
        }
        printf("\n");
    }

#else
    // Learn the XOR function

    // Train a neural network to learn the XOR function
    neural_network<float> nn(0.001, loss_functions<float>::mse, std::make_unique<adam_optimizer<float>>());

    auto hiddenLayer1 = std::make_shared<dense_layer<float>>(4, 2, activation_functions<float>::relu);
    auto hiddenLayer2 = std::make_shared<dense_layer<float>>(4, 4, activation_functions<float>::relu);
    auto outputLayer = std::make_shared<dense_layer<float>>(1, 4, activation_functions<float>::relu);

    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(hiddenLayer1));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(hiddenLayer2));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(outputLayer));

    for(int i = 0; i < 500000; ++i)
    {
        vector<float> inputs = {(float)(rand() % 2), (float)(rand() % 2)};
        vector<float> expected = {(float)(inputs[0] != inputs[1])};

        auto output_layer = nn.forward(inputs);
        auto loss = nn.compute_loss(output_layer, expected);

        if (i % 1000 == 0)
            printf("Iteration: %d, Loss: %f\n", i, loss);

        nn.backpropagate(inputs, expected);
    }
#endif
    return 0;
}
