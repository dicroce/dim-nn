
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

#define WEIGHT_RANDOM_RANGE .1

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
    std::vector<T> neurons;
    std::vector<T> unactivated_neurons;
    std::vector<T> errors;
    std::vector<std::vector<T>> weights;
    activation_function<T> activation_func;

    layer(size_t num_neurons, size_t inputs_per_neuron, activation_function<T> activation = activation_functions<T>::relu) :
        activation_func(activation)
    {
        neurons.resize(num_neurons);
        unactivated_neurons.resize(num_neurons);
        errors.resize(num_neurons);
        weights.resize(num_neurons);

        for (auto& w : weights)
        {
            w.resize(inputs_per_neuron + 1); // +1 for bias
            initialize_weights(w);
        }
    }

    // Weight initialization
    void initialize_weights(std::vector<T>& w)
    {
        constexpr T weightRandRange = T(WEIGHT_RANDOM_RANGE);
        // Loop over all weights, including the bias
        for (size_t i = 0; i < w.size(); ++i)
            w[i] = (T(rand()) / RAND_MAX * 2 - 1) * weightRandRange;
    }

    virtual void forward(const std::vector<T>& inputs) = 0;
    virtual void backward(const std::vector<T>& prev_layer_neurons, T learningRate) = 0;
};

template<typename T>
class dense_layer : public layer<T> {
public:
    dense_layer(size_t num_neurons, size_t inputs_per_neuron, activation_function<T> activation = activation_functions<T>::relu) :
        layer<T>(num_neurons, inputs_per_neuron, activation)
    {
    }

    // Forward propagation for this layer
    void forward(const std::vector<T>& inputs)
    {
        for (size_t i = 0; i < this->neurons.size(); ++i)
            this->unactivated_neurons[i] = std::inner_product(inputs.begin(), inputs.end(), this->weights[i].begin(), this->weights[i].back());
        this->activation_func.activation(this->unactivated_neurons);
        this->neurons = this->unactivated_neurons;
    }

    // Backward propagation for this layer
    void backward(const std::vector<T>& prev_layer_neurons, T learningRate)
    {
        std::vector<T> derivatives(this->neurons.size());
        this->activation_func.derivative(this->unactivated_neurons, derivatives);
        for (size_t i = 0; i < this->neurons.size(); ++i)
        {
            for (size_t j = 0; j < prev_layer_neurons.size(); ++j)
                this->weights[i][j] += learningRate * this->errors[i] * derivatives[i] * prev_layer_neurons[j];
            this->weights[i].back() += learningRate * this->errors[i] * derivatives[i];
        }
    }

    // Error calculation needs to be performed in the neural network class
    // as it requires information about the next layer or the target values.
};

template<typename T = float>
class neural_network {
public:
    neural_network(T learningRate = 0.01f, loss_function<T> lossFunction = loss_functions<T>::mse) :
        m_learningRate(learningRate),
        m_lossFunction(lossFunction)
    {
    }

    void add_layer(std::shared_ptr<layer<T>> layer) {layers.push_back(layer);}

    std::vector<T> forward(const std::vector<T>& inputs)
    {
        std::vector<T> layer_input = inputs;
        for (auto& layer : layers)
        {
            layer->forward(layer_input);
            layer_input = layer->neurons; // Output of current layer is input to next
        }
        return layer_input; // Output of last layer
    }

    T compute_loss(const std::vector<T>& outputs, const std::vector<T>& targets)
    {
        return m_lossFunction.compute_loss(outputs, targets);
    }

    void backpropagate(const std::vector<T>& inputs, const std::vector<T>& targets)
    {
        // Calculate errors for output layer
        auto output_layer = layers.back();
        std::transform(targets.begin(), targets.end(), output_layer->neurons.begin(), output_layer->errors.begin(), std::minus<T>());

        // Calculate errors for hidden layers
        for (long i = layers.size() - 2; i >= 0; --i)
        {
            auto layer = layers[i];
            auto next_layer = layers[i + 1];

            std::vector<T> derivatives(layer->neurons.size());
            layer->activation_func.derivative(layer->unactivated_neurons, derivatives);

            for (size_t j = 0; j < layer->neurons.size(); ++j)
            {
                layer->errors[j] = 0;
                for (size_t k = 0; k < next_layer->neurons.size(); ++k)
                    layer->errors[j] += next_layer->errors[k] * next_layer->weights[k][j];
                layer->errors[j] *= derivatives[j];
            }
        }

        // Update weights
        std::vector<T> prev_layer_output = inputs;
        for (auto& layer : layers)
        {
            layer->backward(prev_layer_output, m_learningRate);
            prev_layer_output = layer->neurons;
        }
    }

private:
    std::vector<std::shared_ptr<layer<T>>> layers;
    T m_learningRate;
    loss_function<T> m_lossFunction;
};

int main(int argc, char* argv[])
{
    // Create a neural network
    neural_network<float> nn(0.01f);

    // Define the network architecture
    auto hiddenLayer1 = std::make_shared<dense_layer<float>>(10, 4, activation_functions<float>::relu);
    auto hiddenLayer2 = std::make_shared<dense_layer<float>>(10, 10, activation_functions<float>::relu);
    auto outputLayer = std::make_shared<dense_layer<float>>(3, 10, activation_functions<float>::softmax);

    // Add layers to the network
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(hiddenLayer1));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(hiddenLayer2));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(outputLayer));

    // Train the network
    int numEpochs = 10000;
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

#if 0
    // Learn the XOR function

    // Train a neural network to learn the XOR function
    neural_network<float> nn(0.01f);

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
