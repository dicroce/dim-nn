
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
#include <fstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <cstdlib>

#define WEIGHT_RANDOM_RANGE .1

using namespace std;

template<typename T>
struct activation_function
{
    std::function<T(T)> activation;
    std::function<T(T)> derivative;
};

template<typename T>
struct activation_functions
{
    inline static const activation_function<T> relu = { [](T x) { return (x > T(0)) ? x : 0; }, [](T x) { return T(x > 0); } };
    inline static const activation_function<T> sigm = { [](T x) { return T(1) / (T(1) + exp(-x)); }, [](T x) { T s = T(1) / (T(1) + exp(-x)); return s * (T(1) - s); } };
    inline static const activation_function<T> tanh = { [](T x) { return std::tanh(x); }, [](T x) { T t = std::tanh(x); return T(1.0f) - t * t; } };
    inline static const activation_function<T> step = { [](T x) { return x <= T(0) ? T(-1) : T(1); }, [](T x) { return 1; } };
    inline static const activation_function<T> none = { [](T x) { return x; }, [](T x) { return 1; } };
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
        {
            this->unactivated_neurons[i] = std::inner_product(inputs.begin(), inputs.end(), this->weights[i].begin(), this->weights[i].back());
            this->neurons[i] = this->activation_func.activation(this->unactivated_neurons[i]);
        }
    }

    // Backward propagation for this layer
    void backward(const std::vector<T>& prev_layer_neurons, T learningRate)
    {
        for (size_t i = 0; i < this->neurons.size(); ++i)
        {
            for (size_t j = 0; j < prev_layer_neurons.size(); ++j)
                this->weights[i][j] += learningRate * this->errors[i] * this->activation_func.derivative(this->unactivated_neurons[i]) * prev_layer_neurons[j];
            this->weights[i].back() += learningRate * this->errors[i] * this->activation_func.derivative(this->unactivated_neurons[i]); // Update bias weight
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
        return loss_functions<T>::mse.compute_loss(outputs, targets);
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

            for (size_t j = 0; j < layer->neurons.size(); ++j)
            {
                layer->errors[j] = 0;
                for (size_t k = 0; k < next_layer->neurons.size(); ++k)
                    layer->errors[j] += next_layer->errors[k] * next_layer->weights[k][j];
                layer->errors[j] *= layer->activation_func.derivative(layer->unactivated_neurons[j]);
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
    // Train a neural network to learn the XOR function
    neural_network<float> nn(0.01f);

    auto firstLayer = std::make_shared<dense_layer<float>>(4, 2, activation_functions<float>::relu);
    auto outputLayer = std::make_shared<dense_layer<float>>(1, 4, activation_functions<float>::relu);

    // Add layers to the network
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(firstLayer));
    nn.add_layer(std::dynamic_pointer_cast<layer<float>>(outputLayer));

    for(int i = 0; i < 10000; ++i)
    {
       vector<float> inputs = { ((rand()%2)==0)?0.0f:1.0f, ((rand()%2)==0)?0.0f:1.0f };

       vector<float> expected = { (float)((int)inputs[0] ^ (int)inputs[1]) };

       auto output_layer = nn.forward(inputs);

       auto loss = nn.compute_loss(output_layer, expected);

       printf("Expected Output: %f Actual Output: %f Loss %f\n", expected[0], output_layer[0], loss);

       nn.backpropagate(inputs, expected);
    }

    return 0;
}
