# dim-nn
## A super short neural network written in C++.

#### Q: Why is it dim?
#### A: Because it's not that bright.

#### Q: What is it good for?
#### A: Learning about neural networks and backpropagation.

#### Q: What is it not good for?
#### A: Anything else.

#### Q: Will it ever be good for anything else?
#### A: Probably not. If I ever learn enough about them I would like to implement a convolutional layer.

Here is an example that shows dim-nn learning the XOR function:
```
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
```
