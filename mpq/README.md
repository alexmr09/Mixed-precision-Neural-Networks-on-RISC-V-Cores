# Mixed Precision Quantization

This directory provides details and examples on building and evaluating the performance (in terms of accuracy) of Quantized Neural Networks with Mixed-Precision variables. After completing the Design Space Exploration (DSE) and selecting the optimal solution, the necessary files for inference and simulation of each model on the modified Ibex core will be stored in the [inference_codes](https://github.com/alexmr09/Mixed-precision-Neural-Networks-on-RISC-V-Cores/tree/main/inference_codes) directory. The generated folder, named after the executed .py file, will contain the codes and parameters for both the optimized (with the new instructions) and the unoptimized (original RV32IMC) algorithms.

To start with, we will need to install all the necessary dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## File Description

- **setup.sh**: Sets up the paths and environmental variables for smoother execution of the programs.
- **init_utils.py**: Reads the arguments passed.
- **mpq_quantize.py**: Performs the training and evaluation of both full-precision and quantized models, as well as executes a DSE to find optimal solutions.
- **pareto_sols.py**: For *Exhaustive* search, it identifies the Pareto optimal solutions and exports a PNG image of the Pareto space.
- **configure_ibex.py**: Generates the header files that contain the network's parameters (weights, inputs and scale parameters) and the C program that defines its architecture.
- **simulate_ibex.py**: For the examples included on the subfolders, we also provide scripts that simulate the behavior of the modified core, ensuring that the results from Python match those produced by Ibex.
- **common.py**: Contains the main function, which is called after defining the model and dataset, and initiates the procedures described in the previous files.


## Use Cases

On the subfolders of this repository, you can check some simple examples that demonstrate how to execute the steps for the DSE. To run them:

```
source .venv/bin/activate
source setup_env.sh
python3 <folder_name>/<python_script_name>.py --max_acc_drop <max_acc_drop> --device <device>
```

#### Parameters Explained

- **max_acc_drop**: This parameter allows the user to define the maximum acceptable accuracy degradation for the Quantized Model compared to the full-precision model. When specified, a Binary search algorithm is used to explore the design space. If this parameter is not defined, an Exhaustive search approach is employed (default).
- **device**: Specifies the target device for running the model. The available options are CPU (default) and CUDA.

#### Exhaustive DSE example

```
python3 elderly_fall/elderly_fall.py --device 'cuda'
```

#### Binary DSE example

```
python3 elderly_fall/elderly_fall.py --max_acc_drop 2 --device 'cpu'
```

## How to test a new model

To use the provided scripts on a new model, follow these four simple steps:

1. **Initialization and setup**: At the beginning of your Python script, include the following lines to initialize the environment and retrieve arguments:

```python
import init_utils
import common

# Initialize the environment and get the name
name = init_utils.initialize_environment(__file__)
args = init_utils.get_args()
```

2. **Import and Preprocess Dataset**: Import your dataset and preprocess it. Ensure to split it into training, testing, and (optionally) validation subsets. Save these subsets as numpy arrays.
   
3. **Model Architecture**: Define your neural network architecture. Ensure that activation functions for hidden layers are defined as separate layers (objects of the torch.nn library).
   
4. **Create and Run the Quantized Neural Network**: Create an instance of your neural network and call the *create_ibex_qnn* function from common.py.
   
```python
common.create_ibex_qnn(net, name, device, X_train, y_train, X_test, y_test, 
                X_val = X_val, y_val = y_val, BATCH_SIZE = BATCH_SIZE, 
                epochs = epochs, lr = lr, max_acc_drop = max_acc_drop)
```

#### Notes
- The parameters for epochs and learning rate can be either single values or lists of numbers, allowing for fine-tuning of the model.
- Currently, the provided scripts do not support models with Residual connections.
