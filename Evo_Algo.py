

"""
Note: You have to connect your Qiskit IBM Account to be able to use this.
Here is some intuition on what the code does:
    ansatz_ckt : Create the ansatz we're testing on algorithms on
    MKCircuit  : Create the entire circuit along with qubits and gates required at the input
    santafe_preprocess and santafe_preprocess_cma: Preprocess dataset. (Usage guidelines: Use either one depending on requirement. I've found during testing that the datasets I was given to run the algorithms on were mostly 'solved' and so we had to introduce artificial difficulty to demonstrate improvement)
    evaluate_single_data_point : Used to parallelize the CMA training
    fitness_fn : The fitness function used to train the CMA
    train_custom : Used to train the QRNN
    hybrid_training : Used as part of the study to combine the benefits of both the gradient and non-gradient based methods.
"""

# Importing standard Qiskit libraries

#from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import *
from qiskit_ibm_runtime import Estimator

# Loading your IBM Quantum account(s)
#service = QiskitRuntimeService(channel="ibm_quantum")
import warnings
warnings.filterwarnings("ignore")

import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from torch.nn import MSELoss
from qiskit.circuit import Parameter

from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.quantum_info import SparsePauliOp



import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv
def ansatz_ckt():
    
    qreg_q = QuantumRegister(6, 'q')
    #creg_c = ClassicalRegister(1, 'c')
    params1 = [Parameter("weight1"),Parameter("weight2"),Parameter("weight3"),Parameter("weight4"),Parameter("weight5"),Parameter("weight6"),Parameter("weight7"),Parameter("weight8"),Parameter("weight9"),Parameter("weight10"),Parameter("weight11"),Parameter("weight12"),Parameter("weight13"),Parameter("weight14"),Parameter("weight15"),Parameter("weight16"),Parameter("weight17"),Parameter("weight18"),Parameter("weight19"),Parameter("weight20"),Parameter("weight21"),Parameter("weight22"),Parameter("weight23"),Parameter("weight24")]
    circuit = QuantumCircuit(qreg_q)
    circuit.rx(params1[1], qreg_q[0])
    circuit.rx(params1[2], qreg_q[1])
    circuit.rx(params1[3], qreg_q[2])
    circuit.rx(params1[4], qreg_q[3])
    circuit.rx(params1[5], qreg_q[4])
    circuit.rx(params1[6], qreg_q[5])
    circuit.rz(params1[7], qreg_q[0])
    circuit.rz(params1[8], qreg_q[1])
    circuit.rz(params1[9], qreg_q[2])
    circuit.rz(params1[10], qreg_q[3])
    circuit.rz(params1[11], qreg_q[4])
    circuit.rz(params1[12], qreg_q[5])
    circuit.rx(params1[13], qreg_q[0])
    circuit.rx(params1[14], qreg_q[1])
    circuit.rx(params1[15], qreg_q[2])
    circuit.rx(params1[16], qreg_q[3])
    circuit.rx(params1[17], qreg_q[4])
    circuit.rx(params1[18], qreg_q[5])
    circuit.cx(qreg_q[0],qreg_q[1])
    circuit.rz(params1[19],qreg_q[1])
    circuit.cx(qreg_q[0],qreg_q[1])
    circuit.cx(qreg_q[1],qreg_q[2])
    circuit.rz(params1[20],qreg_q[2])
    circuit.cx(qreg_q[1],qreg_q[2])
    circuit.cx(qreg_q[2],qreg_q[3])
    circuit.rz(params1[21],qreg_q[3])
    circuit.cx(qreg_q[2],qreg_q[3])
    circuit.cx(qreg_q[3],qreg_q[4])
    circuit.rz(params1[22],qreg_q[4])
    circuit.cx(qreg_q[3],qreg_q[4])
    circuit.cx(qreg_q[4],qreg_q[5])
    circuit.rz(params1[23],qreg_q[5])
    circuit.cx(qreg_q[4],qreg_q[5])
    circuit.cx(qreg_q[5],qreg_q[0])
    circuit.rz(params1[0],qreg_q[0])
    circuit.cx(qreg_q[5],qreg_q[0])
    return circuit,params1
#circuit.measure(qreg_q[0], creg_c[0])



def MKCircuit(numQubits):


    params_i = [Parameter("input1")]
    qreg_q = QuantumRegister(6, 'q')
    #creg_c = ClassicalRegister(1, 'c')
    circuit_x = QuantumCircuit(qreg_q)

    circuit_x.ry(params_i[0], qreg_q[0])
    circuit_x.ry(params_i[0], qreg_q[1])
    circuit_x.ry(params_i[0], qreg_q[2])
    circuit_x.barrier()
    ansatz,weights = ansatz_ckt()
    qnn_qc = QuantumCircuit(QuantumRegister(numQubits), ClassicalRegister(1))
    qnn_qc.compose(circuit_x, inplace=True)
    qnn_qc.compose(ansatz, inplace=True)

    return qnn_qc, params_i, weights

### Refer to the paper to add additional timesteps
def santafe_preprocess():
    y = []
    count = 0
    with open("C:\HPCA\mgdata.dat.txt") as tsv:
        for line in csv.reader(tsv, delimiter='\t'):
            y.append(float(line[0].split()[1]))
    y = np.float64(y)
    train = y.reshape((-1, 1))
    training_set = train[0:100]

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    X_train = []
    y_train = []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7: i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, newshape = (X_train.shape[0], X_train.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(     X_train, y_train, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test
### This function can be commented out and the dataset_preprocess function shoould be sufficient. We used two functions here as the CMA expected the dataset in a different shape.
def santafe_preprocess_cma():
    y = []
    with open("C:\HPCA\mgdata.dat.txt") as tsv:
         for line in csv.reader(tsv, delimiter='\t'):
             y.append(float(line[0].split()[1]))   
    y = np.float64(y)
    train = y.reshape((-1,1))
    training_set = train[0:100]
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train = []
    y_train = []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7: i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, newshape = (X_train.shape[0], X_train.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(     X_train, y_train, test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test
#USE THIS TO DO ONE SET OF 7 MEASUREMENTS]
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
service = QiskitRuntimeService(channel_strategy="q-ctrl")
from qiskit.providers.fake_provider import GenericBackendV2
backend_sim = GenericBackendV2(6)
from qiskit_aer import AerSimulator,Aer
backend = AerSimulator.from_backend(backend_sim)

#from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
estimator = Estimator(backend=backend)
#estimator = Estimator(backend=backend)
from sklearn.metrics import mean_squared_error
circuit, input_params, weights = MKCircuit(6)
circuit2, input_params2, weights2 = MKCircuit(6)
op = SparsePauliOp.from_list([("IIIIIZ", 1)]) ##CHANGE BY REMOVING ONE I
qasm_simulator = Aer.get_backend('qasm_simulator')
from concurrent.futures import ThreadPoolExecutor

import re
import time
import logging
from qiskit_ibm_runtime import IBMRuntimeError
MAX_RETRIES = 16  # Define how many times you want to retry

# Feel free to tweak these lines depending on your system's specs
def evaluate_single_data_point(data_input, est_weights, circuit, input_params, weights, estimator, op):
    bc = circuit

    for i in range(0, 24):

        bc = bc.assign_parameters({weights[i]: est_weights[i]})

    bc1 = bc.assign_parameters({input_params[0]: data_input[6][0]})
    bc1.measure(0, 0)
    for attempt in range(MAX_RETRIES):
        try:
            expectation_value = qasm_simulator.run(bc1).result().data()
            expectation_value = str(expectation_value)
            match = re.search(r"'0x1'\s*:\s*(\d+)", expectation_value)
            if match:
                value = match.group(1)
                gg = float(value)/1024
            return gg

        except IBMRuntimeError as e:
            if "Too Many Requests" in str(e):
                # If we hit the rate limit, apply exponential backoff with a cap of 60 seconds.
                wait_time = min(2 ** (attempt + 1), 60)
                logging.warning(f"Rate limit hit. Retrying in {wait_time} seconds. Attempt {attempt + 1}/{MAX_RETRIES}.")
                time.sleep(wait_time)
                continue
            else:
                # For any other exception, raise it.
                raise

    logging.error("Max retries hit for IBMRuntime. Consider increasing MAX_RETRIES or further optimizing request patterns.")
    return None


import multiprocessing as mp
print("NUMBER OF CPU CORES:", mp.cpu_count)

def fitness_fn(est_weights):
    # Similar setup code...
    data_inputs, _, data_outputs, _ = santafe_preprocess_cma()
    y_pred = []

    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [
            executor.submit(
                evaluate_single_data_point,
                data_inputs[i], est_weights, circuit2, input_params2, weights2, estimator, op
            )
            for i in range(len(data_inputs[:,0,:]))
        ]
        for future in futures:
            y_pred.append(future.result())

    absolute = mean_squared_error(data_outputs, y_pred)
    print(absolute)
    return absolute

import random
import cma

#WORKING QRNN CODE
def train_custom(qnn_qc, input_parameters, weights,num_epochs):
    qnn = EstimatorQNN(
    circuit=qnn_qc,
    input_params=input_parameters,
    weight_params=weights

    )
    np.random.seed(42)      ###COMMENT OUT THE SEED AND TRY AGAIN IF REQUIRED
    estimator_qnn_weights = np.random.random(qnn.num_weights)

    model = TorchConnector(qnn)
    model.train()
    X_train,x_test,y_train,y_test = santafe_preprocess()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = MSELoss()
    y_predicted = []
    prev_param = torch.tensor([])
    total_loss = []
    hybrid_loss_list = []  # Store loss history
    for epoch in range(num_epochs):
        for i in range(len(X_train[:,0,:])):
            optimizer.zero_grad()
            if i%200==0:
                for name, params in model.named_parameters():
                    #Comment these lines as well for reduced clutter
                    #print(name, params, prev_param)
                    if prev_param.numel() > 0:
                        diff = prev_param - params
                        print("Diff:")
                        print(torch.nonzero(diff))
                        print("-------- \n \n")
                    prev_param = params.clone().detach()

            for j in range(0,7):
                output = model(torch.Tensor(X_train[i][j]))
            #Uncomment the line below to add multiple time steps (You can copy the line multiple times for more time steps)
            #second_timestep = model(torch.Tensor(output))
            

            y_pred = torch.Tensor(np.array(y_train[i]))
            y_pred = torch.reshape(y_pred, (-1,))

            y_predicted.append(output)
            loss = criterion(output, y_pred)
            loss.backward()
            optimizer.step()
            print(loss)
            total_loss.append(loss.item())

        hybrid_loss_list.append(sum(total_loss) / len(total_loss))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / num_epochs, hybrid_loss_list[-1]))
        with open("C:/Users/vigne/OneDrive/Documents/DataDump_Glass_step4_Hybrid_New.txt", "a") as file:
            file.write("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / num_epochs, hybrid_loss_list[-1]))
        
    final_weights = [param.data.numpy() for param in model.parameters()]
    #Comment out the lines below to reduce clutter if required.
    y_predicted_test = []
    with torch.no_grad():
        for i in range(len(x_test[:,0,:])):
            for j in range(0,7):
                output = model(torch.Tensor(x_test[i][j]))
            print(f"\n Predicted: {output}, Target: {y_pred}")
            y_predicted_test.append(output)

    return final_weights
    # Saving the weights for future use
    #save_model(model, "model_weights_all.pth")

    # model_state_dict = model.state_dict()

def hybrid_training(qnn_qc, input_params, weights, qrnn_epochs, cma_es_activation):
    # Train with QRNN
    if qrnn_epochs == 0:
        qrnn_weights = np.random.random(24)
        print("Using this one")
    else:
        qrnn_weights = train_custom(qnn_qc, input_params, weights, num_epochs=qrnn_epochs)

    # Initialize CMA-ES with QRNN weights
    es = cma.CMAEvolutionStrategy(qrnn_weights, sigma0=0.5, inopts={"popsize": 3})
    for i in range(cma_es_activation):
        es.optimize(fitness_fn)
    return es.result[0]  # This returns the best solution found by CMA-ES



    #return final_weights_array
if __name__ == "__main__":
    # Define your quantum circuit, parameters, etc.
    qnn_qc, input_params, weights = MKCircuit(6)

    # Define the number of epochs for QRNN and CMA-ES
    qrnn_epochs = 100  # User-defined number of epochs for QRNN training
    cma_es_activation = 1  # Note: The current version of this code had an issue setting an epoch limit. For the purposes of this paper, We let the CMA code and then stop it manually at the desired epoch number

    # Perform hybrid training
    final_weights = hybrid_training(qnn_qc, input_params, weights, qrnn_epochs, cma_es_activation)

    # final_weights now hold the optimized weights after hybrid training



