"use strict";

//////////////////////////////////////////////////////////////
// !!! TESTING V2:  Biases on 2nd Layer are blacked out !!! //
//////////////////////////////////////////////////////////////

// Create Log Count for reporting NN Loss
////////////////////////////////////////////

const LOG_ON = true;  // Show NN Loss in console
const LOG_FREQ = 20000;  // Frequency of logging Loss in console

/********************
    NEURAL NETWORK
 ********************/

class NeuralNetwork {
    constructor(numInputs, numHidden1, numHidden2, numOutputs) {
        this._inputs = [];
        this._hidden1 = [];
        this._hidden2 = [];
        this._numInputs = numInputs;
        this._numHidden1 = numHidden1;
        this._numHidden2 = numHidden2;
        this._numOutputs = numOutputs;
        this._weights0 = new Matrix(this._numInputs, this._numHidden1);
        this._weights1 = new Matrix(this._numHidden1, this._numHidden2);
        this._weights2 = new Matrix(this._numHidden2, this._numOutputs);
        this._bias0 = new Matrix(1, this._numHidden1);
        this._bias1 = new Matrix(1, this._numHidden2);
        //this._bias2 = new Matrix(1, this._numOutputs);

        this._logCount = LOG_FREQ;

        // Randomize initial Weights & Biases
        this._weights0.randomWeights();
        this._weights1.randomWeights();
        this._weights2.randomWeights();
        this._bias0.randomWeights();
        this._bias1.randomWeights();
        //this._bias2.randomWeights();
    }

    get inputs() {
        return this._inputs;
    }

    set inputs(inputs) {
        this._inputs = inputs;
    }

    get hidden1() {
        return this._hidden1;
    }

    set hidden1(hidden1) {
        this._hidden1 = hidden1;
    }

    get hidden2() {
        return this._hidden2;
    }

    set hidden2(hidden2) {
        this._hidden2 = hidden2;
    }

    get weights0() {
        return this._weights0;
    }

    set weights0(weights) {
        this._weights0 = weights;
    }

    get weights1() {
        return this._weights1;
    }

    set weights1(weights) {
        this._weights1 = weights;
    }

    get weights2() {
        return this._weights2;
    }

    set weights2(weights) {
        this._weights2 = weights;
    }
    
    get bias0() {
        return this._bias0;
    }

    set bias0(bias0) {
        this._bias0 = bias0;
    }

    get bias1() {
        return this._bias1;
    }

    set bias1(bias1) {
        this._bias1 = bias1;
    }

    get bias2() {
        return this._bias2;
    }

    set bias2(bias2) {
        this._bias2 = bias2;
    }

    get logCount() {
        return this._logCount;
    }

    set logCount(count) {
        this._logCount = count;
    }

    // Forward Propogation
    /////////////////////////

    feedForward(inputArray) {
        // Convert input array to a Matrix
        this.inputs = Matrix.convertFromArray(inputArray);

        // Find Hidden Layer 1's 'z' values & apply Activation Function 'a'
        this.hidden1 = Matrix.dot(this.inputs, this._weights0);
        this.hidden1 = Matrix.add(this.hidden1, this.bias0);
        this.hidden1 = Matrix.map(this.hidden1, x => sigmoid(x));

        // Find Hidden Layer 2's 'z' values & apply Activation Function 'a'
        this.hidden2 = Matrix.dot(this.hidden1, this._weights1);
        this.hidden2 = Matrix.add(this.hidden2, this.bias1);
        this.hidden2 = Matrix.map(this.hidden2, x => sigmoid(x));

        // Find Output 'z' values & apply Activation Function 'a'
        let outputs = Matrix.dot(this.hidden2, this._weights2);
        //outputs = Matrix.add(outputs, this.bias2);
        outputs = Matrix.map(outputs, x => sigmoid(x));

        return outputs;
    }

    // Back Propogation
    //////////////////////

    train(inputArray, targetArray) {

        // Feed input data to the Neural Network
        let outputs = this.feedForward(inputArray);

        // Calculate Loss (Target Output - NN Output)
        let targets = Matrix.convertFromArray(targetArray);
        let outputErrors = Matrix.subtract(targets, outputs);

        // Console log the Loss
        if (LOG_ON) {
            if (this.logCount == LOG_FREQ) {
                console.log("Loss is " + outputErrors.data[0][0]);
            }
            this.logCount--;
            if (this.logCount == 0) {
                this.logCount = LOG_FREQ;
            }
        }

        // Calculate Loss Deltas (Loss * Derivative of Output)
        let outputDerivs = Matrix.map(outputs, x => sigmoid(x, true));
        let outputDeltas = Matrix.multiply(outputErrors, outputDerivs);

        // Calculate Hidden Layer 2 Errors (Deltas 'dot' Transpose of weights1)
        let weights2T = Matrix.transpose(this.weights2);
        let hidden2Errors = Matrix.dot(outputDeltas, weights2T);

        // Calculate Hidden Layer 2 Deltas (Errors * Derivative of Hidden)
        let hidden2Derivs = Matrix.map(this.hidden2, x => sigmoid(x, true));
        let hidden2Deltas = Matrix.multiply(hidden2Errors, hidden2Derivs);

        // Calculate Hidden Layer 1 Errors (Deltas 'dot' Transpose of weights1)
        let weights1T = Matrix.transpose(this.weights1);
        let hidden1Errors = Matrix.dot(hidden2Deltas, weights1T);

        // Calculate Hidden Layer 1 Deltas (Errors * Derivative of Hidden)
        let hidden1Derivs = Matrix.map(this.hidden1, x => sigmoid(x, true));
        let hidden1Deltas = Matrix.multiply(hidden1Errors, hidden1Derivs);

        // Update Weights (Add Transpose of Hidden Layer 'dot' Deltas)
        let hidden2T = Matrix.transpose(this.hidden2);
        this.weights2 = Matrix.add(this.weights2, Matrix.dot(hidden2T, outputDeltas));
        let hidden1T = Matrix.transpose(this.hidden1);
        this.weights1 = Matrix.add(this.weights1, Matrix.dot(hidden1T, hidden2Deltas));
        let inputsT = Matrix.transpose(this.inputs);
        this.weights0 = Matrix.add(this.weights0, Matrix.dot(inputsT, hidden1Deltas));
        
        // Update Biases
        //this.bais2 = Matrix.add(this.bias2, outputDeltas);
        this.bais1 = Matrix.add(this.bias1, hidden2Deltas);
        this.bais0 = Matrix.add(this.bias0, hidden1Deltas);

    }

}


/**************************
    ACTIVATION FUNCTIONS
 **************************/

// Currently used on all Layers. Necessary for final Layer.
 function sigmoid(x, deriv = false) {
    if (deriv) {
        return x * (1 - x)  // Where x = sigmoid(x)
    }
    return 1 / (1 + Math.exp(-x));
}

// NOT YET USED (TO ADD/TEST TANH, RELU, LEAKLY RELU ON FIRST)
function tanh(x) {
    return (2 / (1 + Math.exp(-2 * x))) - 1;
}


/*********************************************
    MATRIX, MATRIX MATH, & MATRIX FUNCTIONS
 *********************************************/

class Matrix {
    constructor(rows, cols, data = []) {
        this._rows = rows;
        this._cols = cols;
        this._data = data;

        // Initialize Matrix with zeros
        if (data == null || data.length == 0) {
            this._data = [];
            for (let i = 0; i < this._rows; i++) {
                this._data[i] = [];
                for (let j = 0; j < this._cols; j++) {
                    this._data[i][j] = 0;
                }
            }
        } else {  // Check that data shape matches Matrix shape
            if (data.length != rows || data[0].length != cols) {
                throw new Error("Matrix and data dimensions don't match!");
            }
        }

    }

    get rows() {
        return this._rows;
    }

    get cols() {
        return this._cols;
    }

    get data() {
        return this._data;
    }

    // Add two Matrices
    static add(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] + m1.data[i][j];
            }
        }
        return m;
    }

    // Subtract two Matrices
    static subtract(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] - m1.data[i][j];
            }
        }
        return m;
    }

    // Multiply two Matrices
    static multiply(m0, m1) {
        Matrix.checkDimensions(m0, m1);
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                m.data[i][j] = m0.data[i][j] * m1.data[i][j];
            }
        }
        return m;
    }

    // Dot Product of two Matrices
    static dot(m0, m1) {
        if (m0.cols != m1.rows) {
            throw new Error("Matrices are not dot compatible!");
        }
        let m = new Matrix(m0.rows, m1.cols);
        for (let i = 0; i < m.rows; i++) {
            for (let j = 0; j < m.cols; j++) {
                let sum = 0;
                for (let k = 0; k < m0.cols; k++) {
                    sum += m0.data[i][k] * m1.data [k][j];
                }
                m.data[i][j] = sum;
            }
        }
        return m;
    }

    // Transpose Matrix (Switch Rows and Columns for Dimensional Needs)
    static transpose(m0) {
        let m = new Matrix(m0.cols, m0.rows);
        for (let i = 0; i < m0.rows; i++) {
            for (let j = 0; j < m0.cols; j++) {
                m.data[j][i] = m0.data[i][j];
            }
        }
        return m;
    }

    // Apply a Function to a Matrix
    static map(m0, mFunction) {
        let m = new Matrix(m0.rows, m0.cols);
        for (let i = 0; i < m0.rows; i++) {
            for (let j = 0; j < m0.cols; j++) {
                m.data[i][j] = mFunction(m0.data[i][j]);
            }
        }
        return m;
    }

    // Check that the two Matrices have matching dimensions
    static checkDimensions(m0, m1) {
        if (m0.rows != m1.rows || m0.cols != m1.cols) {
            throw new Error("Matrices have non-matching dimensions!");
        }
    }

    // Convert Array to Matrix
    static convertFromArray(arr) {
        return new Matrix(1, arr.length, [arr]);
    }

    // Apply random Numbers between -1 and 1
    randomWeights() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }

}