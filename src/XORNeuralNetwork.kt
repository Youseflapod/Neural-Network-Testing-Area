/* 8/29/2017 - Yousef Abdelgaber

    Disclaimer:
    Hello, if my future self or anyone else is reading this, I apologize for the disgusting nature of this code structure. I completely understand
    how toxic and self destructive my programming methods are, I apologize for any damage I may have caused to your mental state.

 */

fun main(args: Array<String>) {

    val trainingData = Array(3, {IntArray(4)})

    trainingData[0][0] = 0
    trainingData[1][0] = 0
    trainingData[2][0] = 0

    trainingData[0][1] = 0
    trainingData[1][1] = 1
    trainingData[2][1] = 1

    trainingData[0][2] = 1
    trainingData[1][2] = 0
    trainingData[2][2] = 1

    trainingData[0][3] = 1
    trainingData[1][3] = 1
    trainingData[2][3] = 0

    val neuronOfLayers = ArrayList<Int>()
    neuronOfLayers.add(2)
        neuronOfLayers.add(3)
        //neuronOfLayers.add(4)
        //neuronOfLayers.add(3)
    neuronOfLayers.add(1)
    val neuralNet = NeuralNetwork(neuronOfLayers)

    println("XOR MACHINE LEARNING ALGORITHIM STARTED")
    println("NEURAL NETWORK INITIALIZATION COMPLETE")
    println("FORWARD PROPAGATION SEQUENCE STARTED")
    println()

    for (j in 1..2) {
        println("Cycle number $j of training data set")
        println()
        for (i in 0..3) {
            println("Data updated with data number ${i+1}")
            neuralNet.refreshTrainingData()
            neuralNet.inputValues.add(trainingData[0][i].toFloat())
            neuralNet.inputValues.add(trainingData[1][i].toFloat())
            neuralNet.expectedOutputValues.add(trainingData[2][i].toFloat())
            neuralNet.forwardPropagate()
            neuralNet.backPropagate()
            println("Input data (${trainingData[0][i].toFloat()}, ${trainingData[1][i].toFloat()}) resulted in ${neuralNet.actualOutputValues[0]}")
            println("Expected output value: ${trainingData[2][i].toFloat()} ")
            println("Error: ${trainingData[2][i].toFloat() - neuralNet.actualOutputValues[0]} ")
            println()
        }
    }
}

class NeuralNetwork(neuronsOfLayers: ArrayList<Int>) {

    var inputValues = ArrayList<Float>()
    var expectedOutputValues = ArrayList<Float>()
    var actualOutputValues = ArrayList<Float>()

    var deltaOutputSum: Float = 0f

    var layers = ArrayList<Layer>()

    init {
        // for all of the values except the last one create a layer in the neural network
        for (i in 1..neuronsOfLayers.size-1) layers.add(Layer(neuronsOfLayers[i-1],false,i-1,this))
        layers.add(Layer(neuronsOfLayers.last(),true, layers.size,this)) // add on the output layer after loop with custom boolean
        for (i in 0 until layers.size) layers[i].randomlyGenerateWeights() // after constructing the network, go through and randomly generate the weights
    }

    fun refreshTrainingData() {
        inputValues.clear()
        expectedOutputValues.clear()
        actualOutputValues.clear()
    }

    fun forwardPropagate() {
        for (i in 0 until layers.size) layers[i].forwardPropagate()
    }
    fun backPropagate() {
        for (i in layers.size-1 downTo 0) layers[i].backPropagate()
    }

}

class Layer(val neuronCount: Int, val isOutputLayer: Boolean, val layerNumber: Int, network: NeuralNetwork) {
    var neurons = ArrayList<Neuron>()
    init {
        for (i in 1..neuronCount) neurons.add(Neuron(this,  network))
    }
    fun randomlyGenerateWeights() {
        for (i in 0 until neurons.size) neurons[i].randomlyGenerateWeighs()
    }
    fun forwardPropagate() {
        for (i in 0 until neurons.size) neurons[i].forwardPropagate()
    }
    fun backPropagate() {
        for (i in 0 until neurons.size) neurons[i].backPropagate()
    }
}

class Neuron(val layer: Layer, val network: NeuralNetwork) {

    var outputWeights = ArrayList<Float>()
    var inputWeights = ArrayList<Float>()
    var unactivatedValue: Float = 0f
    var activatedValue: Float = 0f

    fun randomlyGenerateWeighs() {
        if (!layer.isOutputLayer)  // only layers with outputs
            for (i in 1..network.layers[layer.layerNumber+1].neuronCount) // get the neuron count of the layer ahead
                outputWeights.add(Math.random().toFloat()) // randomize output weights between 0 and 1
    }

    fun forwardPropagate() {
        unactivatedValue = 0f // reset the unactivated value
        if (layer.layerNumber == 0) { // if input layer
            activatedValue = network.inputValues[layer.neurons.indexOf(this)] // get input value for the index of input neuron
        } else { // if a hidden layer or output layer // note that you can sum the input weights like this because it starts off with the input layer, which requires no input weights
            for (i in 0 until inputWeights.size) unactivatedValue += inputWeights[i] // add up the unactivated value
            activatedValue = sigmoidActivation(unactivatedValue) // activation function on sum of input weights
        }
        if (layer.isOutputLayer) // if the output layer, add the activated neuron value to the list of neural net outputs
            network.actualOutputValues.add(activatedValue)
        else // if an input layer, or hidden layer, loop through each neuron in the layer ahead
            for (i in 1..network.layers[layer.layerNumber+1].neuronCount) // gets the number of neurons in the layer ahead
                network.layers[layer.layerNumber+1].neurons[i - 1].inputWeights.add(outputWeights[i - 1] * activatedValue)
                // get each neuron in the layer ahead and add each of this neurons output weights multiplied by its activated value to that neurons list of input weights
        inputWeights.clear() // clear all of the input weights, to reset it
    }
    fun backPropagate() {
        if (layer.isOutputLayer) {
            network.deltaOutputSum = sigmoidPrime(unactivatedValue) * (network.expectedOutputValues[layer.neurons.indexOf(this)] - activatedValue)
        }
    }
}
fun sigmoidActivation(x: Float): Float = 1f / (1 + Math.pow(Math.E, -x.toDouble())).toFloat()
fun sigmoidPrime(x: Float): Float = (Math.pow(Math.E, x.toDouble()) / Math.pow((Math.pow(Math.E, x.toDouble()) + 1), 2.0)).toFloat()