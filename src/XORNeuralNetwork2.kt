/* 9/21/2017 - Yousef Abdelgaber

    Author's Note:
    Hello, I gave up on my first neural network after I started learning more and more of the mathematical matrix side of neural
    networks. I did a lot of reading on mathematical descriptions of neural nets and I made on in python using only around 40 lines of code.
    I even made one on my TI-nspire graphing calculator. I came back to try to fix my original neural network and then I just found it so broken
    and poorly made that I decided to give up on it an implement a new version that uses matrix multiplication instead.

    9/24/2017 - Yousef Abdelgaber
    Update:
    I've finally done it, I've made a neural network that functions correctly using Kotlin by programming a library of matrix functions.
    I'm quite proud of myself and of my work here. This is good code, and it works quite nicely. Much, much nicer than version 1. 
 */

fun main(args: Array<String>) {

    val inputs = createMatrix(
            row(0f,0f,1f),
            row(0f,1f,1f),
            row(1f,0f,1f),
            row(1f,1f,1f))

    val actualOutputs = createMatrix(
            row(0f),
            row(1f),
            row(1f),
            row(0f))

    var synapses0 = randomMatrix(3,3) // (2 inputs + bias) x 3 hidden layer neurons
    var synapses1 = randomMatrix(3,1) // 3 hidden layer neurons x 1 output layer neuron

    val learningRate = 0.1f

    for (i in 1..900000) {

        // FEED-FORWARD THROUGH THE NETWORK
        val layer0 = inputs
        val layer1 = activate(matrixMultiply(layer0, synapses0))
        val layer2 = activate(matrixMultiply(layer1, synapses1))

        val layer2Error = dotSubtract(actualOutputs, layer2)
        if ((i % 10000) == 0) {
            printMatrix(layer2)
            println("Error: " + matrixMean(matrixAbsolute(layer2Error)))
            println()
        }

        val layer2Delta = matrixMultiply(layer2Error, derivActivate(layer2))

        val layer1Error = matrixMultiply(layer2Delta, transpose(synapses1))
        val layer1Delta = dotMultiply(layer1Error, derivActivate(layer1))

        synapses1 = dotAdd(synapses1, scalarMultiply(learningRate, matrixMultiply(transpose(layer1), layer2Delta)))
        synapses0 = dotAdd(synapses0, scalarMultiply(learningRate, matrixMultiply(transpose(layer0), layer1Delta)))

    }

}

fun activate(matrix: Array<FloatArray>): Array<FloatArray> =
        scalarDivide(1f, scalarAdd(1f, scalarExponent(Math.E.toFloat(), scalarMultiply(-1f, matrix))))
fun derivActivate(matrix: Array<FloatArray>): Array<FloatArray> = // note that this requires already activated values
        dotMultiply(matrix,scalarAdd(1f, scalarMultiply(-1f, matrix)))
//        dotDivide(scalarExponent(Math.E.toFloat(), matrix), scalarExponent(scalarAdd(1f,scalarExponent(Math.E.toFloat(),matrix)),2f))

@Suppress("UNCHECKED_CAST")
fun createMatrix(vararg rows: FloatArray): Array<FloatArray> = rows as Array<FloatArray>
fun row(vararg x: Float): FloatArray = x
fun randomMatrix(rows: Int, columns: Int): Array<FloatArray> {
    val matrix = Array(rows, {FloatArray(columns)})
    for (i in 0 until rows)
        for (j in 0 until columns)
            matrix[i][j] = Math.random().toFloat()
    return matrix
}
fun matrixMultiply(matrix1: Array<FloatArray>, matrix2: Array<FloatArray>): Array<FloatArray> {
    val product = Array(rows(matrix1), {FloatArray(columns(matrix2))})
    for (i in 0 until matrix1.size)
        for (j in 0 until matrix2.first().size)
            for (k in 0 until matrix1.first().size)
                product[i][j] += matrix1[i][k] * matrix2[k][j]
    return product
}
fun scalarAdd(scalar: Float, matrix: Array<FloatArray>): Array<FloatArray> {
    val output = Array(rows(matrix), {FloatArray(columns(matrix))})
    for (i in 0 until rows(matrix))
        for (j in 0 until columns(matrix))
            output[i][j] = matrix[i][j] + scalar
    return output
}
fun scalarMultiply(scalar: Float, matrix: Array<FloatArray>): Array<FloatArray> {
    val output = Array(rows(matrix), {FloatArray(columns(matrix))})
    for (i in 0 until rows(matrix))
        for (j in 0 until columns(matrix))
            output[i][j] = matrix[i][j] * scalar
    return output
}
fun scalarDivide(scalar: Float, matrix: Array<FloatArray>): Array<FloatArray> {
    val output = Array(rows(matrix), {FloatArray(columns(matrix))})
    for (i in 0 until rows(matrix))
        for (j in 0 until columns(matrix))
            output[i][j] = scalar / matrix[i][j]
    return output
}
fun scalarExponent(scalar: Float, matrix: Array<FloatArray>): Array<FloatArray> {
    val output = Array(rows(matrix), {FloatArray(columns(matrix))})
    for (i in 0 until rows(matrix))
        for (j in 0 until columns(matrix))
            output[i][j] = Math.pow(scalar.toDouble(), matrix[i][j].toDouble()).toFloat()
    return output
}
fun scalarExponent(matrix: Array<FloatArray>, scalar: Float): Array<FloatArray> {
    val output = Array(rows(matrix), {FloatArray(columns(matrix))})
    for (i in 0 until rows(matrix))
        for (j in 0 until columns(matrix))
            output[i][j] = Math.pow(matrix[i][j].toDouble(), scalar.toDouble()).toFloat()
    return output
}
fun dotAdd(matrix1: Array<FloatArray>, matrix2: Array<FloatArray>): Array<FloatArray> {
    checkMatrixSizes(matrix1, matrix2)
    val output = Array(rows(matrix1), {FloatArray(columns(matrix1))})
    for (i in 0 until rows(matrix1))
        for (j in 0 until columns(matrix1))
            output[i][j] = matrix1[i][j] + matrix2[i][j]
    return output
}
fun dotSubtract(matrix1: Array<FloatArray>, matrix2: Array<FloatArray>): Array<FloatArray> {
    checkMatrixSizes(matrix1, matrix2)
    val output = Array(rows(matrix1), {FloatArray(columns(matrix1))})
    for (i in 0 until rows(matrix1))
        for (j in 0 until columns(matrix1))
            output[i][j] = matrix1[i][j] - matrix2[i][j]
    return output
}
fun dotDivide(matrix1: Array<FloatArray>, matrix2: Array<FloatArray>): Array<FloatArray> {
    checkMatrixSizes(matrix1, matrix2)
    val output = Array(rows(matrix1), {FloatArray(columns(matrix1))})
    for (i in 0 until rows(matrix1))
        for (j in 0 until columns(matrix1))
            output[i][j] = matrix1[i][j] / matrix2[i][j]
    return output
}
fun dotMultiply(matrix1: Array<FloatArray>, matrix2: Array<FloatArray>): Array<FloatArray> {
    checkMatrixSizes(matrix1, matrix2)
    val output = Array(rows(matrix1), {FloatArray(columns(matrix1))})
    for (i in 0 until rows(matrix1))
        for (j in 0 until columns(matrix1))
            output[i][j] = matrix1[i][j] * matrix2[i][j]
    return output
}
fun transpose(matrix: Array<FloatArray>): Array<FloatArray> {
    val output = Array(columns(matrix), {FloatArray(rows(matrix))})
    for (i in 0 until rows(matrix))
        for (j in 0 until columns(matrix))
            output[j][i] = matrix[i][j]
    return output
}
fun matrixAbsolute(matrix: Array<FloatArray>): Array<FloatArray> {
    val output = Array(rows(matrix), {FloatArray(columns(matrix))})
    for (i in 0 until rows(matrix))
        for (j in 0 until columns(matrix))
            output[i][j] = Math.abs(matrix[i][j])
    return output
}
fun matrixMean(matrix: Array<FloatArray>): Float {
    var mean = 0f
    for (i in 0 until rows(matrix))
        for (j in 0 until columns(matrix))
            mean += matrix[i][j]
    return mean / 4f
}
fun printMatrix(matrix: Array<FloatArray>) {
    for (row in matrix) {
        for (column in row)
            print("$column    ")
        println()
    }
}
fun checkMatrixSizes(matrix1: Array<FloatArray>, matrix2: Array<FloatArray>) {
    if (rows(matrix1) != rows(matrix2)) println("ERROR: ${rows(matrix1)} rows of matrix#1 != ${rows(matrix2)} rows of matrix#2")
    if (columns(matrix1) != columns(matrix2)) println("ERROR: ${columns(matrix1)} columns of matrix#1 != ${columns(matrix2)} columns of matrix#2")
}
fun rows(matrix: Array<FloatArray>): Int = matrix.size
fun columns(matrix: Array<FloatArray>): Int = matrix.first().size
