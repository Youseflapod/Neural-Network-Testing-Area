/* 9/21/2017 - Yousef Abdelgaber

    Author's Note:
    Hello, I gave up on my first neural network after I started learning more and more of the mathematical matrix side of neural
    networks. I did a lot of reading on mathematical descriptions of neural nets and I made on in python using only around 40 lines of code.
    I even made one on my TI-nspire graphing calculator. I came back to try to fix my original neural network and then I just found it so broken
    and poorly made that I decided to give up on it an implement a new version that uses matrix multiplication instead.

 */

fun main(args: Array<String>) {

    val inputs = arrayOf( // third input neuron is bias term
            intArrayOf(0, 0, 1),
            intArrayOf(0, 1, 1),
            intArrayOf(1, 0, 1),
            intArrayOf(1, 1, 1))

    val actualOutputs = arrayOf(
            intArrayOf(0),
            intArrayOf(1),
            intArrayOf(1),
            intArrayOf(0))

    val syn0 =

    /*val firstMatrix = arrayOf(
            intArrayOf(3, -2, 5),
            intArrayOf(3, 0, 4))
    val secondMatrix = arrayOf(
            intArrayOf(2, 3),
            intArrayOf(-9, 0),
            intArrayOf(0, 4))

    displayMatrix(multiplyMatrices(firstMatrix, secondMatrix))
    */
}

fun multiplyMatrices(matrix1: Array<IntArray>, matrix2: Array<IntArray>): Array<IntArray> {
    val product = Array(matrix1.size, {IntArray(matrix2.first().size)})
    for (i in 0 until matrix1.size)
        for (j in 0 until matrix2.first().size)
            for (k in 0 until matrix1.first().size)
                product[i][j] += matrix1[i][k] * matrix2[k][j]
    return product
}

fun displayMatrix(product: Array<IntArray>) {
    println("Product of two matrices is: ")
    for (row in product) {
        for (column in row)
            print("$column    ")
        println()
    }
}

/*fun main(args: Array<String>) {
    val r1 = 2
    val c1 = 3
    val r2 = 3
    val c2 = 2
    val firstMatrix = arrayOf(intArrayOf(3, -2, 5), intArrayOf(3, 0, 4))
    val secondMatrix = arrayOf(intArrayOf(2, 3), intArrayOf(-9, 0), intArrayOf(0, 4))

    // Mutliplying Two matrices
    val product = multiplyMatrices(firstMatrix, secondMatrix, r1, c1, c2)

    // Displaying the result
    displayProduct(product)
}

fun multiplyMatrices(firstMatrix: Array, secondMatrix: Array, r1: Int, c1: Int, c2: Int): Array {
    val product = Array(r1) { IntArray(c2) }
    for (i in 0..r1 - 1) {
        for (j in 0..c2 - 1) {
            for (k in 0..c1 - 1) {
                product[i][j] += firstMatrix[i][k] * secondMatrix[k][j]
            }
        }
    }

    return product
}

fun displayProduct(product: Array) {
    println("Product of two matrices is: ")
    for (row in product) {
        for (column in row) {
            print("$column    ")
        }
        println()
    }
}*/