import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkException

import org.apache.spark.sql._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, StructType}

import org.apache.spark.ml.util.{MetadataUtils, SchemaUtils}

import java.nio.ByteBuffer
import java.util.Random
import java.util.{Random => JavaRandom}

import scala.util.hashing.MurmurHash3

import breeze.linalg.{*, axpy => Baxpy, DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, Vector => BV, sum => Bsum, norm}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import breeze.numerics.{log => brzlog}

import com.github.fommil.netlib.{BLAS => NetlibBLAS, F2jBLAS}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector, SparseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import org.apache.spark.internal.Logging

import org.apache.log4j.{Level, LogManager, PropertyConfigurator}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object MultiLayerPerceptron {

  var mm_time: Double = 0.0
  var tick: Double = 0.0
  var tok : Double = 0.0

  @transient lazy val log = org.apache.log4j.LogManager.getLogger("myLogger")

  class XORShiftRandom(init: Long) extends JavaRandom(init) {
    def this() = this(System.nanoTime)

    var seed = XORShiftRandom.hashSeed(init)

    // we need to just override next - this will be called by nextInt, nextDouble,
    // nextGaussian, nextLong, etc.
    override protected def next(bits: Int): Int = {
      var nextSeed = seed ^ (seed << 21)
      nextSeed ^= (nextSeed >>> 35)
      nextSeed ^= (nextSeed << 4)
      seed = nextSeed
      (nextSeed & ((1L << bits) -1)).asInstanceOf[Int]
    }

    override def setSeed(s: Long) {
      seed = XORShiftRandom.hashSeed(s)
    }
  }

  object XORShiftRandom {
    /** Hash seeds to have 0/1 bits throughout. */
    def hashSeed(seed: Long): Long = {
      val bytes = ByteBuffer.allocate(java.lang.Long.SIZE).putLong(seed).array()
      val lowBits = MurmurHash3.bytesHash(bytes)
      val highBits = MurmurHash3.bytesHash(bytes, lowBits)
      (highBits.toLong << 32) | (lowBits.toLong & 0xFFFFFFFFL)
    }

  }

  object VectorUtil {
    def toDenseVector(v: Vector): DenseVector = v match {
      case dense: DenseVector => dense
      case sparse: SparseVector => sparse.toDense
    }

    def toLinalgVector(v: org.apache.spark.ml.linalg.Vector): Vector = v match {
      case dense: org.apache.spark.ml.linalg.DenseVector => {
        Vectors.dense(dense.values)
      } 
      case sparse: org.apache.spark.ml.linalg.SparseVector => {
        val dv = sparse.toDense
        Vectors.dense(dv.values)
      }
    }
  }

  object BreezeUtil {

    var _f2jBLAS: NetlibBLAS = _
    var _nativeBLAS: NetlibBLAS = _

    def f2jBLAS: NetlibBLAS = {
      if (_f2jBLAS == null) {
        _f2jBLAS = new F2jBLAS
      }
      _f2jBLAS
    }

    def nativeBLAS: NetlibBLAS = {
      if (_nativeBLAS == null) {
        _nativeBLAS = NativeBLAS
      }
      _nativeBLAS
    }

    def asBreeze(vector: Vector): BDV[Double] = {
      val dv = VectorUtil.toDenseVector(vector)
      val result = new BDV[Double](dv.values)
      result
    } 

    //def apply(i: Int): Double = asBreeze(i)

    def fromBreeze(breezeVector: BV[Double]): Vector = {
      breezeVector match {
        case v: BDV[Double] =>
          if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
            new DenseVector(v.data)
          } else {
            new DenseVector(v.toArray)  // Can't use underlying array directly, so make a new one
          }
        case v: BSV[Double] =>
          if (v.index.length == v.used) {
            new SparseVector(v.length, v.index, v.data)
          } else {
            new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
          }
        case v: BV[_] =>
          sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
      }
    }

    def axpy(a: Double, x: Vector, y: Vector): Unit = {
      val dx = VectorUtil.toDenseVector(x)
      val dy = VectorUtil.toDenseVector(y)
      val n = x.size
      this.f2jBLAS.daxpy(n, a, dx.values, 1, dy.values, 1)
    }

    def transposeString(A: BDM[Double]): String = if (A.isTranspose) "T" else "N"

    /**
     * DGEMM: C := alpha * A * B + beta * C
     * @param alpha alpha
     * @param A Matrix
     * @param B Matrix
     * @param beta beta
     * @param C Matrix
     */
    def dgemm(alpha: Double, A: BDM[Double], B: BDM[Double], beta: Double, C: BDM[Double]): Unit = {
      // TODO: add code if matrices isTranspose!!!
      require(A.cols == B.rows, "A & B Dimension mismatch!")
      require(A.rows == C.rows, "A & C Dimension mismatch!")
      require(B.cols == C.cols, "A & C Dimension mismatch!")
      NativeBLAS.dgemm(transposeString(A), transposeString(B), C.rows, C.cols, A.cols,
        alpha, A.data, A.offset, A.majorStride, B.data, B.offset, B.majorStride,
        beta, C.data, C.offset, C.rows)
    }

    /**
     * DGEMV: y := alpha * A * x + beta * y
     * @param alpha alpha
     * @param A Matrix
     * @param x Vector
     * @param beta beta
     * @param y Vector
     */
    def dgemv(alpha: Double, A: BDM[Double], x: BDV[Double], beta: Double, y: BDV[Double]): Unit = {
      require(A.cols == x.length, "A & x Dimension mismatch!")
      require(A.rows == y.length, "A & y Dimension mismatch!")
      NativeBLAS.dgemv(transposeString(A), A.rows, A.cols,
        alpha, A.data, A.offset, A.majorStride, x.data, x.offset, x.stride,
        beta, y.data, y.offset, y.stride)
    }
  }

  abstract class Updater extends Serializable {
    /**
     * Compute an updated value for weights given the gradient, stepSize, iteration number and
     * regularization parameter. Also returns the regularization value regParam * R(w)
     * computed using the *updated* weights.
     *
     * @param weightsOld - Column matrix of size dx1 where d is the number of features.
     * @param gradient - Column matrix of size dx1 where d is the number of features.
     * @param stepSize - step size across iterations
     * @param iter - Iteration number
     * @param regParam - Regularization parameter
     *
     * @return A tuple of 2 elements. The first element is a column matrix containing updated weights,
     *         and the second element is the regularization value computed using updated weights.
     */
    def compute(
        weightsOld: Vector,
        gradient: Vector,
        stepSize: Double,
        iter: Int,
        regParam: Double): (Vector, Double)
  }


  trait Optimizer extends Serializable {

    /**
     * Solve the provided convex optimization problem.
     */
    def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector
  }

  abstract class Gradient extends Serializable {
    /**
     * Compute the gradient and loss given the features of a single data point.
     *
     * @param data features for one data point
     * @param label label for this data point
     * @param weights weights/coefficients corresponding to features
     *
     * @return (gradient: Vector, loss: Double)
     */
    def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
      val gradient = Vectors.zeros(weights.size)
      val loss = compute(data, label, weights, gradient)
      (gradient, loss)
    }

    /**
     * Compute the gradient and loss given the features of a single data point,
     * add the gradient to a provided vector to avoid creating new objects, and return loss.
     *
     * @param data features for one data point
     * @param label label for this data point
     * @param weights weights/coefficients corresponding to features
     * @param cumGradient the computed gradient will be added to this vector
     *
     * @return loss
     */
    def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double

    def getGradientMatMulTime() : Double
  }

  /**
   * GradientDescent
  */
  class GradientDescent(private var gradient: Gradient, private var updater: Updater) extends Optimizer {
    private var stepSize: Double = 1.0
    private var numIterations: Int = 100
    private var regParam: Double = 0.0
    private var miniBatchFraction: Double = 1.0
    private var convergenceTol: Double = 0.001

    /**
     * Set the initial step size of SGD for the first step. Default 1.0.
     * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
     */
    def setStepSize(step: Double): this.type = {
      require(step > 0,
        s"Initial step size must be positive but got ${step}")
      this.stepSize = step
      this
    }

    /**
     * Set fraction of data to be used for each SGD iteration.
     * Default 1.0 (corresponding to deterministic/classical gradient descent)
     */
    def setMiniBatchFraction(fraction: Double): this.type = {
      require(fraction > 0 && fraction <= 1.0,
        s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
      this.miniBatchFraction = fraction
      this
    }

    /**
     * Set the number of iterations for SGD. Default 100.
     */
    def setNumIterations(iters: Int): this.type = {
      require(iters >= 0,
        s"Number of iterations must be nonnegative but got ${iters}")
      this.numIterations = iters
      this
    }

    /**
     * Set the regularization parameter. Default 0.0.
     */
    def setRegParam(regParam: Double): this.type = {
      require(regParam >= 0,
        s"Regularization parameter must be nonnegative but got ${regParam}")
      this.regParam = regParam
      this
    }

    /**
     * Set the convergence tolerance. Default 0.001
     * convergenceTol is a condition which decides iteration termination.
     * The end of iteration is decided based on below logic.
     *
     *  - If the norm of the new solution vector is greater than 1, the diff of solution vectors
     *    is compared to relative tolerance which means normalizing by the norm of
     *    the new solution vector.
     *  - If the norm of the new solution vector is less than or equal to 1, the diff of solution
     *    vectors is compared to absolute tolerance which is not normalizing.
     *
     * Must be between 0.0 and 1.0 inclusively.
     */
    def setConvergenceTol(tolerance: Double): this.type = {
      require(tolerance >= 0.0 && tolerance <= 1.0,
        s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
      this.convergenceTol = tolerance
      this
    }

    /**
     * Set the gradient function (of the loss function of one single data example)
     * to be used for SGD.
     */
    def setGradient(gradient: Gradient): this.type = {
      this.gradient = gradient
      this
    }


    /**
     * Set the updater function to actually perform a gradient step in a given direction.
     * The updater is responsible to perform the update from the regularization term as well,
     * and therefore determines what kind or regularization is used, if any.
     */
    def setUpdater(updater: Updater): this.type = {
      this.updater = updater
      this
    }
    /**
     * :: DeveloperApi ::
     * Runs gradient descent on the given training data.
     * @param data training data
     * @param initialWeights initial weights
     * @return solution vector
     */
    def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
      val (weights, _) = GradientDescent.runMiniBatchSGD(
        data,
        gradient,
        updater,
        stepSize,
        numIterations,
        regParam,
        miniBatchFraction,
        initialWeights,
        convergenceTol)
      weights
    }
  }
  /**
   * :: DeveloperApi ::
   * Top-level method to run gradient descent.
   */
  object GradientDescent extends Logging {
    /**
   * Run stochastic gradient descent (SGD) in parallel using mini batches.
   * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
   * in order to compute a gradient estimate.
   * Sampling, and averaging the subgradients over this subset is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data Input data for SGD. RDD of the set of data examples, each of
   *             the form (label, [feature values]).
   * @param gradient Gradient object (used to compute the gradient of the loss function of
   *                 one single data example)
   * @param updater Updater function to actually perform a gradient step in a given direction.
   * @param stepSize initial step size for the first step
   * @param numIterations number of iterations that SGD should be run.
   * @param regParam regularization parameter
   * @param miniBatchFraction fraction of the input data set that should be used for
   *                          one iteration of SGD. Default value 1.0.
   * @param convergenceTol Minibatch iteration will end before numIterations if the relative
   *                       difference between the current weight and the previous weight is less
   *                       than this value. In measuring convergence, L2 norm is calculated.
   *                       Default value 0.001. Must be between 0.0 and 1.0 inclusively.
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the
   *         stochastic loss computed for every iteration.
   */
    def runMiniBatchSGD(
        data: RDD[(Double, Vector)],
        gradient: Gradient,
        updater: Updater,
        stepSize: Double,
        numIterations: Int,
        regParam: Double,
        miniBatchFraction: Double,
        initialWeights: Vector,
        convergenceTol: Double): (Vector, Array[Double]) = {

      // convergenceTol should be set with non minibatch settings
      if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
        logWarning("Testing against a convergenceTol when using miniBatchFraction " +
          "< 1.0 can be unstable because of the stochasticity in sampling.")
      }

      if (numIterations * miniBatchFraction < 1.0) {
        logWarning("Not all examples will be used if numIterations * miniBatchFraction < 1.0: " +
          s"numIterations=$numIterations and miniBatchFraction=$miniBatchFraction")
      }

      val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
      // Record previous weight and current one to calculate solution vector difference

      var previousWeights: Option[Vector] = None
      var currentWeights: Option[Vector] = None

      val numExamples = data.count()

      println("Number of Data : " + numExamples + " Number of Iterations : " + numIterations)

      // if no data, return initial weights to avoid NaNs
      if (numExamples == 0) {
        logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
        return (initialWeights, stochasticLossHistory.toArray)
      }

      if (numExamples * miniBatchFraction < 1) {
        logWarning("The miniBatchFraction is too small")
      }

      // Initialize weights as a column vector
      var weights = Vectors.dense(initialWeights.toArray)
      val n = weights.size

      /**
       * For the first iteration, the regVal will be initialized as sum of weight squares
       * if it's L2 updater; for L1 updater, the same logic is followed.
       */
      var regVal = updater.compute(
        weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

      var converged = false // indicates whether converged based on convergenceTol
      var i = 1
      //while (!converged && i <= numIterations) {
      while (i <= numIterations) {
        val bcWeights = data.context.broadcast(weights)
        // Sample a subset (fraction miniBatchFraction) of the total data
        // compute and sum up the subgradients on this subset (this is one map-reduce)
        tick = System.nanoTime()
        var numComputeGradient: Int = 0
        var compute_total_time: Double = 0.0
        var matmul_total_time: Double = 0.0
        val (gradientSum, lossSum, miniBatchSize) = data.sample(false, miniBatchFraction, 42 + i)
          .treeAggregate((BDV.zeros[Double](n), 0.0, 0L))(
            seqOp = (c, v) => {
              // c: (grad, loss, count), v: (label, features)
              val tick_compute = System.nanoTime()
              val l = gradient.compute(v._2, v._1, bcWeights.value, BreezeUtil.fromBreeze(c._1))      
              val tok_compute = (System.nanoTime() - tick_compute) / 1e9
              compute_total_time += tok_compute
              logWarning("idx : " + numComputeGradient + " Compute Gradient Time : " + tok_compute + " sec")
              val gradient_matmul_time = gradient.getGradientMatMulTime()
              matmul_total_time += gradient_matmul_time
              println("[*] idx : " + numComputeGradient)
              println("Compute Total Time : " + compute_total_time + " sec")
              println("MatMul Total Time : " + gradient_matmul_time + " sec")
              numComputeGradient += 1
              (c._1, c._2 + l, c._3 + 1)
            },
            combOp = (c1, c2) => {
              // c: (grad, loss, count)
              (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
            })
        tok = (System.nanoTime() - tick) / 1e9
        println("Compute Gradient Time : " + tok + " sec")
        mm_time += tok
        
        bcWeights.destroy()
        
        println("[-] idx : " + i + " / Loss Sum : " + lossSum + " / miniBatchSize : " + miniBatchSize )
        
        if (miniBatchSize > 0) {
          /**
           * lossSum is computed using the weights from the previous iteration
           * and regVal is the regularization value computed in the previous iteration as well.
           */
          stochasticLossHistory += lossSum / miniBatchSize + regVal
          val update = updater.compute(
            weights, BreezeUtil.fromBreeze(gradientSum / miniBatchSize.toDouble),
            stepSize, i, regParam)
          weights = update._1
          regVal = update._2

          previousWeights = currentWeights
          currentWeights = Some(weights)
          if (previousWeights != None && currentWeights != None) {
            converged = isConverged(previousWeights.get,
              currentWeights.get, convergenceTol)
          }
        } else {
          logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
        }
        i += 1
      }
      logWarning("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
        stochasticLossHistory.takeRight(10).mkString(", ")))
      (weights, stochasticLossHistory.toArray)
    }

    /**
     * Alias of `runMiniBatchSGD` with convergenceTol set to default value of 0.001.
     */
    def runMiniBatchSGD(
        data: RDD[(Double, Vector)],
        gradient: Gradient,
        updater: Updater,
        stepSize: Double,
        numIterations: Int,
        regParam: Double,
        miniBatchFraction: Double,
        initialWeights: Vector): (Vector, Array[Double]) =
      GradientDescent.runMiniBatchSGD(data, gradient, updater, stepSize, numIterations,
                                      regParam, miniBatchFraction, initialWeights, 0.001)


    private def isConverged(
        previousWeights: Vector,
        currentWeights: Vector,
        convergenceTol: Double): Boolean = {
      // To compare with convergence tolerance.
      val previousBDV = BreezeUtil.asBreeze(previousWeights)
      val currentBDV = BreezeUtil.asBreeze(currentWeights)

      // This represents the difference of updated weights in the iteration.
      val solutionVecDiff: Double = norm(previousBDV - currentBDV)

      solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
    }

  }
  /**
   * :: DeveloperApi ::
   * Class used to solve an optimization problem using Limited-memory BFGS.
   * Reference: <a href="http://en.wikipedia.org/wiki/Limited-memory_BFGS">
   * Wikipedia on Limited-memory BFGS</a>
   * @param gradient Gradient function to be used.
   * @param updater Updater to be used to update weights after every iteration.
   */
  class LBFGS(private var gradient: Gradient, private var updater: Updater) extends Optimizer {
    private var numCorrections = 10
    private var convergenceTol = 1E-6
    private var maxNumIterations = 200
    private var regParam = 0.0

    /**
     * Set the number of corrections used in the LBFGS update. Default 10.
     * Values of numCorrections less than 3 are not recommended; large values
     * of numCorrections will result in excessive computing time.
     * numCorrections must be positive, and values from 4 to 9 are generally recommended.
     */
    def setNumCorrections(corrections: Int): this.type = {
      require(corrections > 0,
        s"Number of corrections must be positive but got ${corrections}")
      this.numCorrections = corrections
      this
    }

    /**
     * Set the convergence tolerance of iterations for L-BFGS. Default 1E-6.
     * Smaller value will lead to higher accuracy with the cost of more iterations.
     * This value must be nonnegative. Lower convergence values are less tolerant
     * and therefore generally cause more iterations to be run.
     */
    def setConvergenceTol(tolerance: Double): this.type = {
      require(tolerance >= 0,
        s"Convergence tolerance must be nonnegative but got ${tolerance}")
      this.convergenceTol = tolerance
      this
    }

    /*
     * Get the convergence tolerance of iterations.
     */
    def getConvergenceTol(): Double = {
      this.convergenceTol
    }

    /**
     * Set the maximal number of iterations for L-BFGS. Default 100.
     */
    def setNumIterations(iters: Int): this.type = {
      require(iters >= 0,
        s"Maximum of iterations must be nonnegative but got ${iters}")
      this.maxNumIterations = iters
      this
    }

    /**
     * Get the maximum number of iterations for L-BFGS. Defaults to 100.
     */
    def getNumIterations(): Int = {
      this.maxNumIterations
    }

    /**
     * Set the regularization parameter. Default 0.0.
     */
    def setRegParam(regParam: Double): this.type = {
      require(regParam >= 0,
        s"Regularization parameter must be nonnegative but got ${regParam}")
      this.regParam = regParam
      this
    }

    /**
     * Get the regularization parameter.
     */
    def getRegParam(): Double = {
      this.regParam
    }

    /**
     * Set the gradient function (of the loss function of one single data example)
     * to be used for L-BFGS.
     */
    def setGradient(gradient: Gradient): this.type = {
      this.gradient = gradient
      this
    }

    /**
     * Set the updater function to actually perform a gradient step in a given direction.
     * The updater is responsible to perform the update from the regularization term as well,
     * and therefore determines what kind or regularization is used, if any.
     */
    def setUpdater(updater: Updater): this.type = {
      this.updater = updater
      this
    }

    /**
     * Returns the updater, limited to internal use.
     */
    def getUpdater(): Updater = {
      updater
    }

    def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
      val (weights, _) = LBFGS.runLBFGS(
        data,
        gradient,
        updater,
        numCorrections,
        convergenceTol,
        maxNumIterations,
        regParam,
        initialWeights)
      weights
    }
  }

  object LBFGS {
    /**
     * Run Limited-memory BFGS (L-BFGS) in parallel.
     * Averaging the subgradients over different partitions is performed using one standard
     * spark map-reduce in each iteration.
     *
     * @param data - Input data for L-BFGS. RDD of the set of data examples, each of
     *               the form (label, [feature values]).
     * @param gradient - Gradient object (used to compute the gradient of the loss function of
     *                   one single data example)
     * @param updater - Updater function to actually perform a gradient step in a given direction.
     * @param numCorrections - The number of corrections used in the L-BFGS update.
     * @param convergenceTol - The convergence tolerance of iterations for L-BFGS which is must be
     *                         nonnegative. Lower values are less tolerant and therefore generally
     *                         cause more iterations to be run.
     * @param maxNumIterations - Maximal number of iterations that L-BFGS can be run.
     * @param regParam - Regularization parameter
     *
     * @return A tuple containing two elements. The first element is a column matrix containing
     *         weights for every feature, and the second element is an array containing the loss
     *         computed for every iteration.
     */
    def runLBFGS(
        data: RDD[(Double, Vector)],
        gradient: Gradient,
        updater: Updater,
        numCorrections: Int,
        convergenceTol: Double,
        maxNumIterations: Int,
        regParam: Double,
        initialWeights: Vector): (Vector, Array[Double]) = {

      val lossHistory = mutable.ArrayBuilder.make[Double]

      val numExamples = data.count()

      val costFun =
        new CostFun(data, gradient, updater, regParam, numExamples)

      val lbfgs = new BreezeLBFGS[BDV[Double]](maxNumIterations, numCorrections, convergenceTol)

      val states =
        lbfgs.iterations(new CachedDiffFunction(costFun), BreezeUtil.asBreeze(initialWeights))

      /**
       * NOTE: lossSum and loss is computed using the weights from the previous iteration
       * and regVal is the regularization value computed in the previous iteration as well.
       */
      var state = states.next()
      while (states.hasNext) {
        state = states.next()
      }

      lossHistory += state.value

      val lossHistoryArray = lossHistory.result()

      val weights = BreezeUtil.fromBreeze(state.x)

      (weights, lossHistoryArray)
    }

    /**
     * CostFun implements Breeze's DiffFunction[T], which returns the loss and gradient
     * at a particular point (weights). It's used in Breeze's convex optimization routines.
     */
    class CostFun(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      regParam: Double,
      numExamples: Long) extends DiffFunction[BDV[Double]] {

      override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
        // Have a local copy to avoid the serialization of CostFun object which is not serializable.
        val w = BreezeUtil.fromBreeze(weights)
        val n = w.size
        val bcW = data.context.broadcast(w)
        val localGradient = gradient

        val seqOp = (c: (Vector, Double), v: (Double, Vector)) =>
          (c, v) match {
            case ((grad, loss), (label, features)) =>
              val denseGrad = grad.toDense
              val l = localGradient.compute(features, label, bcW.value, denseGrad)
              (denseGrad, loss + l)
          }

        val combOp = (c1: (Vector, Double), c2: (Vector, Double)) =>
          (c1, c2) match { case ((grad1, loss1), (grad2, loss2)) =>
            val denseGrad1 = grad1.toDense
            val denseGrad2 = grad2.toDense
            BreezeUtil.axpy(1.0, denseGrad2, denseGrad1)
            (denseGrad1, loss1 + loss2)
         }

        val zeroSparseVector = Vectors.sparse(n, Seq())
        val (gradientSum, lossSum) = data.treeAggregate((zeroSparseVector, 0.0))(seqOp, combOp)

        // broadcasted model is not needed anymore
        bcW.destroy()//blocking = false)

        /**
         * regVal is sum of weight squares if it's L2 updater;
         * for other updater, the same logic is followed.
         */
        val regVal = updater.compute(w, Vectors.zeros(n), 0, 1, regParam)._2

        val loss = lossSum / numExamples + regVal
        /**
         * It will return the gradient part of regularization using updater.
         *
         * Given the input parameters, the updater basically does the following,
         *
         * w' = w - thisIterStepSize * (gradient + regGradient(w))
         * Note that regGradient is function of w
         *
         * If we set gradient = 0, thisIterStepSize = 1, then
         *
         * regGradient(w) = w - w'
         *
         * TODO: We need to clean it up by separating the logic of regularization out
         *       from updater to regularizer.
         */
        // The following gradientTotal is actually the regularization part of gradient.
        // Will add the gradientSum computed from the data with weights in the next step.
        val gradientTotal = w.copy
        BreezeUtil.axpy(-1.0, updater.compute(w, Vectors.zeros(n), 1, 1, regParam)._1, gradientTotal)

        // gradientTotal = gradientSum / numExamples + gradientTotal
        BreezeUtil.axpy(1.0 / numExamples, gradientSum, gradientTotal)

        (loss, BreezeUtil.asBreeze(gradientTotal).asInstanceOf[BDV[Double]])
      }
    }
  }

  /**
   * Trait that holds Layer properties, that are needed to instantiate it.
   * Implements Layer instantiation.
   *
   */
  trait Layer extends Serializable {

    /**
     * Number of weights that is used to allocate memory for the weights vector
     */
    val weightSize: Int

    /**
     * Returns the output size given the input size (not counting the stack size).
     * Output size is used to allocate memory for the output.
     *
     * @param inputSize input size
     * @return output size
     */
    def getOutputSize(inputSize: Int): Int

    /**
     * If true, the memory is not allocated for the output of this layer.
     * The memory allocated to the previous layer is used to write the output of this layer.
     * Developer can set this to true if computing delta of a previous layer
     * does not involve its output, so the current layer can write there.
     * This also mean that both layers have the same number of outputs.
     */
    val inPlace: Boolean

    /**
     * Returns the instance of the layer based on weights provided.
     * Size of weights must be equal to weightSize
     *
     * @param initialWeights vector with layer weights
     * @return the layer model
     */
    def createModel(initialWeights: BDV[Double]): LayerModel

    /**
     * Returns the instance of the layer with random generated weights.
     *
     * @param weights vector for weights initialization, must be equal to weightSize
     * @param random random number generator
     * @return the layer model
     */
    def initModel(weights: BDV[Double], random: Random): LayerModel
  }

  /**
   * Trait that holds Layer weights (or parameters).
   * Implements functions needed for forward propagation, computing delta and gradient.
   * Can return weights in Vector format.
   */
  trait LayerModel extends Serializable {

    val weights: BDV[Double]
    /**
     * Evaluates the data (process the data through the layer).
     * Output is allocated based on the size provided by the
     * LayerModel implementation and the stack (batch) size.
     * Developer is responsible for checking the size of output
     * when writing to it.
     *
     * @param data data
     * @param output output (modified in place)
     */
    def eval(data: BDM[Double], output: BDM[Double]): Unit

    /**
     * Computes the delta for back propagation.
     * Delta is allocated based on the size provided by the
     * LayerModel implementation and the stack (batch) size.
     * Developer is responsible for checking the size of
     * prevDelta when writing to it.
     *
     * @param delta delta of this layer
     * @param output output of this layer
     * @param prevDelta the previous delta (modified in place)
     */
    def computePrevDelta(delta: BDM[Double], output: BDM[Double], prevDelta: BDM[Double]): Unit

    /**
     * Computes the gradient.
     * cumGrad is a wrapper on the part of the weight vector.
     * Size of cumGrad is based on weightSize provided by
     * implementation of LayerModel.
     *
     * @param delta delta for this layer
     * @param input input data
     * @param cumGrad cumulative gradient (modified in place)
     */
    def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit

    def getMMLayerTime(): Double
  }

  /**
   * Layer properties of affine transformations, that is y=A*x+b
   *
   * @param numIn number of inputs
   * @param numOut number of outputs
   */
  class AffineLayer(val numIn: Int, val numOut: Int) extends Layer {

    override val weightSize = numIn * numOut + numOut

    override def getOutputSize(inputSize: Int): Int = numOut

    override val inPlace = false

    override def createModel(weights: BDV[Double]): LayerModel = new AffineLayerModel(weights, this)

    override def initModel(weights: BDV[Double], random: Random): LayerModel =
      AffineLayerModel(this, weights, random)
  }

  /**
   * Model of Affine layer
   *
   * @param weights weights
   * @param layer layer properties
   */
  class AffineLayerModel (
      val weights: BDV[Double],
      val layer: AffineLayer) extends LayerModel with Logging {
    val w = new BDM[Double](layer.numOut, layer.numIn, weights.data, weights.offset)
    val b =
      new BDV[Double](weights.data, weights.offset + (layer.numOut * layer.numIn), 1, layer.numOut)

    private var ones: BDV[Double] = null

    var mm_layer_time: Double = 0.0

    def getMMLayerTime(): Double = mm_layer_time

    override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
      output(::, *) := b
      val tick = System.nanoTime()
      BreezeUtil.dgemm(1.0, w, data, 1.0, output)
      val tok = (System.nanoTime() - tick) / 1e9
      mm_layer_time += tok
      //logWarning("[*] Eval MM Time : " + tok + "sec")
      //logWarning("   - Weight Matrix row : (" + w.rows + "," + w.cols + ")")
      //logWarning("   - Data Matrix row : (" + data.rows + "," + data.cols + ")")
    }

    override def computePrevDelta(
      delta: BDM[Double],
      output: BDM[Double],
      prevDelta: BDM[Double]): Unit = {
      val tick = System.nanoTime()
      BreezeUtil.dgemm(1.0, w.t, delta, 0.0, prevDelta)
      val tok = (System.nanoTime() - tick) / 1e9
      mm_layer_time += tok
      //logWarning("[*] Compute Prev Delta MM Time : " + tok + "sec")
      //logWarning("   - Weight Matrix row : (" + w.t.rows + "," + w.t.cols + ")")
      //logWarning("   - Delta Matrix row : (" + delta.rows + "," + delta.cols + ")")
    }

    override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {
      // compute gradient of weights
      val cumGradientOfWeights = new BDM[Double](w.rows, w.cols, cumGrad.data, cumGrad.offset)
      val tick = System.nanoTime()
      BreezeUtil.dgemm(1.0 / input.cols, delta, input.t, 1.0, cumGradientOfWeights)
      val tok = (System.nanoTime() - tick) / 1e9
      mm_layer_time += tok
      //logWarning("[*] Gradient MM Time : " + tok + "sec")
      //logWarning("   - Weight Matrix row : (" + delta.rows + "," + delta.cols + ")")
      //logWarning("   - Input Matrix row : (" + input.t.rows + "," + input.t.cols + ")")

      if (ones == null || ones.length != delta.cols) ones = BDV.ones[Double](delta.cols)
      // compute gradient of bias
      val cumGradientOfBias = new BDV[Double](cumGrad.data, cumGrad.offset + w.size, 1, b.length)
      BreezeUtil.dgemv(1.0 / input.cols, delta, ones, 1.0, cumGradientOfBias)
    }
  }

  /**
   * Fabric for Affine layer models
   */
  object AffineLayerModel {

    /**
     * Creates a model of Affine layer
     *
     * @param layer layer properties
     * @param weights vector for weights initialization
     * @param random random number generator
     * @return model of Affine layer
     */
    def apply(layer: AffineLayer, weights: BDV[Double], random: Random): AffineLayerModel = {
      randomWeights(layer.numIn, layer.numOut, weights, random)
      new AffineLayerModel(weights, layer)
    }

    /**
     * Initialize weights randomly in the interval.
     * Uses [Bottou-88] heuristic [-a/sqrt(in); a/sqrt(in)],
     * where `a` is chosen in such a way that the weight variance corresponds
     * to the points to the maximal curvature of the activation function
     * (which is approximately 2.38 for a standard sigmoid).
     *
     * @param numIn number of inputs
     * @param numOut number of outputs
     * @param weights vector for weights initialization
     * @param random random number generator
     */
    def randomWeights(
      numIn: Int,
      numOut: Int,
      weights: BDV[Double],
      random: Random): Unit = {
      var i = 0
      val sqrtIn = math.sqrt(numIn)
      while (i < weights.length) {
        weights(i) = (random.nextDouble * 4.8 - 2.4) / sqrtIn
        i += 1
      }
    }
  }

  object ApplyInPlace {
    def apply(x: BDM[Double], y: BDM[Double], func: Double => Double): Unit = {
      var i = 0
      while (i < x.rows) {
        var j = 0
        while (j < x.cols) {
          y(i, j) = func(x(i, j))
          j += 1
        }
        i += 1
      }
    }

    def apply(
      x1: BDM[Double],
      x2: BDM[Double],
      y: BDM[Double],
      func: (Double, Double) => Double): Unit = {
      var i = 0
      while (i < x1.rows) {
        var j = 0
        while (j < x1.cols) {
          y(i, j) = func(x1(i, j), x2(i, j))
          j += 1
        }
        i += 1
      }
    }
  }

  class SoftmaxLayerWithCrossEntropyLoss extends Layer {
    override val weightSize = 0
    override val inPlace = true

    override def getOutputSize(inputSize: Int): Int = inputSize
    override def createModel(weights: BDV[Double]): LayerModel =
      new SoftmaxLayerModelWithCrossEntropyLoss()
    override def initModel(weights: BDV[Double], random: Random): LayerModel =
      new SoftmaxLayerModelWithCrossEntropyLoss()
  }

  class SoftmaxLayerModelWithCrossEntropyLoss extends LayerModel with LossFunction {

    // loss layer models do not have weights
    val weights = new BDV[Double](0)

    override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
      var j = 0
      // find max value to make sure later that exponent is computable
      while (j < data.cols) {
        var i = 0
        var max = Double.MinValue
        while (i < data.rows) {
          if (data(i, j) > max) {
            max = data(i, j)
          }
          i += 1
        }
        var sum = 0.0
        i = 0
        while (i < data.rows) {
          val res = math.exp(data(i, j) - max)
          output(i, j) = res
          sum += res
          i += 1
        }
        i = 0
        while (i < data.rows) {
          output(i, j) /= sum
          i += 1
        }
        j += 1
      }
    }
    override def computePrevDelta(
      nextDelta: BDM[Double],
      input: BDM[Double],
      delta: BDM[Double]): Unit = {
      /* loss layer model computes delta in loss function */
    }

    override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {
      /* loss layer model does not have weights */
    }

    override def loss(output: BDM[Double], target: BDM[Double], delta: BDM[Double]): Double = {
      ApplyInPlace(output, target, delta, (o: Double, t: Double) => o - t)
      -Bsum( target *:* brzlog(output)) / output.cols
    }

    def getMMLayerTime(): Double = 0.0
  }

  /**
   * Trait for functions and their derivatives for functional layers
   */
  trait ActivationFunction extends Serializable {

    /**
     * Implements a function
     */
    def eval: Double => Double

    /**
     * Implements a derivative of a function (needed for the back propagation)
     */
    def derivative: Double => Double
  }

  /**
   * Implements Sigmoid activation function
   */
  class SigmoidFunction extends ActivationFunction {

    override def eval: (Double) => Double = x => 1.0 / (1 + math.exp(-x))

    override def derivative: (Double) => Double = z => (1 - z) * z
  }

  /**
   * Functional layer properties, y = f(x)
   *
   * @param activationFunction activation function
   */
  class FunctionalLayer (val activationFunction: ActivationFunction) extends Layer {

    override val weightSize = 0

    override def getOutputSize(inputSize: Int): Int = inputSize

    override val inPlace = true

    override def createModel(weights: BDV[Double]): LayerModel = new FunctionalLayerModel(this)

    override def initModel(weights: BDV[Double], random: Random): LayerModel =
      createModel(weights)
  }

  /**
   * Functional layer model. Holds no weights.
   *
   * @param layer functional layer
   */
  class FunctionalLayerModel (val layer: FunctionalLayer) extends LayerModel {

    // empty weights
    val weights = new BDV[Double](0)

    override def eval(data: BDM[Double], output: BDM[Double]): Unit = {
      ApplyInPlace(data, output, layer.activationFunction.eval)
    }

    override def computePrevDelta(
      nextDelta: BDM[Double],
      input: BDM[Double],
      delta: BDM[Double]): Unit = {
      ApplyInPlace(input, delta, layer.activationFunction.derivative)
      delta :*= nextDelta
    }

    override def grad(delta: BDM[Double], input: BDM[Double], cumGrad: BDV[Double]): Unit = {}

    override def getMMLayerTime(): Double = 0.0
  }

  /**
   * Trait for loss function
   */
  trait LossFunction {
    /**
     * Returns the value of loss function.
     * Computes loss based on target and output.
     * Writes delta (error) to delta in place.
     * Delta is allocated based on the outputSize
     * of model implementation.
     *
     * @param output actual output
     * @param target target output
     * @param delta delta (updated in place)
     * @return loss
     */
    def loss(output: BDM[Double], target: BDM[Double], delta: BDM[Double]): Double
  }

  /**
   * Trait for the artificial neural network (ANN) topology properties
   */
  trait Topology extends Serializable {
    def model(weights: Vector): TopologyModel
    def model(seed: Long): TopologyModel
  }

  /**
   * Trait for ANN topology model
   */
  trait TopologyModel extends Serializable {

    val weights: Vector
    /**
     * Array of layers
     */
    val layers: Array[Layer]

    /**
     * Array of layer models
     */
    val layerModels: Array[LayerModel]

    /**
     * Forward propagation
     *
     * @param data input data
     * @param includeLastLayer Include the last layer in the output. In
     *                         MultilayerPerceptronClassifier, the last layer is always softmax;
     *                         the last layer of outputs is needed for class predictions, but not
     *                         for rawPrediction.
     *
     * @return array of outputs for each of the layers
     */
    def forward(data: BDM[Double], includeLastLayer: Boolean): Array[BDM[Double]]

    /**
     * Prediction of the model. See {@link ProbabilisticClassificationModel}
     *
     * @param features input features
     * @return prediction
     */
    def predict(features: Vector): Double

    /**
     * Raw prediction of the model. See {@link ProbabilisticClassificationModel}
     *
     * @param features input features
     * @return raw prediction
     *
     * Note: This interface is only used for classification Model.
     */
    def predictRaw(features: Vector): Vector

    /**
     * Probability of the model. See {@link ProbabilisticClassificationModel}
     *
     * @param rawPrediction raw prediction vector
     * @return probability
     *
     * Note: This interface is only used for classification Model.
     */
    def raw2ProbabilityInPlace(rawPrediction: Vector): Vector

    /**
     * Computes gradient for the network
     *
     * @param data input data
     * @param target target output
     * @param cumGradient cumulative gradient
     * @param blockSize block size
     * @return error
     */
    def computeGradient(data: BDM[Double], target: BDM[Double], cumGradient: Vector,
                        blockSize: Int): Double

    def transform(dataset: Dataset[_]): DataFrame

    def getMMTime(): Double
  }

  /**
   * Feed forward ANN
   *
   * @param layers Array of layers
   */
  class FeedForwardTopology (val layers: Array[Layer]) extends Topology {
    override def model(weights: Vector): TopologyModel = FeedForwardModel(this, weights)

    override def model(seed: Long): TopologyModel = FeedForwardModel(this, seed)
  }

  /**
    Feed Forward ANN
  */
  object FeedForwardTopology {
    /**
     * Creates a feed forward topology from the array of layers
     *
     * @param layers array of layers
     * @return feed forward topology
     */
    def apply(layers: Array[Layer]): FeedForwardTopology = {
      new FeedForwardTopology(layers)
    }

    /**
     * Creates a multi-layer perceptron
     *
     * @param layerSizes sizes of layers including input and output size
     * @param softmaxOnTop whether to use SoftMax or Sigmoid function for an output layer.
     *                Softmax is default
     * @return multilayer perceptron topology
    */
    def multiLayerPerceptron(layerSizes: Array[Int]): FeedForwardTopology = {
      val layers = new Array[Layer]((layerSizes.length - 1) * 2)
      for (i <- 0 until layerSizes.length - 1) {
        layers(i * 2) = new AffineLayer(layerSizes(i), layerSizes(i + 1))
        layers(i * 2 + 1) =
          if (i == layerSizes.length - 2) {
            new SoftmaxLayerWithCrossEntropyLoss()
          } else {
            new FunctionalLayer(new SigmoidFunction())
          }
      }
      FeedForwardTopology(layers)
    }

  }

  /**
   * Model of Feed Forward Neural Network.
   * Implements forward, gradient computation and can return weights in vector format.
   *
   * @param weights network weights
   * @param topology network topology
   */
  class FeedForwardModel (val weights: Vector, val topology: FeedForwardTopology) extends TopologyModel with Logging{
    
    val layers = topology.layers

    val layerModels = new Array[LayerModel](layers.length)

    private var offset = 0
    for (i <- 0 until layers.length) {
      layerModels(i) = layers(i).createModel(
        new BDV[Double](weights.toArray, offset, 1, layers(i).weightSize))
      offset += layers(i).weightSize
    }
    
    var outputs: Array[BDM[Double]] = null
    var deltas: Array[BDM[Double]] = null  

    /** 
     * Forward propagation
     *
     * @param data input data
     * @param includeLastLayer Include the last layer in the output. In
     *                         MultilayerPerceptronClassifier, the last layer is always softmax;
     *                         the last layer of outputs is needed for class predictions, but not
     *                         for rawPrediction.
     *
     * @return array of outputs for each of the layers
     */
    def forward(data: BDM[Double], includeLastLayer: Boolean): Array[BDM[Double]] = {
      // Initialize output arrays for all layers. Special treatment for InPlace
      val currentBatchSize = data.cols
      // TODO: allocate outputs as one big array and then create BDMs from it
      if (outputs == null || outputs(0).cols != currentBatchSize) {
        outputs = new Array[BDM[Double]](layers.length)
        var inputSize = data.rows
        for (i <- 0 until layers.length) {
          if (layers(i).inPlace) {
            outputs(i) = outputs(i - 1)
          } else {
            val outputSize = layers(i).getOutputSize(inputSize)
            outputs(i) = new BDM[Double](outputSize, currentBatchSize)
            inputSize = outputSize
          }
        }
      }
      layerModels(0).eval(data, outputs(0))
      val end = if (includeLastLayer) layerModels.length else layerModels.length - 1
      for (i <- 1 until end) {
        layerModels(i).eval(outputs(i - 1), outputs(i))
      }
      outputs
    }

    /**
     * Computes gradient for the network
     *
     * @param data input data
     * @param target target output
     * @param cumGradient cumulative gradient
     * @param blockSize block size
     * @return error
     */
    def computeGradient(
      data: BDM[Double], 
      target: BDM[Double], 
      cumGradient: Vector,
      blockSize: Int): Double = {
      val outputs = forward(data, true)
      val currentBatchSize = data.cols
      // TODO: allocate deltas as one big array and then create BDMs from it
      if (deltas == null || deltas(0).cols != currentBatchSize) {
        deltas = new Array[BDM[Double]](layerModels.length)
        var inputSize = data.rows
        for (i <- 0 until layerModels.length - 1) {
          val outputSize = layers(i).getOutputSize(inputSize)
          deltas(i) = new BDM[Double](outputSize, currentBatchSize)
          inputSize = outputSize
        }
      }
      val L = layerModels.length - 1
      // TODO: explain why delta of top layer is null (because it might contain loss+layer)
      val loss = layerModels.last match {
        case levelWithError: LossFunction => levelWithError.loss(outputs.last, target, deltas(L - 1))
        case _ =>
          throw new UnsupportedOperationException("Top layer is required to have objective.")
      }
      for (i <- (L - 2) to (0, -1)) {
        layerModels(i + 1).computePrevDelta(deltas(i + 1), outputs(i + 1), deltas(i))
      }
      val cumGradientArray = cumGradient.toArray
      var offset = 0
      for (i <- 0 until layerModels.length) {
        val input = if (i == 0) data else outputs(i - 1)
        layerModels(i).grad(deltas(i), input,
          new BDV[Double](cumGradientArray, offset, 1, layers(i).weightSize))
        mm_time += tok
        offset += layers(i).weightSize
      }
      loss
    }

    /**
     * Prediction of the model. See {@link ProbabilisticClassificationModel}
     *
     * @param features input features
     * @return prediction
     */
    def predict(data: Vector): Double = {
      val size = data.size
      val result = forward(new BDM[Double](size, 1, data.toArray), true)
      LabelConverter.decodeLabel(Vectors.dense(result.last.toArray))
    }

    override def predictRaw(data: Vector): Vector = {
      val result = forward(new BDM[Double](data.size, 1, data.toArray), false)
      Vectors.dense(result(result.length - 2).toArray)
    }

    override def raw2ProbabilityInPlace(data: Vector): Vector = {
      val dataMatrix = new BDM[Double](data.size, 1, data.toArray)
      layerModels.last.eval(dataMatrix, dataMatrix)
      data
    }

    def transform(dataset: Dataset[_]): DataFrame = {
      var outputData = dataset
      import org.apache.spark.ml.linalg.SparseVector
      val predictRawUDF = udf { (f: org.apache.spark.ml.linalg.SparseVector) => {
          val features = new DenseVector(f.toDense.values)
          predict(features.asInstanceOf[Vector])
        }
      }
      outputData = outputData.withColumn("prediction", predictRawUDF(col("features")))
      outputData.toDF
    }

    def getMMTime(): Double = {
      var sum = 0.0
      for (i <- 1 until (layers.length)) {
        val mm_layer_time = layerModels(i).getMMLayerTime()
        sum += mm_layer_time
      }
      sum
    }
  }

  /**
   * Fabric for feed forward ANN models
   */
  object FeedForwardModel {

    /**
     * Creates a model from a topology and weights
     *
     * @param topology topology
     * @param weights weights
     * @return model
     */
    def apply(topology: FeedForwardTopology, weights: Vector): FeedForwardModel = {
      val expectedWeightSize = topology.layers.map(_.weightSize).sum
      require(weights.size == expectedWeightSize,
        s"Expected weight vector of size ${expectedWeightSize} but got size ${weights.size}.")
      new FeedForwardModel(weights, topology)
    }

    /**
     * Creates a model given a topology and seed
     *
     * @param topology topology
     * @param seed seed for generating the weights
     * @return model
     */
    def apply(topology: FeedForwardTopology, seed: Long = 11L): FeedForwardModel = {
      val layers = topology.layers
      val layerModels = new Array[LayerModel](layers.length)
      val weights = BDV.zeros[Double](topology.layers.map(_.weightSize).sum)
      var offset = 0
      val random = new XORShiftRandom(seed)
      for (i <- 0 until layers.length) {
        layerModels(i) = layers(i).
          initModel(new BDV[Double](weights.data, offset, 1, layers(i).weightSize), random)
        offset += layers(i).weightSize
      }
      new FeedForwardModel(BreezeUtil.fromBreeze(weights), topology)
    }
  }

  /**
   * Stacks pairs of training samples (input, output) in one vector allowing them to pass
   * through Optimizer/Gradient interfaces. If stackSize is more than one, makes blocks
   * or matrices of inputs and outputs and then stack them in one vector.
   * This can be used for further batch computations after unstacking.
   *
   * @param stackSize stack size
   * @param inputSize size of the input vectors
   * @param outputSize size of the output vectors
   */
  class DataStacker(stackSize: Int, inputSize: Int, outputSize: Int) extends Serializable {

    /**
     * Stacks the data
     *
     * @param data RDD of vector pairs
     * @return RDD of double (always zero) and vector that contains the stacked vectors
     */
    def stack(data: RDD[(Vector, Vector)]): RDD[(Double, Vector)] = {
      val stackedData = if (stackSize == 1) {
        data.map { v =>
          (0.0,
            BreezeUtil.fromBreeze(BDV.vertcat(
              BreezeUtil.asBreeze(v._1),
              BreezeUtil.asBreeze(v._2)))
            ) }
      } else {
        data.mapPartitions { it =>
          it.grouped(stackSize).map { seq =>
            val size = seq.size
            val bigVector = new Array[Double](inputSize * size + outputSize * size)
            var i = 0
            seq.foreach { case (in, out) =>
              System.arraycopy(in.toArray, 0, bigVector, i * inputSize, inputSize)
              System.arraycopy(out.toArray, 0, bigVector,
                inputSize * size + i * outputSize, outputSize)
              i += 1
            }
            (0.0, Vectors.dense(bigVector))
          }
        }
      }
      stackedData
    }

    /**
     * Unstack the stacked vectors into matrices for batch operations
     *
     * @param data stacked vector
     * @return pair of matrices holding input and output data and the real stack size
     */
    def unstack(data: Vector): (BDM[Double], BDM[Double], Int) = {
      val arrData = data.toArray
      val realStackSize = arrData.length / (inputSize + outputSize)
      val input = new BDM(inputSize, realStackSize, arrData)
      val target = new BDM(outputSize, realStackSize, arrData, inputSize * realStackSize)
      (input, target, realStackSize)
    }
  }

  /**
   * Neural network gradient. Does nothing but calling Model's gradient
   *
   * @param topology topology
   * @param dataStacker data stacker
   */
  class ANNGradient(topology: Topology, dataStacker: DataStacker) extends Gradient with Logging {

    var gradient_matmul_time: Double = 0.0

    override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
      val (input, target, realBatchSize) = dataStacker.unstack(data)
      val model = topology.model(weights)
      val tick = System.nanoTime()
      val gradient = model.computeGradient(input, target, cumGradient, realBatchSize).toDouble
      val tok = (System.nanoTime() - tick) / 1e9
      log.warn("Model : " + model + " Matrix Multiplication Time : " + model.getMMTime)
      gradient_matmul_time += tok
      gradient
    }

    def getGradientMatMulTime() : Double = gradient_matmul_time
  }


  /**
   * Simple updater
   */
  class ANNUpdater extends Updater {
    def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
      val thisIterStepSize = stepSize
      //val brzWeights: BV[Double] = BreezeUtil.asBreeze(weightsOld)
      //Baxpy(-thisIterStepSize, BreezeUtil.asBreeze(gradient), brzWeights)
      BreezeUtil.axpy(-thisIterStepSize, gradient , weightsOld)
      (weightsOld, 0)
    }
  }

  /**
   * MLlib-style trainer class that trains a network given the data and topology
   *
   * @param topology topology of ANN
   * @param inputSize input size
   * @param outputSize output size
   */
  class FeedForwardTrainer(
      topology: Topology,
      val inputSize: Int,
      val outputSize: Int,
      val iteration: Int,
      val stackSize : Int) extends Serializable {
    private var _seed = this.getClass.getName.hashCode.toLong
    private var _weights: Vector = null
    private var _stackSize: Int = stackSize
    private var dataStacker = new DataStacker(stackSize, inputSize, outputSize) 
    private var _gradient: Gradient = new ANNGradient(topology, dataStacker) 
    private var _updater: Updater = new ANNUpdater() 
    //private var optimizer: Optimizer = LBFGSOptimizer.setConvergenceTol(1e-4).setNumIterations(1000)
    private var optimizer: Optimizer = SGDOptimizer.setConvergenceTol(1e-4).setNumIterations(iteration)

    /**
     * Returns seed
     */
    def getSeed: Long = _seed

    /**
     * Sets seed
     */
    def setSeed(value: Long): this.type = {
      _seed = value
      this
    }

    /**
     * Returns weights
     */
    def getWeights: Vector = _weights

    /**
     * Sets weights
     *
     * @param value weights
     * @return trainer
     */
    def setWeights(value: Vector): this.type = {
      _weights = value
      this
    }

    /**
     * Sets the stack size
     *
     * @param value stack size
     * @return trainer
     */
    def setStackSize(value: Int): this.type = {
      _stackSize = value
      dataStacker = new DataStacker(value, inputSize, outputSize)
      this
    }

    /**
     * Sets the SGD optimizer
     *
     * @return SGD optimizer
     */
    def SGDOptimizer: GradientDescent = {
      val sgd = new GradientDescent(_gradient, _updater)
      optimizer = sgd
      sgd
    }

    /**
     * Sets the LBFGS optimizer
     *
     * @return LBGS optimizer
     */
    def LBFGSOptimizer: LBFGS = {
      val lbfgs = new LBFGS(_gradient, _updater)
      optimizer = lbfgs
      lbfgs
    }

    /**
     * Sets the updater
     *
     * @param value updater
     * @return trainer
     */
    def setUpdater(value: Updater): this.type = {
      _updater = value
      updateUpdater(value)
      this
    }

    /**
     * Sets the gradient
     *
     * @param value gradient
     * @return trainer
     */
    def setGradient(value: Gradient): this.type = {
      _gradient = value
      updateGradient(value)
      this
    }
    
    def updateGradient(gradient: Gradient): Unit = {
      optimizer match {
        case lbfgs: LBFGS => lbfgs.setGradient(gradient)
        case sgd: GradientDescent => sgd.setGradient(gradient)
        case other => throw new UnsupportedOperationException(
          s"Only LBFGS and GradientDescent are supported but got ${other.getClass}.")
      }
    }

    def updateUpdater(updater: Updater): Unit = {
      optimizer match {
        case lbfgs: LBFGS => lbfgs.setUpdater(updater)
        case sgd: GradientDescent => sgd.setUpdater(updater)
        case other => throw new UnsupportedOperationException(
          s"Only LBFGS and GradientDescent are supported but got ${other.getClass}.")
      }
    }

    /**
     * Trains the ANN
     *
     * @param data RDD of input and output vector pairs
     * @return model
     */
    def train(data: RDD[(Vector, Vector)]): TopologyModel = {
      val w = if (getWeights == null) {
        // TODO: will make a copy if vector is a subvector of BDV (see Vectors code)
        topology.model(_seed).weights
      } else {
        getWeights
      }

      // TODO: deprecate standard optimizer because it needs Vector
      val trainData = dataStacker.stack(data).map { v =>
        (v._1, v._2)
      }
      val handlePersistence = trainData.getStorageLevel == StorageLevel.NONE
      if (handlePersistence) trainData.persist(StorageLevel.MEMORY_AND_DISK)
      val newWeights = optimizer.optimize(trainData, w)
      //println("Update Weight : " + newWeights)
      if (handlePersistence) trainData.unpersist()
      topology.model(newWeights)
    }
  }

  /** Label to vector converter. */
  object LabelConverter {
  // TODO: Use OneHotEncoder instead
  /**
   * Encodes a label as a vector.
   * Returns a vector of given length with zeroes at all positions
   * and value 1.0 at the position that corresponds to the label.
   *
   * @param labeledPoint labeled point
   * @param labelCount total number of labels
   * @return pair of features and vector encoding of a label
   */
    def encodeLabeledPoint(label: Double, labelCount: Int): Vector = {
      val output = Array.fill(labelCount)(0.0)
      output(label.toInt) = 1.0
      Vectors.dense(output)
    }

    /**
     * Converts a vector to a label.
     * Returns the position of the maximal element of a vector.
     *
     * @param output label encoded with a vector
     * @return label
     */
    def decodeLabel(output: Vector): Double = {
      output.argmax.toDouble
    }
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("MultiLayerPerceptron")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val iterations = args(0).toInt
    val stackSize = args(1).toInt

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    println("[*]------- Start")
    var tik = System.nanoTime()
    val dataset = sqlContext.read.format("libsvm").option("inferSchema", true).option("numFeatures", "780").load("hdfs:///datasets/mnist/mnist.bz2")

    val rows: RDD[Row] = dataset.rdd
    rows.cache
    
    val numOfClass = 10 // mnist
    
    val data: RDD[(Vector, Vector)] = rows.map { case Row(label:Double, features: org.apache.spark.ml.linalg.Vector) => (VectorUtil.toLinalgVector(features), LabelConverter.encodeLabeledPoint(label, numOfClass)) }
    var tok = (System.nanoTime() - tik) / 1e9
    println("Read Dataset and Generate data : " + tok)
    val numData = data.count
    println("Dataset Size = " + numData)

    val tick = System.nanoTime()
    val layers = Array[Int](780, 14*14, 10*10, 5*5, numOfClass)

    val mlpModel = FeedForwardTopology.multiLayerPerceptron(layers)
    val trainer = new FeedForwardTrainer(mlpModel, layers(0), numOfClass, iterations, stackSize)

    val model = trainer.train(data)
    tok = (System.nanoTime() - tick) / 1e9

    println("[*]------- Result")
    println("[*] Total execution time  : " + tok + " sec")

    val result = model.transform(dataset)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
  }
}
