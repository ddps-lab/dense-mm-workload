import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkException
import scala.util.Random

import org.apache.spark.sql._
import org.apache.spark.sql.SQLContext

import org.apache.spark.mllib.linalg.{ Vectors, Vector, DenseVector, SparseVector }
import breeze.linalg._
import breeze.linalg.{ DenseMatrix => BDM, DenseVector => BDV, SparseVector => BSV, Vector => BV, Matrix => BM }
import org.apache.spark.mllib.linalg.distributed.{ CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry, DistributedMatrix, BlockMatrix }
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{ Matrix, DenseMatrix, SparseMatrix, Matrices }
import org.apache.spark.Partitioner

import scala.collection.mutable._

import java.util.NoSuchElementException

import org.apache.spark.mllib.recommendation.Rating

class GridPartitioner(
  val rows: Int,
  val cols: Int,
  val rowsPerPart: Int,
  val colsPerPart: Int) extends Partitioner {

  require(rows > 0)
  require(cols > 0)
  require(rowsPerPart > 0)
  require(colsPerPart > 0)

  private val rowPartitions = math.ceil(rows * 1.0 / rowsPerPart).toInt
  private val colPartitions = math.ceil(cols * 1.0 / colsPerPart).toInt

  override val numPartitions: Int = rowPartitions * colPartitions

  /**
   * Returns the index of the partition the input coordinate belongs to.
   *
   * @param key The partition id i (calculated through this method for coordinate (i, j) in
   *            `simulateMultiply`, the coordinate (i, j) or a tuple (i, j, k), where k is
   *            the inner index used in multiplication. k is ignored in computing partitions.
   * @return The index of the partition, which the coordinate belongs to.
  */
  override def getPartition(key: Any): Int = {
    key match {
      case i: Int => i
      case (i: Int, j: Int) =>
        getPartitionId(i, j)
      case (i: Int, j: Int, _: Int) =>
        getPartitionId(i, j)
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key: $key.")
    }
  }

  /** Partitions sub-matrices as blocks with neighboring sub-matrices. */
  private def getPartitionId(i: Int, j: Int): Int = {
    require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
    require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
    i / rowsPerPart + j / colsPerPart * rowPartitions
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: GridPartitioner =>
        (this.rows == r.rows) && (this.cols == r.cols) &&
          (this.rowsPerPart == r.rowsPerPart) && (this.colsPerPart == r.colsPerPart)
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      rows: java.lang.Integer,
      cols: java.lang.Integer,
      rowsPerPart: java.lang.Integer,
      colsPerPart: java.lang.Integer)
  }
}

object GridPartitioner {

  /** Creates a new [[GridPartitioner]] instance. */
  def apply(rows: Int, cols: Int, rowsPerPart: Int, colsPerPart: Int): GridPartitioner = {
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }

  /** Creates a new [[GridPartitioner]] instance with the input suggested number of partitions. */
  def apply(rows: Int, cols: Int, suggestedNumPartitions: Int): GridPartitioner = {
    require(suggestedNumPartitions > 0)
    val scale = 1.0 / math.sqrt(suggestedNumPartitions)
    val rowsPerPart = math.round(math.max(scale * rows, 1.0)).toInt
    val colsPerPart = math.round(math.max(scale * cols, 1.0)).toInt
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }
}

object MovieLensNMFBlockMatrix {

  var mm_time: Double = 0.0

  var wt: Double = 0.0
  var wta: Double = 0.0
  var wtw: Double = 0.0
  var wtwh: Double = 0.0
  var ht: Double = 0.0
  var aht: Double = 0.0
  var hht: Double = 0.0
  var whht: Double = 0.0
  var elemwise: Double = 0.0

  object Utils {
    def toBreeze(v: Vector): BV[Double] = v match {
      case DenseVector(values) => new BDV[Double](values)
      case SparseVector(size, indices, values) => new BSV[Double](indices, values, size)
    }
    
    def toDenseMatrix(A: Matrix) : DenseMatrix = {
      A match {
        case dense: DenseMatrix => dense
        case sparse: SparseMatrix => sparse.toDense
      } 
    }
    
    def toSparseMatrix(A: Matrix) : SparseMatrix = {
      A match {
        case dense: DenseMatrix => dense.toSparse
        case sparse: SparseMatrix => sparse
      }
    }
    

    def toMatrix(A: DenseMatrix) : Matrix = {
      A
    }

    def asBreeze(A: DenseMatrix): BM[Double] = {
      new BDM[Double](A.numRows, A.numCols, A.values)
    }
    
    def fromBreeze(breeze: BM[Double]): Matrix = {
      breeze match {
        case dm: BDM[Double] =>
          new DenseMatrix(dm.rows, dm.cols, dm.data, dm.isTranspose)
      }
    }
    
    
    def elemWiseblock(A: BlockMatrix,
      B: BlockMatrix,
      C: BlockMatrix): BlockMatrix = {
      //A.blocks join B.blocks join C.blocks
      val newBlocks = A.blocks.cogroup(B.blocks, C.blocks)
      .map { case((blockRowIndex, blockColIndex), (a, b, c)) =>
          val arr = new ArrayBuffer[Array[Double]]()
          val aBDM = this.asBreeze(this.toDenseMatrix(a.head))
          val bBDM = this.asBreeze(this.toDenseMatrix(a.head))
          val cBDM = this.asBreeze(this.toDenseMatrix(a.head))
          val resultBDM = aBDM *:* bBDM /:/ cBDM
          val result: Matrix = this.fromBreeze(resultBDM)
          ((blockRowIndex, blockColIndex), result)
      }
      new BlockMatrix(newBlocks, A.rowsPerBlock, A.colsPerBlock, A.numRows(), A.numCols())
    }
    

    def elemWiseProduct(A: BlockMatrix, 
      B: BlockMatrix): BlockMatrix = {
      if(A.rowsPerBlock == B.rowsPerBlock && A.colsPerBlock == B.colsPerBlock) {
        val newBlocks = A.blocks.cogroup(B.blocks, GridPartitioner(B.numRowBlocks, B.numColBlocks, B.blocks.partitions.length))
        .map { case ((blockRowIndex, blockColIndex), (a, b)) =>
            val arr = new ArrayBuffer[Array[Double]]()
            val matA: DenseMatrix = toDenseMatrix(a.head)
            val matB: DenseMatrix = toDenseMatrix(b.head)
            val matAItr = matA.colIter
            val matBItr = matB.colIter
            while (matAItr.hasNext)
              arr += matAItr.next.toArray.zip(matBItr.next.toArray).map { case (a, b) => a * b }
            val result: Matrix = new DenseMatrix(matA.numRows, matA.numCols, arr.flatten.toArray)
            ((blockRowIndex, blockColIndex), result)
        }
        new BlockMatrix(newBlocks, A.rowsPerBlock, A.colsPerBlock, A.numRows(), A.numCols())
      } else {
        throw new SparkException("Cannot perform on matrices with different block dimensions")
      }
    }

    def elemWiseDivision(A: BlockMatrix, 
      B: BlockMatrix): BlockMatrix = {
      if(A.rowsPerBlock == B.rowsPerBlock && A.colsPerBlock == B.colsPerBlock) {
        val newBlocks = A.blocks.cogroup(B.blocks, GridPartitioner(B.numRowBlocks, B.numColBlocks, B.blocks.partitions.length))
        .map { case ((blockRowIndex, blockColIndex), (a, b)) =>
            val arr = new ArrayBuffer[Array[Double]]()
            val matA: DenseMatrix = toDenseMatrix(a.head)
            val matB: DenseMatrix = toDenseMatrix(b.head)
            val matAItr = matA.colIter
            val matBItr = matB.colIter
            while (matAItr.hasNext)
              arr += matAItr.next.toArray.zip(matBItr.next.toArray).map { case (a, b) => a / b }
            val result: Matrix = new DenseMatrix(matA.numRows, matA.numCols, arr.flatten.toArray)
            ((blockRowIndex, blockColIndex), result)
        }
        new BlockMatrix(newBlocks, A.rowsPerBlock, A.colsPerBlock, A.numRows(), A.numCols())
      } else {
        throw new SparkException("Cannot perform on matrices with different block dimensions")
      }
    }

    def printInfo(info: String, tok: Double, A: BlockMatrix) : Unit = {
      println(info + " Time : " + tok + " sec / Matrix Size : (" + A.numRows() + "," + A.numCols() + ") / Block Size : (" + A.numRows()/A.numRowBlocks + "," + A.numCols()/A.numColBlocks + ")")
    }
  }

  object NMF extends Serializable {
    var tick: Double = 0.0
    var tok : Double = 0.0

    def updateH(A: org.apache.spark.mllib.linalg.distributed.BlockMatrix,
        W: org.apache.spark.mllib.linalg.distributed.BlockMatrix,
        H: org.apache.spark.mllib.linalg.distributed.BlockMatrix): org.apache.spark.mllib.linalg.distributed.BlockMatrix = {
      
      tick = System.nanoTime()
      val wT = W.transpose.cache() // W^T 
      wT.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("W^T Transpose", tok, wT)
      wt += tok
      
      tick = System.nanoTime()
      // [STEP 1] x = W^T * A
      val x = wT.multiply(A)
      x.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("W^T * A", tok, x)
      wta += tok
      mm_time += tok

      tick = System.nanoTime()
      // [STEP 2] wTw = W^T * W
      val wTw = wT.multiply(W).cache()
      wTw.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("W^T * W", tok, wTw)
      wtw += tok
      mm_time += tok

      tick = System.nanoTime()
      // [STEP 3] y = W^T * W * H
      val y = wTw.multiply(H)
      y.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("W^T * W * H", tok, y)
      wtwh += tok
      mm_time += tok

      // [STEP 4] Element-wise product, divide
      tick = System.nanoTime()
      val updateH = Utils.elemWiseblock(H,x,y)
      updateH.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("Element-wise product, divid", tok, updateH)
      elemwise += tok
      
      updateH
    }

    def updateW(A: org.apache.spark.mllib.linalg.distributed.BlockMatrix,
        W: org.apache.spark.mllib.linalg.distributed.BlockMatrix,
        H: org.apache.spark.mllib.linalg.distributed.BlockMatrix): org.apache.spark.mllib.linalg.distributed.BlockMatrix = {
      
      tick = System.nanoTime()
      val hT = H.transpose.cache() // H^T 
      hT.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("H^T Transpose", tok, hT)
      ht += tok
       
      tick = System.nanoTime()
      // [STEP 1] x = A * H^T
      val x = A.multiply(hT)
      x.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("A * H^T", tok, x)
      aht += tok
      mm_time += tok

      tick = System.nanoTime()
      // [STEP 2] x = H * H^T
      val hhT = H.multiply(hT).cache()
      hhT.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("H * H^T", tok, hhT)
      hht += tok
      mm_time += tok

      tick = System.nanoTime()
      // [STEP 3] y = W * H * H^T
      val y = W.multiply(hhT)
      y.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo(" W * H * H^T", tok, y)
      whht += tok
      mm_time += tok

      // [STEP 4] Element-wise product, divide
      tick = System.nanoTime()
      //val updateW = Utils.elemWiseDivision(Utils.elemWiseProduct(W, x), y)
      val updateW = Utils.elemWiseblock(W,x,y)
      updateW.blocks.take(1)
      tok = (System.nanoTime() - tick) / 1e9
      Utils.printInfo("Element-wise product, divid", tok, updateW)
      elemwise += tok

      updateW
    }
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("MovieLensNMFBlockMatrix")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    println("[*]------- Start")
    
    //val tik = System.nanoTime()
    
    val rank = args(0).toInt
    val iteration = args(1).toInt
    val parallelism = args(2).toInt

    val p = parallelism

    var tik_init = System.nanoTime()

    val dataset = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("numPartitions", p * p).option("inferSchema", true).load("hdfs:///datasets/movielens/ratings.csv")

    val rows: RDD[Row] = dataset.rdd
    rows.cache

    val matrixEntries: RDD[MatrixEntry] = rows.map { case Row(userId:Int, movieId:Int, rating:Double) => MatrixEntry(userId, movieId, rating) }
    
    val coordMatrix = new CoordinateMatrix(matrixEntries)

    val rowsPerBlock = coordMatrix.numRows().toInt / p
    val colsPerBlock = coordMatrix.numCols().toInt / p

    val A = coordMatrix.toBlockMatrix(rowsPerBlock, colsPerBlock)
    
    val m: Int = A.numRows().toInt
    val n: Int = A.numCols().toInt
    
    println("A Matrix Size : (" + m + "," + n + ")")

    val wRowSize = m / p
    val wColSize = rank / p

    val hRowSize = rank / p
    val hColSize = n / p

    val w_rdd = sc.parallelize( {for(i <- 0 until p; j <- 0 until p) yield (i, j)}, p * p).map( coord => (coord, Matrices.rand(wRowSize, wColSize, scala.util.Random.self)))
    val h_rdd = sc.parallelize( {for(i <- 0 until p; j <- 0 until p) yield (i, j)}, p * p).map( coord => (coord, Matrices.rand(hRowSize, hColSize, scala.util.Random.self)))

    var W = new BlockMatrix(w_rdd, wRowSize, wColSize).cache()
    var H = new BlockMatrix(h_rdd, hRowSize, hColSize).cache()

    W.blocks.take(1)
    H.blocks.take(1)

    var tok = (System.nanoTime() - tik_init) / 1e9
    println("Read Dataset and Generate A, W, H Time : " + tok)

    var i = 1
    val tik = System.nanoTime()
    while (i <= iteration) {
      // [*] H Update
      println("[*]--- H Matrix Size : (" + H.numRows + "," + H.numCols + ")")
      H = NMF.updateH(A, W, H)
      
      // [*] W Update
      println("[*]--- W Matrix Size : (" + W.numRows + "," + W.numCols + ")")
      W = NMF.updateW(A, W, H)
      
      i += 1
    }
    
    var tik_validate = System.nanoTime()

    W.validate()
    H.validate()

    tok = (System.nanoTime() - tik_validate) / 1e9
    println("W, H Validate Time : " + tok)

    val latency = (System.nanoTime() - tik) / 1e9

    println("[*]------- Result")
    println("A Matrix Size : (" + A.numRows + "," + A.numCols + ")")
    println("W Matrix Size : (" + W.numRows + "," + W.numCols + ")")
    println("H Matrix Size : (" + H.numRows + "," + H.numCols + ")")
    println("[-] W^T Transpose time : " + wt + " sec")
    println("[-] W^T * A time : " + wta + " sec")
    println("[-] W^T * W time : " + wtw + " sec")
    println("[-] W^T * W * H time : " + wtwh + " sec")
    println("[-] H^T Transpose time : " + ht + " sec")
    println("[-] A * H^T time : " + aht + " sec")
    println("[-] H * H^T time : " + hht + " sec")
    println("[-] W * H * H^T time : " + whht + " sec")
    println("[-] Element-wise Multiplication Devision time : " + elemwise + " sec")
    println("[*] Total execution time  : " + latency + " sec")
    println("[*] Matrix Multiplication time : " + mm_time + " sec")
    println("[*] MM/Total : " + (mm_time / latency) * 100 + " %")
    println("[*]------- End")
  }
}
