# dense-mm-workload
Scalable Systematic Performance Predictorfor Distributed Matrix Multiplication Kernel for Data Mining Tasks

### Dataset Download and Upload HDFS
```bash
sbt package

wget https://s3-us-west-2.amazonaws.com/spark-dev-shm/ml-latest/ratings.csv
wget https://s3-us-west-2.amazonaws.com/spark-dev-shm/mnist.bz2

hdfs dfs -mkdir -p /datasets/movielens/
hdfs dfs -put ratings.csv /datasets/movielens/ 

hdfs dfs -mkdir /datasets/mnist
hdfs dfs -put mnist.bz2 /datasets/mnist/
```


### NMF 
spark-submit --class MovieLensNMFBlockMatrix --master spark://[EC2-URL]:7077 MovieLensNMFBlockMatrix/movielensnmfblockmatrix_2.11-1.0.jar [RANK] [ITERATIONS] [PARALLELISM]

### MLP
spark-submit --class MultiLayerPerceptron --master spark://[EC2-URL]:7077 MultiLayerPerceptron/mlp_2.11-1.0.jar [ITERATIONS] [BATCH-SIZE]
