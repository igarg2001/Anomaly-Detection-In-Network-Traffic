/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// $example on$
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
// $example off$
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

import java.util.Arrays;


/**
 * An example demonstrating k-means clustering.
 * Run with
 * <pre>
 * bin/run-example ml.JavaKMeansExample
 * </pre>
 */
public class SparkAppMain {

  public static void main(String[] args) {
    // Create a SparkSession.
    SparkSession spark = SparkSession
      .builder()
      .appName("JavaKMeansExample")
      .getOrCreate();

    // $example on$
    // Loads data.
    Dataset<Row> dataset = spark.read().option("header", false).option("inferSchema", true).csv("hdfs://localhost:9000/user/hdoop/data/kddcup.data");
    dataset = dataset.drop("_c1", "_c2", "_c3");

    String[] cols = dataset.columns();
    cols = Arrays.copyOf(cols, cols.length-1);
    System.out.println(Arrays.toString(cols));

    VectorAssembler assembler = new VectorAssembler()
            .setInputCols(cols)
            .setOutputCol("features");

    dataset = assembler.transform(dataset);

    for(int i=2; i<150; i++){
      KMeans kmeans = new KMeans().setK(i).setSeed(1L);
      KMeansModel model = kmeans.fit(dataset);

      // Make predictions
      Dataset<Row> predictions = model.transform(dataset);

      // Evaluate clustering by computing Silhouette score
      ClusteringEvaluator evaluator = new ClusteringEvaluator();

      double silhouette = evaluator.evaluate(predictions);
      System.out.println("Silhouette with squared euclidean distance = " + silhouette + " | Clusters: " + i);
      // $example off$
      // Trains a k-means model.
    }
    spark.stop();
  }
}