import org.apache.commons.collections.map.LinkedMap;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.clustering.*;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;
import shapeless.Tuple;

//import javax.swing.*;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
//import java.util.stream.Collectors;

class Pair<T, V> {
    T key;
    V value;
    Pair(T key, V value) {
        this.key = key;
        this.value = value;
    }
    T getKey() {
        return this.key;
    }
    V getValue() {
        return this.value;
    }
}

public class SparkAppMain {
    static <K,V extends Comparable<? super V>>
    List<Map.Entry<K, V>> entriesSortedByValues(Map<K,V> map) {

        List<Map.Entry<K,V>> sortedEntries = new ArrayList<Map.Entry<K,V>>(map.entrySet());

        Collections.sort(sortedEntries,
                new Comparator<Map.Entry<K,V>>() {
                    @Override
                    public int compare(Map.Entry<K,V> e1, Map.Entry<K,V> e2) {
                        return e2.getValue().compareTo(e1.getValue());
                    }
                }
        );

        return sortedEntries;
    }
    public static <A, B> List<Pair<A, B>> zipJava8(List<A> as, List<B> bs) {
        return IntStream.range(0, Math.min(as.size(), bs.size()))
                .mapToObj(i -> new Pair<>(as.get(i), bs.get(i)))
                .collect(Collectors.toList());
    }
    public static double distanceToCentroid(Vector datum, KMeansModel model) {
        int cluster = model.predict(datum);
        Vector[] centroid = model.clusterCenters();
        return Math.sqrt(Vectors.sqdist(datum, centroid[0]));
    }
    public static double clusteringScore(RDD<Vector> data, int k) {
        KMeans kMeans = new KMeans();
        kMeans.setK(k);
        KMeansModel model = kMeans.run(data);
        JavaDoubleRDD doubleRDD = data.toJavaRDD().mapToDouble(datum -> distanceToCentroid(datum, model));
        return doubleRDD.stats().mean();
    }
    public static void main(String[] args) throws IOException {
        SparkConf sparkConf = new SparkConf().setAppName("Anomaly Detection");
                //.setMaster("local[*]");
        JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
        JavaRDD<String> javaRDD = sparkContext.textFile("hdfs://localhost:9000/user/hdoop/data/kddcup.data");
//        JavaRDD<String> splits = javaRDD.map(r -> r.substring(r.lastIndexOf(",") + 1));
//        Map<String, Long> countByValue = splits.countByValue();
//        Map<String, Long> sortedCountByValue = new LinkedHashMap<String, Long>();
//        countByValue.entrySet().stream().sorted(Map.Entry.comparingByValue(Comparator.reverseOrder())).forEachOrdered(x -> sortedCountByValue.put(x.getKey(), x.getValue()))
        JavaPairRDD<String, Vector> labelsAndData = javaRDD.mapToPair(line -> {
//            System.out.println(line);
           List<String> ll = Arrays.asList(line.split(","));
           List<String> list = new ArrayList<String>(ll);
           list.remove(1);
           list.remove(1);
           list.remove(1);
           String label = list.remove(list.size()-1);
           double vectorArray[] = new double[list.size()];
           for(int l=0; l<list.size(); ++l) {
               vectorArray[l] = Double.parseDouble(list.get(l));
           }
           Vector dv = Vectors.dense(vectorArray);
           return new Tuple2<String, Vector>(label, dv);
                });
           JavaRDD<Vector> data =  labelsAndData.values().cache();
           KMeans kmeans = new KMeans();
           KMeansModel model = kmeans.run(data.rdd());
//           for(int i=0; i<model.clusterCenters().length; ++i) {
//               System.out.println(model.clusterCenters()[i]);
//           }
        System.out.println("------------------------------------------");
        Map<Tuple2<Integer, String>, Long> clusterLabelCountRDD = labelsAndData.mapToPair(t -> {
            String label = t._1;
            Vector datum = t._2;
            int cluster = model.predict(datum);
            return new Tuple2<Integer, String>(cluster, label);
        }).countByValue();

       // System.out.println(clusterLabelCountRDD);

        for(Tuple2<Integer, String> key: clusterLabelCountRDD.keySet()){
            System.out.println(key.toString() + " : " + clusterLabelCountRDD.get(key));
        }




    }
}
